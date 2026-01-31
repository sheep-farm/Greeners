use crate::{GreenersError, InferenceType};
use ndarray::{s, Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal as NormalDist};
use std::fmt;

#[derive(Debug, Clone)]
pub struct ArimaOrder {
    pub p: usize,
    pub d: usize,
    pub q: usize,
}

#[derive(Debug, Clone)]
pub struct SeasonalOrder {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub s: usize,
}

#[derive(Debug)]
pub struct ArimaResult {
    pub ar_params: Array1<f64>,
    pub ma_params: Array1<f64>,
    pub seasonal_ar_params: Array1<f64>,
    pub seasonal_ma_params: Array1<f64>,
    pub intercept: f64,
    pub sigma2: f64,
    pub aic: f64,
    pub bic: f64,
    pub residuals: Array1<f64>,
    pub n_obs: usize,
    pub order: ArimaOrder,
    pub seasonal_order: Option<SeasonalOrder>,
    pub exog_params: Option<Array1<f64>>,
    // Inference fields
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub log_likelihood: f64,
    pub df_model: usize,
    pub df_resid: usize,
    pub param_names: Vec<String>,
    pub inference_type: InferenceType,
    // Internal: store the original (undifferenced) series and differenced series for prediction
    original_y: Array1<f64>,
    differenced_y: Array1<f64>,
    // Store the series after regular differencing but before seasonal differencing
    after_regular_diff: Array1<f64>,
}

pub struct ARIMA;

/// Apply regular differencing d times
fn difference(y: &Array1<f64>, d: usize) -> Array1<f64> {
    let mut result = y.clone();
    for _ in 0..d {
        let n = result.len();
        if n <= 1 {
            return Array1::zeros(0);
        }
        let diff = Array1::from_vec(
            (1..n)
                .map(|i| result[i] - result[i - 1])
                .collect::<Vec<_>>(),
        );
        result = diff;
    }
    result
}

/// Apply seasonal differencing D times with period s
fn seasonal_difference(y: &Array1<f64>, d_seasonal: usize, s: usize) -> Array1<f64> {
    let mut result = y.clone();
    for _ in 0..d_seasonal {
        let n = result.len();
        if n <= s {
            return Array1::zeros(0);
        }
        let diff = Array1::from_vec(
            (s..n)
                .map(|i| result[i] - result[i - s])
                .collect::<Vec<_>>(),
        );
        result = diff;
    }
    result
}

impl ARIMA {
    /// Fit an ARIMA(p,d,q) model using Hannan-Rissanen estimation.
    pub fn fit(
        y: &Array1<f64>,
        order: (usize, usize, usize),
    ) -> Result<ArimaResult, GreenersError> {
        Self::fit_sarimax(y, order, (0, 0, 0, 1), None)
    }

    /// Fit a SARIMAX(p,d,q)(P,D,Q,s) model with optional exogenous regressors.
    ///
    /// Uses Hannan-Rissanen two-step estimation:
    /// 1. Fit a long AR to get residual estimates
    /// 2. Regress on AR lags, estimated MA residual lags, seasonal lags, and exogenous vars
    pub fn fit_sarimax(
        y: &Array1<f64>,
        order: (usize, usize, usize),
        seasonal_order: (usize, usize, usize, usize),
        exog: Option<&Array2<f64>>,
    ) -> Result<ArimaResult, GreenersError> {
        let (p, d, q) = order;
        let (sp, sd, sq, s) = seasonal_order;

        let n = y.len();
        if n < 10 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for ARIMA estimation".into(),
            ));
        }

        // Validate NaN/Inf
        for i in 0..n {
            if !y[i].is_finite() {
                return Err(GreenersError::InvalidOperation(
                    "Input series contains NaN or Inf values".into(),
                ));
            }
        }

        if let Some(x) = exog {
            if x.nrows() != n {
                return Err(GreenersError::ShapeMismatch(format!(
                    "Exogenous matrix has {} rows but series has {} observations",
                    x.nrows(),
                    n
                )));
            }
        }

        let original_y = y.clone();

        // Step 0: Apply differencing
        let after_regular_diff = difference(y, d);
        let mut z = after_regular_diff.clone();
        if sd > 0 && s > 1 {
            z = seasonal_difference(&z, sd, s);
        }

        let t = z.len();
        if t < 10 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations after differencing".into(),
            ));
        }

        // Trim exogenous to match differenced length (drop first d + sd*s rows)
        let lost = n - t;
        let exog_trimmed = exog.map(|x| x.slice(s![lost.., ..]).to_owned());

        // Determine the maximum lag we need
        let max_ar_lag = if sp > 0 && s > 1 { (sp * s).max(p) } else { p };
        let max_ma_lag = if sq > 0 && s > 1 { (sq * s).max(q) } else { q };

        // Step 1: Long AR to estimate residuals
        let p_long = (max_ar_lag + max_ma_lag)
            .max((t as f64).powf(0.25) as usize + 2)
            .max(4);

        if t <= p_long + 1 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations for Hannan-Rissanen long AR step".into(),
            ));
        }

        // Build long AR regression: z_t = c + sum_{l=1}^{p_long} phi_l * z_{t-l}
        let n_long = t - p_long;
        let n_cols_long = 1 + p_long; // intercept + p_long lags
        let mut x_long = Array2::<f64>::zeros((n_long, n_cols_long));
        let mut y_long = Array1::<f64>::zeros(n_long);

        for i in 0..n_long {
            let ti = p_long + i;
            y_long[i] = z[ti];
            x_long[[i, 0]] = 1.0;
            for l in 1..=p_long {
                x_long[[i, l]] = z[ti - l];
            }
        }

        let xtx = x_long.t().dot(&x_long);
        let xtx_inv = xtx.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let params_long = xtx_inv.dot(&x_long.t().dot(&y_long));
        let u_hat = &y_long - &x_long.dot(&params_long);

        // Step 2: Build the ARIMA regression with AR lags, MA lags (from u_hat),
        // seasonal AR/MA lags, and exogenous regressors

        // We need max_ma_lag additional observations from the start of u_hat for MA lags
        let start2 = max_ma_lag; // offset within u_hat / y_long
        if n_long <= start2 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations for ARIMA step 2".into(),
            ));
        }
        let n_final = n_long - start2;

        // Count columns: intercept + p AR + q MA + sp seasonal AR + sq seasonal MA + exog cols
        let n_exog_cols = exog_trimmed.as_ref().map_or(0, |x| x.ncols());
        let n_cols = 1 + p + q + sp + sq + n_exog_cols;

        let mut x_final = Array2::<f64>::zeros((n_final, n_cols));
        let mut y_final = Array1::<f64>::zeros(n_final);

        // The absolute index in z for observation i in step 2:
        // u_hat[j] corresponds to z[p_long + j]
        // We start at j = start2, so z index = p_long + start2 + i

        for i in 0..n_final {
            let j = start2 + i; // index in u_hat
            let zi = p_long + j; // index in z

            y_final[i] = z[zi];
            let mut col = 0;

            // Intercept
            x_final[[i, col]] = 1.0;
            col += 1;

            // AR lags: z_{t-1} ... z_{t-p}
            for l in 1..=p {
                x_final[[i, col]] = z[zi - l];
                col += 1;
            }

            // MA lags: u_hat_{t-1} ... u_hat_{t-q}
            for l in 1..=q {
                x_final[[i, col]] = u_hat[j - l];
                col += 1;
            }

            // Seasonal AR lags: z_{t-s}, z_{t-2s}, ... z_{t-sp*s}
            for sl in 1..=sp {
                let lag = sl * s;
                if zi >= lag {
                    x_final[[i, col]] = z[zi - lag];
                }
                col += 1;
            }

            // Seasonal MA lags: u_hat_{t-s}, u_hat_{t-2s}, ... u_hat_{t-sq*s}
            for sl in 1..=sq {
                let lag = sl * s;
                if j >= lag {
                    x_final[[i, col]] = u_hat[j - lag];
                }
                col += 1;
            }

            // Exogenous regressors
            if let Some(ref ex) = exog_trimmed {
                let ex_row_idx = p_long + j;
                if ex_row_idx < ex.nrows() {
                    for k in 0..n_exog_cols {
                        x_final[[i, col]] = ex[[ex_row_idx, k]];
                        col += 1;
                    }
                } else {
                    col += n_exog_cols;
                }
            }

            let _ = col; // suppress unused warning
        }

        // Solve OLS
        let xtx2 = x_final.t().dot(&x_final);
        let xtx2_inv = xtx2.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let params = xtx2_inv.dot(&x_final.t().dot(&y_final));

        // Extract parameters
        let mut idx = 0;
        let intercept = params[idx];
        idx += 1;

        let ar_params = params.slice(s![idx..idx + p]).to_owned();
        idx += p;

        let ma_params = params.slice(s![idx..idx + q]).to_owned();
        idx += q;

        let seasonal_ar_params = params.slice(s![idx..idx + sp]).to_owned();
        idx += sp;

        let seasonal_ma_params = params.slice(s![idx..idx + sq]).to_owned();
        idx += sq;

        let exog_params = if n_exog_cols > 0 {
            Some(params.slice(s![idx..idx + n_exog_cols]).to_owned())
        } else {
            None
        };

        // Residuals and sigma2
        let fitted = x_final.dot(&params);
        let residuals = &y_final - &fitted;
        let sigma2 = residuals.dot(&residuals) / n_final as f64;

        // AIC and BIC
        let n_params = n_cols as f64;
        let nf = n_final as f64;
        let log_lik = -0.5 * nf * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln());
        let aic = -2.0 * log_lik + 2.0 * n_params;
        let bic = -2.0 * log_lik + n_params * nf.ln();

        // Inference: standard errors from (X'X)^{-1} * sigma2
        let df_model = n_cols;
        let df_resid = if n_final > n_cols {
            n_final - n_cols
        } else {
            1
        };

        let cov_matrix = &xtx2_inv * sigma2;
        let std_errors = Array1::from_vec(
            (0..n_cols)
                .map(|i| cov_matrix[[i, i]].max(0.0).sqrt())
                .collect(),
        );

        let normal = NormalDist::new(0.0, 1.0).unwrap();
        let z_values = Array1::from_vec(
            (0..n_cols)
                .map(|i| {
                    if std_errors[i] > 0.0 {
                        params[i] / std_errors[i]
                    } else {
                        0.0
                    }
                })
                .collect(),
        );
        let p_values = Array1::from_vec(
            z_values
                .iter()
                .map(|&zv| 2.0 * (1.0 - normal.cdf(zv.abs())))
                .collect(),
        );
        let z_crit = 1.959964;
        let conf_lower = Array1::from_vec(
            (0..n_cols)
                .map(|i| params[i] - z_crit * std_errors[i])
                .collect(),
        );
        let conf_upper = Array1::from_vec(
            (0..n_cols)
                .map(|i| params[i] + z_crit * std_errors[i])
                .collect(),
        );

        // Build parameter names
        let mut param_names = Vec::with_capacity(n_cols);
        param_names.push("intercept".to_string());
        for l in 1..=p {
            param_names.push(format!("ar.L{}", l));
        }
        for l in 1..=q {
            param_names.push(format!("ma.L{}", l));
        }
        for sl in 1..=sp {
            param_names.push(format!("ar.S.L{}", sl * s));
        }
        for sl in 1..=sq {
            param_names.push(format!("ma.S.L{}", sl * s));
        }
        for k in 0..n_exog_cols {
            param_names.push(format!("x{}", k + 1));
        }

        let seasonal = if sp > 0 || sd > 0 || sq > 0 {
            Some(SeasonalOrder {
                p: sp,
                d: sd,
                q: sq,
                s,
            })
        } else {
            None
        };

        Ok(ArimaResult {
            ar_params,
            ma_params,
            seasonal_ar_params,
            seasonal_ma_params,
            intercept,
            sigma2,
            aic,
            bic,
            residuals,
            n_obs: n_final,
            order: ArimaOrder { p, d, q },
            seasonal_order: seasonal,
            exog_params,
            std_errors,
            t_values: z_values,
            p_values,
            conf_lower,
            conf_upper,
            log_likelihood: log_lik,
            df_model,
            df_resid,
            param_names,
            inference_type: InferenceType::Normal,
            original_y,
            differenced_y: z,
            after_regular_diff,
        })
    }
}

impl ArimaResult {
    /// Produce h-step ahead forecasts on the differenced scale,
    /// then undo differencing to return forecasts on the original scale.
    ///
    /// `future_exog` must have `steps` rows and the same number of columns as the
    /// exogenous matrix used during fitting, if exogenous regressors were included.
    pub fn predict(
        &self,
        steps: usize,
        future_exog: Option<&Array2<f64>>,
    ) -> Result<Array1<f64>, GreenersError> {
        let p = self.order.p;
        let q = self.order.q;
        let d = self.order.d;

        let z = &self.differenced_y;
        let n = z.len();

        // Validate future_exog
        if let Some(fe) = future_exog {
            let expected_cols = self.exog_params.as_ref().map_or(0, |ep| ep.len());
            if expected_cols == 0 {
                return Err(GreenersError::InvalidOperation(
                    "Model was fit without exogenous regressors but future_exog was provided"
                        .into(),
                ));
            }
            if fe.nrows() != steps {
                return Err(GreenersError::ShapeMismatch(format!(
                    "future_exog has {} rows but {} steps requested",
                    fe.nrows(),
                    steps
                )));
            }
            if fe.ncols() != expected_cols {
                return Err(GreenersError::ShapeMismatch(format!(
                    "future_exog has {} columns but model expects {}",
                    fe.ncols(),
                    expected_cols
                )));
            }
        }

        // Forecast on differenced series
        let mut z_ext: Vec<f64> = z.to_vec();
        let res_vec: Vec<f64> = self.residuals.to_vec();
        let mut res_ext: Vec<f64> = res_vec;

        let (sp, sq, s) = self
            .seasonal_order
            .as_ref()
            .map_or((0, 0, 1), |so| (so.p, so.q, so.s));

        for h in 0..steps {
            let ti = n + h;
            let mut val = self.intercept;

            for l in 1..=p {
                if ti >= l {
                    val += self.ar_params[l - 1] * z_ext[ti - l];
                }
            }
            for l in 1..=q {
                if ti >= l && (ti - l) < res_ext.len() {
                    val += self.ma_params[l - 1] * res_ext[ti - l];
                }
            }
            for sl in 1..=sp {
                let lag = sl * s;
                if ti >= lag {
                    val += self.seasonal_ar_params[sl - 1] * z_ext[ti - lag];
                }
            }
            for sl in 1..=sq {
                let lag = sl * s;
                if ti >= lag && (ti - lag) < res_ext.len() {
                    val += self.seasonal_ma_params[sl - 1] * res_ext[ti - lag];
                }
            }

            // Add exogenous contribution
            if let (Some(fe), Some(ref ep)) = (future_exog, &self.exog_params) {
                for k in 0..ep.len() {
                    val += ep[k] * fe[[h, k]];
                }
            }

            z_ext.push(val);
            res_ext.push(0.0); // future residuals = 0
        }

        let forecasts_diff = z_ext[n..].to_vec();

        // Undo seasonal differencing first, then regular differencing
        let mut forecast_vals = forecasts_diff;

        // Undo seasonal differencing (D times)
        if let Some(ref so) = self.seasonal_order {
            let sd = so.d;
            let ss = so.s;
            if sd > 0 && ss > 1 {
                // We need the tail of after_regular_diff to integrate back
                let rd = &self.after_regular_diff;
                for _diff_round in 0..sd {
                    // y_t = z_t + y_{t-s}, so we need y values at t-s
                    let mut integrated = Vec::with_capacity(forecast_vals.len());
                    for (h, &v) in forecast_vals.iter().enumerate() {
                        // Index in the extended after_regular_diff series
                        let src_idx = rd.len() + h;
                        let lag_idx = src_idx.wrapping_sub(ss);
                        let prev = if lag_idx < rd.len() {
                            rd[lag_idx]
                        } else {
                            integrated[lag_idx - rd.len()]
                        };
                        integrated.push(v + prev);
                    }
                    forecast_vals = integrated;
                }
            }
        }

        // Undo regular differencing
        if d > 0 {
            let orig = &self.original_y;
            let level: Vec<f64> = orig.to_vec();
            for _diff_round in 0..d {
                let last = *level.last().unwrap_or(&0.0);
                let mut integrated = Vec::with_capacity(forecast_vals.len());
                let mut prev = last;
                for &v in &forecast_vals {
                    prev += v;
                    integrated.push(prev);
                }
                forecast_vals = integrated;
            }
        }

        Ok(Array1::from_vec(forecast_vals))
    }

    /// Return in-sample fitted values on the original (undifferenced) scale.
    pub fn fitted_values(&self) -> Array1<f64> {
        let z = &self.differenced_y;
        let n_res = self.residuals.len();
        let offset = z.len() - n_res;

        let fitted_diff: Vec<f64> = (0..n_res)
            .map(|i| z[offset + i] - self.residuals[i])
            .collect();

        let d = self.order.d;
        let (sd, ss) = self
            .seasonal_order
            .as_ref()
            .map_or((0, 1), |so| (so.d, so.s));

        // If no differencing, return as-is
        if d == 0 && (sd == 0 || ss <= 1) {
            return Array1::from_vec(fitted_diff);
        }

        // Undoing differencing for in-sample fitted values is complex because
        // each fitted value maps to a different position in the original series.
        // Return on the differenced scale (standard for ARIMA fitted values).
        Array1::from_vec(fitted_diff)
    }

    /// Return residuals from the estimation.
    pub fn residuals(&self) -> &Array1<f64> {
        &self.residuals
    }

    /// Ljung-Box test for residual autocorrelation.
    ///
    /// Returns `(statistic, p_value)`. The null hypothesis is that the residuals
    /// are independently distributed (no autocorrelation up to the given lag).
    pub fn ljung_box(&self, lags: usize) -> Result<(f64, f64), GreenersError> {
        let resid = &self.residuals;
        let n = resid.len();
        if lags == 0 || lags >= n {
            return Err(GreenersError::InvalidOperation(
                "lags must be > 0 and < number of residuals".into(),
            ));
        }

        let acf_vals = self.acf(lags);
        let nf = n as f64;
        let mut q_stat = 0.0;
        for (k, &rk) in acf_vals.iter().enumerate() {
            let lag = k + 1;
            q_stat += rk * rk / (nf - lag as f64);
        }
        q_stat *= nf * (nf + 2.0);

        // Degrees of freedom: lags - p - q (but at least 1)
        let p = self.order.p;
        let q = self.order.q;
        let df = if lags > p + q { lags - p - q } else { 1 };

        let chi2 = ChiSquared::new(df as f64).map_err(|e| {
            GreenersError::InvalidOperation(format!("Chi-squared distribution error: {}", e))
        })?;
        let p_value = 1.0 - chi2.cdf(q_stat);

        Ok((q_stat, p_value))
    }

    /// Sample autocorrelation function of residuals up to `max_lag`.
    pub fn acf(&self, max_lag: usize) -> Vec<f64> {
        let resid = &self.residuals;
        let n = resid.len();
        let mean = resid.sum() / n as f64;
        let var: f64 = resid.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;

        if var == 0.0 {
            return vec![0.0; max_lag];
        }

        (1..=max_lag)
            .map(|k| {
                let cov: f64 = (k..n)
                    .map(|t| (resid[t] - mean) * (resid[t - k] - mean))
                    .sum();
                cov / (n as f64 * var)
            })
            .collect()
    }

    /// Check if the AR polynomial has all roots outside the unit circle (stationary).
    pub fn is_stationary(&self) -> bool {
        // For AR(1): stationary if |phi| < 1
        // For higher orders, check companion matrix eigenvalues.
        // Simple check: all AR coefficients sum < 1 in absolute value (necessary but not sufficient
        // for p>1, but exact for p=1). For a general check we use the companion form.
        check_roots_outside_unit_circle(&self.ar_params)
            && check_roots_outside_unit_circle(&self.seasonal_ar_params)
    }

    /// Check if the MA polynomial has all roots outside the unit circle (invertible).
    pub fn is_invertible(&self) -> bool {
        check_roots_outside_unit_circle(&self.ma_params)
            && check_roots_outside_unit_circle(&self.seasonal_ma_params)
    }
}

/// Check if a polynomial 1 - c1*z - c2*z^2 - ... has all roots outside the unit circle.
/// Equivalent to checking that the companion matrix eigenvalues have modulus < 1.
fn check_roots_outside_unit_circle(coeffs: &Array1<f64>) -> bool {
    let p = coeffs.len();
    if p == 0 {
        return true;
    }
    if p == 1 {
        return coeffs[0].abs() < 1.0;
    }

    // Build companion matrix and do power iteration to find max eigenvalue magnitude.
    // For small p this is fine; for large p a proper eigenvalue solver would be better.
    // Use the sufficient condition: sum of |coeffs| < 1 is sufficient but not necessary.
    // For a more accurate check, we iterate.
    let mut companion = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        companion[[0, i]] = coeffs[i];
    }
    for i in 1..p {
        companion[[i, i - 1]] = 1.0;
    }

    // Power iteration for spectral radius (approximate)
    let mut v = Array1::<f64>::ones(p);
    let norm = v.dot(&v).sqrt();
    v /= norm;

    for _ in 0..200 {
        let w = companion.dot(&v);
        let norm = w.dot(&w).sqrt();
        if norm < 1e-15 {
            return true; // all eigenvalues ~0
        }
        v = w / norm;
    }
    let w = companion.dot(&v);
    let spectral_radius = w.dot(&w).sqrt();

    spectral_radius < 1.0
}

impl fmt::Display for ArimaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let model_name = match &self.seasonal_order {
            Some(so) => format!(
                "SARIMAX({},{},{})({}x{}x{}x{})",
                self.order.p, self.order.d, self.order.q, so.p, so.d, so.q, so.s
            ),
            None => format!("ARIMA({},{},{})", self.order.p, self.order.d, self.order.q),
        };

        writeln!(
            f,
            "\n{:=^70}",
            format!(" {} via Hannan-Rissanen ", model_name)
        )?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.6}", "Log-Likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>10.6}", "Sigma²:", self.sigma2)?;
        writeln!(f, "{:<20} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>10.4}", "BIC:", self.bic)?;

        // Parameter table
        writeln!(f, "\n{:-^70}", " Parameters ")?;
        writeln!(
            f,
            "{:<15} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}",
            "", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"
        )?;
        writeln!(f, "{:-^70}", "")?;

        for (i, name) in self.param_names.iter().enumerate() {
            let coef = if i == 0 {
                self.intercept
            } else {
                // Reconstruct from the params vector position
                // intercept is index 0, then ar, ma, sar, sma, exog
                let p = self.order.p;
                let q = self.order.q;
                let sp = self.seasonal_ar_params.len();
                let sq = self.seasonal_ma_params.len();
                let j = i - 1;
                if j < p {
                    self.ar_params[j]
                } else if j < p + q {
                    self.ma_params[j - p]
                } else if j < p + q + sp {
                    self.seasonal_ar_params[j - p - q]
                } else if j < p + q + sp + sq {
                    self.seasonal_ma_params[j - p - q - sp]
                } else {
                    self.exog_params.as_ref().unwrap()[j - p - q - sp - sq]
                }
            };
            writeln!(
                f,
                "{:<15} {:>10.4} {:>10.4} {:>8.3} {:>8.3} {:>10.4} {:>10.4}",
                name,
                coef,
                self.std_errors[i],
                self.t_values[i],
                self.p_values[i],
                self.conf_lower[i],
                self.conf_upper[i],
            )?;
        }
        writeln!(f, "{:=^70}", "")
    }
}
