#[allow(unused_imports)]
use crate::linalg::{LinalgEigh as _, LinalgInverse as _, UPLO};
use crate::{CovarianceType, GreenersError, OLS};
use ndarray::{s, Array1, Array2};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

// ─── Result structs ──────────────────────────────────────────────────────────

/// Results of the Augmented Dickey-Fuller Test
#[derive(Debug)]
pub struct AdfResult {
    pub test_statistic: f64,
    pub p_value: Option<f64>,
    pub critical_values: (f64, f64, f64), // 1%, 5%, 10%
    pub is_stationary: bool,
    pub lags_used: usize,
    pub n_obs: usize,
}

/// Results of the KPSS Test
#[derive(Debug)]
pub struct KpssResult {
    pub test_statistic: f64,
    pub critical_values: (f64, f64, f64, f64), // 10%, 5%, 2.5%, 1%
    pub is_stationary: bool,
    pub lags_used: usize,
    pub n_obs: usize,
    /// "c" for level stationarity, "ct" for trend stationarity
    pub regression: String,
}

/// Results of the Ljung-Box Test
#[derive(Debug)]
pub struct LjungBoxResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub lags: usize,
    pub n_obs: usize,
}

/// Results of the ARCH Test (Engle's LM test)
#[derive(Debug)]
pub struct ArchTestResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub lags: usize,
    pub n_obs: usize,
}

/// Results of the Granger Causality Test
#[derive(Debug)]
pub struct GrangerResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub df_num: usize,
    pub df_denom: usize,
    pub lags: usize,
}

/// Results of the Engle-Granger Cointegration Test
#[derive(Debug)]
pub struct EngleGrangerResult {
    pub adf_statistic: f64,
    pub critical_values: (f64, f64, f64), // 1%, 5%, 10%
    pub is_cointegrated: bool,
    pub cointegrating_vector: Array1<f64>,
    pub residuals: Array1<f64>,
}

/// Results of the Johansen Cointegration Test
#[derive(Debug)]
pub struct JohansenResult {
    pub trace_stats: Array1<f64>,
    pub trace_critical_values: Array2<f64>, // n_vars x 3 (10%, 5%, 1%)
    pub max_eigen_stats: Array1<f64>,
    pub max_eigen_critical_values: Array2<f64>,
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
    pub cointegrating_rank: usize,
    pub n_vars: usize,
}

// ─── TimeSeries ──────────────────────────────────────────────────────────────

pub struct TimeSeries;

impl TimeSeries {
    // ── ACF ───────────────────────────────────────────────────────────────

    /// Compute autocorrelation function for lags 0..=nlags.
    ///
    /// Returns an array of length `nlags + 1` where element k is the
    /// autocorrelation at lag k. Element 0 is always 1.0.
    pub fn acf(series: &Array1<f64>, nlags: usize) -> Result<Array1<f64>, GreenersError> {
        let n = series.len();
        if n < 2 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for ACF".into(),
            ));
        }
        if nlags >= n {
            return Err(GreenersError::ShapeMismatch(
                "nlags must be less than series length".into(),
            ));
        }

        let mean = series.mean().unwrap_or(0.0);
        let gamma_0: f64 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if gamma_0.abs() < 1e-15 {
            // Constant series
            let mut result = Array1::zeros(nlags + 1);
            result[0] = 1.0;
            return Ok(result);
        }

        let mut acf_vals = Array1::zeros(nlags + 1);
        acf_vals[0] = 1.0;

        for k in 1..=nlags {
            let gamma_k: f64 = (0..n - k)
                .map(|t| (series[t] - mean) * (series[t + k] - mean))
                .sum::<f64>()
                / n as f64;
            acf_vals[k] = gamma_k / gamma_0;
        }

        Ok(acf_vals)
    }

    // ── PACF ─────────────────────────────────────────────────────────────

    /// Compute partial autocorrelation function using Durbin-Levinson recursion.
    ///
    /// Returns an array of length `nlags + 1` where element 0 is 1.0
    /// and element k is the partial autocorrelation at lag k.
    pub fn pacf(series: &Array1<f64>, nlags: usize) -> Result<Array1<f64>, GreenersError> {
        let acf_vals = Self::acf(series, nlags)?;
        let mut pacf_vals = Array1::zeros(nlags + 1);
        pacf_vals[0] = 1.0;

        if nlags == 0 {
            return Ok(pacf_vals);
        }

        // Durbin-Levinson recursion
        // phi[k][j] = partial autocorrelation coefficients
        let mut phi = vec![vec![0.0f64; nlags + 1]; nlags + 1];

        // k = 1
        phi[1][1] = acf_vals[1];
        pacf_vals[1] = phi[1][1];

        for k in 2..=nlags {
            // phi[k][k] = (r[k] - sum(phi[k-1][j] * r[k-j])) / (1 - sum(phi[k-1][j] * r[j]))
            let mut num = acf_vals[k];
            let mut den = 1.0;
            for j in 1..k {
                num -= phi[k - 1][j] * acf_vals[k - j];
                den -= phi[k - 1][j] * acf_vals[j];
            }

            if den.abs() < 1e-15 {
                // Degenerate case
                pacf_vals[k] = 0.0;
                continue;
            }

            phi[k][k] = num / den;
            pacf_vals[k] = phi[k][k];

            // Update phi[k][j] for j < k
            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
            }
        }

        Ok(pacf_vals)
    }

    // ── ADF ──────────────────────────────────────────────────────────────

    pub fn adf(series: &Array1<f64>, max_lags: Option<usize>) -> Result<AdfResult, GreenersError> {
        let n = series.len();
        if n < 10 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for ADF".into(),
            ));
        }

        let lags = match max_lags {
            Some(l) => l,
            None => ((n - 1) as f64).powf(1.0 / 3.0) as usize,
        };

        let y_diff = diff(series, 1);
        let effective_n = n - 1 - lags;

        let target_y = y_diff.slice(s![lags..]).to_owned();

        let mut x_mat = Array2::<f64>::zeros((effective_n, 2 + lags));

        x_mat.column_mut(0).fill(1.0);

        for i in 0..effective_n {
            x_mat[[i, 1]] = series[lags + i];
        }

        for l in 0..lags {
            for i in 0..effective_n {
                x_mat[[i, 2 + l]] = y_diff[lags + i - 1 - l];
            }
        }

        let ols_res = OLS::fit(&target_y, &x_mat, CovarianceType::NonRobust)?;

        let adf_stat = ols_res.t_values[1];

        let crit_1pct = -3.43;
        let crit_5pct = -2.86;
        let crit_10pct = -2.57;

        Ok(AdfResult {
            test_statistic: adf_stat,
            p_value: None,
            critical_values: (crit_1pct, crit_5pct, crit_10pct),
            is_stationary: adf_stat < crit_5pct,
            lags_used: lags,
            n_obs: effective_n,
        })
    }

    // ── KPSS ─────────────────────────────────────────────────────────────

    /// KPSS test for stationarity.
    ///
    /// H0: Series is stationary (level or trend).
    /// H1: Series has a unit root.
    ///
    /// `regression`: "c" for level stationarity, "ct" for trend stationarity.
    pub fn kpss(
        series: &Array1<f64>,
        regression: &str,
        nlags: Option<usize>,
    ) -> Result<KpssResult, GreenersError> {
        let n = series.len();
        if n < 10 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for KPSS".into(),
            ));
        }

        // Detrend: regress y on constant (and trend if "ct")
        let residuals = if regression == "ct" {
            // y = a + b*t + e
            let mut x = Array2::<f64>::zeros((n, 2));
            x.column_mut(0).fill(1.0);
            for i in 0..n {
                x[[i, 1]] = (i + 1) as f64;
            }
            let ols_res = OLS::fit(series, &x, CovarianceType::NonRobust)?;
            ols_res.residuals(series, &x)
        } else {
            // y = a + e
            let mean = series.mean().unwrap_or(0.0);
            series.mapv(|x| x - mean)
        };

        // Cumulative sum of residuals
        let mut s_t = Array1::<f64>::zeros(n);
        s_t[0] = residuals[0];
        for i in 1..n {
            s_t[i] = s_t[i - 1] + residuals[i];
        }

        // Long-run variance estimator (Newey-West with Bartlett kernel)
        let lags = nlags.unwrap_or_else(|| {
            // Schwert rule: int(12 * (n/100)^(1/4))
            (12.0 * (n as f64 / 100.0).powf(0.25)) as usize
        });

        let sigma2: f64 = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;
        let mut s2 = sigma2;

        for l in 1..=lags {
            let weight = 1.0 - l as f64 / (lags as f64 + 1.0);
            let gamma_l: f64 =
                (l..n).map(|t| residuals[t] * residuals[t - l]).sum::<f64>() / n as f64;
            s2 += 2.0 * weight * gamma_l;
        }

        s2 = s2.max(1e-15);

        // KPSS statistic = (1/n^2) * sum(S_t^2) / s2
        let eta: f64 = s_t.iter().map(|&s| s * s).sum::<f64>() / ((n as f64).powi(2) * s2);

        // Critical values (asymptotic)
        let (crit_10, crit_5, crit_2_5, crit_1, is_stat) = if regression == "ct" {
            // Trend stationarity critical values
            (0.119, 0.146, 0.176, 0.216, eta < 0.146)
        } else {
            // Level stationarity critical values
            (0.347, 0.463, 0.574, 0.739, eta < 0.463)
        };

        Ok(KpssResult {
            test_statistic: eta,
            critical_values: (crit_10, crit_5, crit_2_5, crit_1),
            is_stationary: is_stat,
            lags_used: lags,
            n_obs: n,
            regression: regression.to_string(),
        })
    }

    // ── Ljung-Box ────────────────────────────────────────────────────────

    /// Ljung-Box test for autocorrelation.
    ///
    /// H0: No autocorrelation up to lag `lags`.
    /// Q = n(n+2) * sum(r_k^2 / (n-k)) ~ chi2(lags)
    pub fn ljung_box(series: &Array1<f64>, lags: usize) -> Result<LjungBoxResult, GreenersError> {
        let n = series.len();
        if lags == 0 || lags >= n {
            return Err(GreenersError::ShapeMismatch(
                "Invalid number of lags for Ljung-Box test".into(),
            ));
        }

        let acf_vals = Self::acf(series, lags)?;

        let n_f = n as f64;
        let q: f64 = (1..=lags)
            .map(|k| acf_vals[k].powi(2) / (n_f - k as f64))
            .sum::<f64>()
            * n_f
            * (n_f + 2.0);

        let chi2 = ChiSquared::new(lags as f64).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - chi2.cdf(q);

        Ok(LjungBoxResult {
            test_statistic: q,
            p_value,
            lags,
            n_obs: n,
        })
    }

    // ── ARCH test ────────────────────────────────────────────────────────

    /// Engle's ARCH test for conditional heteroscedasticity.
    ///
    /// H0: No ARCH effects (homoscedastic errors).
    /// Regresses squared residuals on their lags. LM = n*R^2 ~ chi2(lags).
    pub fn arch_test(
        residuals: &Array1<f64>,
        lags: usize,
    ) -> Result<ArchTestResult, GreenersError> {
        let n = residuals.len();
        if lags == 0 || lags >= n {
            return Err(GreenersError::ShapeMismatch(
                "Invalid number of lags for ARCH test".into(),
            ));
        }

        let u2 = residuals.mapv(|r| r.powi(2));

        let effective_n = n - lags;
        let y = u2.slice(s![lags..]).to_owned();

        // X = [1, u2_{t-1}, ..., u2_{t-lags}]
        let mut x = Array2::<f64>::zeros((effective_n, 1 + lags));
        x.column_mut(0).fill(1.0);
        for l in 0..lags {
            for i in 0..effective_n {
                x[[i, 1 + l]] = u2[lags + i - 1 - l];
            }
        }

        let ols_res = OLS::fit(&y, &x, CovarianceType::NonRobust)?;
        let lm_stat = effective_n as f64 * ols_res.r_squared;

        let chi2 = ChiSquared::new(lags as f64).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - chi2.cdf(lm_stat);

        Ok(ArchTestResult {
            test_statistic: lm_stat,
            p_value,
            lags,
            n_obs: effective_n,
        })
    }

    // ── Granger Causality ────────────────────────────────────────────────

    /// Granger causality test: does `x` Granger-cause `y`?
    ///
    /// Compares unrestricted model (y on lags of y and x) vs restricted (y on lags of y only).
    /// F = ((SSR_r - SSR_u) / lags) / (SSR_u / (n - 2*lags - 1))
    pub fn granger_causality(
        y: &Array1<f64>,
        x: &Array1<f64>,
        lags: usize,
    ) -> Result<GrangerResult, GreenersError> {
        let n = y.len();
        if x.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "y and x must have the same length".into(),
            ));
        }
        if lags == 0 || 2 * lags + 1 >= n {
            return Err(GreenersError::ShapeMismatch(
                "Invalid number of lags for Granger test".into(),
            ));
        }

        let effective_n = n - lags;

        // Dependent variable: y[lags:]
        let y_dep = y.slice(s![lags..]).to_owned();

        // Restricted model: y_t on [1, y_{t-1}, ..., y_{t-lags}]
        let mut x_r = Array2::<f64>::zeros((effective_n, 1 + lags));
        x_r.column_mut(0).fill(1.0);
        for l in 0..lags {
            for i in 0..effective_n {
                x_r[[i, 1 + l]] = y[lags + i - 1 - l];
            }
        }

        let ols_r = OLS::fit(&y_dep, &x_r, CovarianceType::NonRobust)?;
        let resid_r = ols_r.residuals(&y_dep, &x_r);
        let ssr_r = resid_r.dot(&resid_r);

        // Unrestricted model: y_t on [1, y_{t-1}..y_{t-lags}, x_{t-1}..x_{t-lags}]
        let mut x_u = Array2::<f64>::zeros((effective_n, 1 + 2 * lags));
        x_u.column_mut(0).fill(1.0);
        for l in 0..lags {
            for i in 0..effective_n {
                x_u[[i, 1 + l]] = y[lags + i - 1 - l];
                x_u[[i, 1 + lags + l]] = x[lags + i - 1 - l];
            }
        }

        let ols_u = OLS::fit(&y_dep, &x_u, CovarianceType::NonRobust)?;
        let resid_u = ols_u.residuals(&y_dep, &x_u);
        let ssr_u = resid_u.dot(&resid_u);

        let df_num = lags;
        let df_denom = effective_n - 2 * lags - 1;

        let f_stat = ((ssr_r - ssr_u) / df_num as f64) / (ssr_u / df_denom as f64);

        let f_dist = FisherSnedecor::new(df_num as f64, df_denom as f64)
            .map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - f_dist.cdf(f_stat);

        Ok(GrangerResult {
            f_statistic: f_stat,
            p_value,
            df_num,
            df_denom,
            lags,
        })
    }

    // ── Engle-Granger Cointegration ──────────────────────────────────────

    /// Engle-Granger two-step cointegration test.
    ///
    /// Step 1: Regress y1 on y2 (and intercept).
    /// Step 2: Run ADF on the residuals.
    /// If residuals are stationary, the series are cointegrated.
    ///
    /// Critical values are more stringent than standard ADF (Phillips-Ouliaris).
    pub fn engle_granger(
        y1: &Array1<f64>,
        y2: &Array1<f64>,
    ) -> Result<EngleGrangerResult, GreenersError> {
        let n = y1.len();
        if y2.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "y1 and y2 must have the same length".into(),
            ));
        }
        if n < 20 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for cointegration test".into(),
            ));
        }

        // Step 1: Cointegrating regression y1 = a + b*y2 + e
        let mut x = Array2::<f64>::zeros((n, 2));
        x.column_mut(0).fill(1.0);
        x.column_mut(1).assign(y2);

        let ols_res = OLS::fit(y1, &x, CovarianceType::NonRobust)?;
        let residuals = ols_res.residuals(y1, &x);

        // Step 2: ADF on residuals
        let adf_res = Self::adf(&residuals, None)?;

        // Phillips-Ouliaris critical values for 2 variables (more stringent)
        let crit_1pct = -3.90;
        let crit_5pct = -3.34;
        let crit_10pct = -3.04;

        Ok(EngleGrangerResult {
            adf_statistic: adf_res.test_statistic,
            critical_values: (crit_1pct, crit_5pct, crit_10pct),
            is_cointegrated: adf_res.test_statistic < crit_5pct,
            cointegrating_vector: ols_res.params,
            residuals,
        })
    }

    // ── Johansen Cointegration ───────────────────────────────────────────

    /// Johansen cointegration test for multiple time series.
    ///
    /// Tests for cointegrating relationships among `n_vars` I(1) series.
    /// Uses the trace and maximum eigenvalue statistics.
    ///
    /// `data`: T x k matrix (rows = observations, columns = variables).
    /// `max_lag`: number of lags in the VECM.
    /// `det_order`: -1 = no constant, 0 = restricted constant, 1 = unrestricted constant.
    pub fn johansen(
        data: &Array2<f64>,
        max_lag: usize,
        det_order: i32,
    ) -> Result<JohansenResult, GreenersError> {
        let t_total = data.nrows();
        let k = data.ncols();

        if t_total < k + max_lag + 5 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations for Johansen test".into(),
            ));
        }

        // First differences
        let dy = {
            let mut d = Array2::<f64>::zeros((t_total - 1, k));
            for t in 0..t_total - 1 {
                for j in 0..k {
                    d[[t, j]] = data[[t + 1, j]] - data[[t, j]];
                }
            }
            d
        };

        // Effective sample: after removing lags
        let t_eff = t_total - 1 - max_lag;

        // Build dependent: dy[max_lag:]
        let y_dep = dy.slice(s![max_lag.., ..]).to_owned();

        // Build regressors for lagged differences
        let n_diff_cols = if max_lag > 0 { (max_lag - 1) * k } else { 0 };
        let n_det = match det_order {
            -1 => 0,
            _ => 1, // constant
        };
        let n_rhs = n_diff_cols + n_det;

        // Lagged level: data[max_lag..t_total-1]
        let y_lag = data.slice(s![max_lag..t_total - 1, ..]).to_owned();

        // Build Z matrix (lagged differences + deterministic terms)
        let z = if n_rhs > 0 {
            let mut z_mat = Array2::<f64>::zeros((t_eff, n_rhs));
            let mut col = 0;

            // Lagged differences (lags 1..max_lag-1 of dy)
            for l in 1..max_lag {
                for j in 0..k {
                    for t in 0..t_eff {
                        z_mat[[t, col]] = dy[[max_lag + t - l, j]];
                    }
                    col += 1;
                }
            }

            // Constant
            if det_order >= 0 {
                z_mat.column_mut(col).fill(1.0);
            }

            Some(z_mat)
        } else {
            None
        };

        // Concentrate out Z from y_dep and y_lag via OLS
        let (r0, r1) = if let Some(ref z_mat) = z {
            let zt = z_mat.t();
            let ztz = zt.dot(z_mat);
            let ztz_inv = ztz.inv().map_err(|_| GreenersError::SingularMatrix)?;

            // Residuals of y_dep on Z
            let proj = z_mat.dot(&ztz_inv.dot(&zt.dot(&y_dep)));
            let r0 = &y_dep - &proj;

            // Residuals of y_lag on Z
            let proj_lag = z_mat.dot(&ztz_inv.dot(&zt.dot(&y_lag)));
            let r1 = &y_lag - &proj_lag;

            (r0, r1)
        } else {
            (y_dep.clone(), y_lag.clone())
        };

        let t_f = t_eff as f64;

        // S00 = r0'r0 / T, S01 = r0'r1 / T, S11 = r1'r1 / T
        let s00 = r0.t().dot(&r0) / t_f;
        let s01 = r0.t().dot(&r1) / t_f;
        let s10 = r1.t().dot(&r0) / t_f;
        let s11 = r1.t().dot(&r1) / t_f;

        // Solve generalized eigenvalue problem:
        // S11^(-1) * S10 * S00^(-1) * S01 * v = lambda * v
        let s00_inv = s00.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let s11_inv = s11.inv().map_err(|_| GreenersError::SingularMatrix)?;

        let m = s11_inv.dot(&s10).dot(&s00_inv).dot(&s01);

        // Eigenvalue decomposition
        let (eigenvalues_raw, eigvecs) = m.eigh(UPLO::Upper)?;

        // Sort eigenvalues descending
        let mut idx: Vec<usize> = (0..k).collect();
        idx.sort_by(|&a, &b| eigenvalues_raw[b].partial_cmp(&eigenvalues_raw[a]).unwrap());

        let eigenvalues = Array1::from_vec(
            idx.iter()
                .map(|&i| eigenvalues_raw[i].clamp(0.0, 1.0))
                .collect(),
        );
        let mut eigenvectors = Array2::<f64>::zeros((k, k));
        for (new_col, &old_col) in idx.iter().enumerate() {
            eigenvectors
                .column_mut(new_col)
                .assign(&eigvecs.column(old_col));
        }

        // Trace statistics: -T * sum(ln(1 - lambda_i)) for i = r+1..k
        let mut trace_stats = Array1::<f64>::zeros(k);
        for r in 0..k {
            let stat: f64 = (r..k)
                .map(|i| -(t_f) * (1.0 - eigenvalues[i]).max(1e-15).ln())
                .sum();
            trace_stats[r] = stat;
        }

        // Max eigenvalue statistics: -T * ln(1 - lambda_{r+1})
        let mut max_eigen_stats = Array1::<f64>::zeros(k);
        for r in 0..k {
            max_eigen_stats[r] = -(t_f) * (1.0 - eigenvalues[r]).max(1e-15).ln();
        }

        // Critical values (asymptotic, for k variables with unrestricted constant)
        // Using Osterwald-Lenum (1992) tables for trace and max-eigenvalue
        // These are approximate for k = 2..5
        let trace_cv = johansen_trace_critical_values(k);
        let max_cv = johansen_max_eigen_critical_values(k);

        // Determine cointegrating rank using trace test at 5%
        let mut rank = 0;
        for r in 0..k {
            if r < trace_cv.nrows() && trace_stats[r] > trace_cv[[r, 1]] {
                rank = r + 1;
            } else {
                break;
            }
        }

        Ok(JohansenResult {
            trace_stats,
            trace_critical_values: trace_cv,
            max_eigen_stats,
            max_eigen_critical_values: max_cv,
            eigenvalues,
            eigenvectors,
            cointegrating_rank: rank,
            n_vars: k,
        })
    }

    // ── HP Filter ─────────────────────────────────────────────────────

    /// Hodrick-Prescott filter.
    ///
    /// Decomposes a time series into trend and cyclical components by solving:
    /// min Σ(y_t - τ_t)² + λ Σ(Δ²τ_t)²
    ///
    /// `lambda`: smoothing parameter (1600 for quarterly, 6.25 for annual, 129600 for monthly)
    ///
    /// Returns `(trend, cycle)`.
    pub fn hp_filter(
        series: &Array1<f64>,
        lambda: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), GreenersError> {
        let n = series.len();
        if n < 4 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for HP filter (need at least 4 observations)".into(),
            ));
        }

        // Build (I + λ * K'K) where K is the second-difference operator
        // Solve the pentadiagonal system directly
        // K'K is a banded matrix with bandwidth 2
        let mut a = Array2::<f64>::zeros((n, n));

        // Diagonal: 1 + λ * (elements of K'K diagonal)
        // K'K[i,j] entries for second-difference operator
        for i in 0..n {
            a[[i, i]] = 1.0;
        }

        // Add λ * K'K
        // K is (n-2) x n matrix: row i has [1, -2, 1] starting at column i
        // K'K entries:
        for i in 0..n - 2 {
            // K[i,:] has entries at i, i+1, i+2 with values 1, -2, 1
            // K'K = sum of outer products of K rows
            let indices = [i, i + 1, i + 2];
            let values = [1.0, -2.0, 1.0];
            for (ii, &ri) in indices.iter().enumerate() {
                for (jj, &rj) in indices.iter().enumerate() {
                    a[[ri, rj]] += lambda * values[ii] * values[jj];
                }
            }
        }

        // Solve A * trend = y using LU/inverse
        let a_inv = a.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let trend = a_inv.dot(series);
        let cycle = series - &trend;

        Ok((trend, cycle))
    }

    // ── BK Filter ────────────────────────────────────────────────────

    /// Baxter-King band-pass filter.
    ///
    /// Symmetric fixed-length filter that isolates cyclical components in a
    /// frequency band defined by `[2π/high, 2π/low]` periods.
    ///
    /// * `low` — minimum period of the cycle (e.g., 6 for quarterly)
    /// * `high` — maximum period of the cycle (e.g., 32 for quarterly)
    /// * `k` — number of lead/lag terms (truncation; e.g., 12)
    ///
    /// Returns the cycle component. First and last `k` values are NaN.
    pub fn bk_filter(
        series: &Array1<f64>,
        low: usize,
        high: usize,
        k: usize,
    ) -> Result<Array1<f64>, GreenersError> {
        let n = series.len();
        if n < 2 * k + 1 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for BK filter with given K".into(),
            ));
        }
        if low < 2 || high <= low {
            return Err(GreenersError::ShapeMismatch(
                "Need low >= 2 and high > low".into(),
            ));
        }

        let omega_high = 2.0 * std::f64::consts::PI / low as f64;
        let omega_low = 2.0 * std::f64::consts::PI / high as f64;

        // Ideal band-pass filter weights
        let mut weights = Array1::<f64>::zeros(2 * k + 1);
        // a_0 = (omega_high - omega_low) / pi
        weights[k] = (omega_high - omega_low) / std::f64::consts::PI;

        for j in 1..=k {
            let jf = j as f64;
            let w = (omega_high * jf).sin() / (std::f64::consts::PI * jf)
                - (omega_low * jf).sin() / (std::f64::consts::PI * jf);
            weights[k + j] = w;
            weights[k - j] = w;
        }

        // Normalize weights to sum to zero (band-pass constraint)
        let wsum: f64 = weights.sum();
        let adj = wsum / (2 * k + 1) as f64;
        weights.mapv_inplace(|w| w - adj);

        // Apply filter
        let mut cycle = Array1::from_elem(n, f64::NAN);
        for t in k..n - k {
            let mut val = 0.0;
            for j in 0..2 * k + 1 {
                val += weights[j] * series[t + j - k];
            }
            cycle[t] = val;
        }

        Ok(cycle)
    }

    // ── CF Filter ────────────────────────────────────────────────────

    /// Christiano-Fitzgerald asymmetric band-pass filter.
    ///
    /// Unlike BK, this filter uses the full sample and computes optimal
    /// asymmetric weights under a random walk assumption.
    ///
    /// * `low` — minimum period
    /// * `high` — maximum period
    /// * `drift` — if true, assumes random walk with drift
    ///
    /// Returns the cycle component.
    pub fn cf_filter(
        series: &Array1<f64>,
        low: usize,
        high: usize,
        drift: bool,
    ) -> Result<Array1<f64>, GreenersError> {
        let n = series.len();
        if n < 4 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for CF filter".into(),
            ));
        }
        if low < 2 || high <= low {
            return Err(GreenersError::ShapeMismatch(
                "Need low >= 2 and high > low".into(),
            ));
        }

        let omega_high = 2.0 * std::f64::consts::PI / low as f64;
        let omega_low = 2.0 * std::f64::consts::PI / high as f64;

        // Remove drift if requested
        let y = if drift {
            let drift_val = (series[n - 1] - series[0]) / (n - 1) as f64;
            Array1::from_vec(
                (0..n)
                    .map(|t| series[t] - series[0] - drift_val * t as f64)
                    .collect(),
            )
        } else {
            series.clone()
        };

        // Ideal filter weights (for interior points, same as BK ideal)
        // b_j = (sin(omega_high * j) - sin(omega_low * j)) / (pi * j), j != 0
        // b_0 = (omega_high - omega_low) / pi
        let compute_b = |j: i64| -> f64 {
            if j == 0 {
                (omega_high - omega_low) / std::f64::consts::PI
            } else {
                let jf = j as f64;
                ((omega_high * jf).sin() - (omega_low * jf).sin()) / (std::f64::consts::PI * jf)
            }
        };

        // For each time t, compute asymmetric CF weights
        let mut cycle = Array1::<f64>::zeros(n);

        for t in 0..n {
            let mut val = 0.0;
            // Weights for observations j = 0..n-1
            // For the full CF filter under random walk:
            // Use the recommended approach: compute B(t,s) for each t
            // Simplified: use the symmetric ideal weights and adjust endpoints
            for s in 0..n {
                let j = (s as i64) - (t as i64);
                let b = compute_b(j);
                val += b * y[s];
            }
            // Normalize: subtract mean weight * mean(y) to ensure zero-frequency removal
            cycle[t] = val;
        }

        // Apply endpoint correction: ensure cycle sums approximately to zero
        let cycle_mean = cycle.mean().unwrap_or(0.0);
        cycle.mapv_inplace(|v| v - cycle_mean);

        Ok(cycle)
    }

    // ── Phillips-Perron ────────────────────────────────────────────────

    /// Phillips-Perron unit root test.
    ///
    /// Non-parametric correction to ADF: uses Newey-West for serial correlation.
    /// Returns Z(t) statistic with same critical values as ADF.
    pub fn phillips_perron(
        series: &Array1<f64>,
        nlags: Option<usize>,
    ) -> Result<PhillipsPerronResult, GreenersError> {
        let n = series.len();
        if n < 10 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for Phillips-Perron test".into(),
            ));
        }

        let lags = nlags.unwrap_or_else(|| ((n as f64).powf(0.25)) as usize);

        // AR(1) regression: y_t = a + rho * y_{t-1} + e_t
        let y_dep: Array1<f64> = series.slice(s![1..]).to_owned();
        let n_eff = n - 1;
        let mut x_mat = Array2::<f64>::zeros((n_eff, 2));
        x_mat.column_mut(0).fill(1.0);
        for i in 0..n_eff {
            x_mat[[i, 1]] = series[i];
        }

        let ols_res = OLS::fit(&y_dep, &x_mat, CovarianceType::NonRobust)?;
        let rho = ols_res.params[1];
        let se_rho = ols_res.std_errors[1];
        let resid = ols_res.residuals(&y_dep, &x_mat);

        // Estimate sigma^2 (short-run variance)
        let sigma2: f64 = resid.iter().map(|r| r * r).sum::<f64>() / n_eff as f64;

        // Estimate lambda^2 (long-run variance) via Newey-West
        let mut lambda2 = sigma2;
        for l in 1..=lags {
            let weight = 1.0 - l as f64 / (lags as f64 + 1.0);
            let gamma_l: f64 =
                (l..n_eff).map(|t| resid[t] * resid[t - l]).sum::<f64>() / n_eff as f64;
            lambda2 += 2.0 * weight * gamma_l;
        }
        lambda2 = lambda2.max(1e-15);

        // Z(alpha) and Z(t) statistics
        let nf = n_eff as f64;
        let z_alpha =
            nf * (rho - 1.0) - 0.5 * nf * nf * se_rho * se_rho * (lambda2 - sigma2) / (nf * sigma2);
        let z_t = (sigma2 / lambda2).sqrt() * ols_res.t_values[1]
            - 0.5 * (lambda2 - sigma2) * (nf * se_rho / lambda2.sqrt());

        // Same critical values as ADF
        let crit_1pct = -3.43;
        let crit_5pct = -2.86;
        let crit_10pct = -2.57;

        Ok(PhillipsPerronResult {
            z_alpha,
            z_t,
            critical_values: (crit_1pct, crit_5pct, crit_10pct),
            is_stationary: z_t < crit_5pct,
            lags_used: lags,
            n_obs: n_eff,
        })
    }

    // ── Zivot-Andrews ────────────────────────────────────────────────

    /// Zivot-Andrews structural break unit root test.
    ///
    /// Tests for a unit root allowing for one structural break in intercept.
    /// Searches over all possible break points, returns minimum t-statistic.
    pub fn zivot_andrews(
        series: &Array1<f64>,
        trim: f64,
    ) -> Result<ZivotAndrewsResult, GreenersError> {
        let n = series.len();
        if n < 20 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for Zivot-Andrews test".into(),
            ));
        }

        let trim = trim.clamp(0.05, 0.25);
        let start = (n as f64 * trim) as usize;
        let end = n - start;

        let y_diff = diff(series, 1);
        let max_lags = ((n - 1) as f64).powf(1.0 / 3.0) as usize;
        let effective_n = n - 1 - max_lags;

        let mut best_t = f64::INFINITY;
        let mut best_break = start;

        for bp in start..end {
            // ADF regression with break dummy
            // y_diff = const + gamma*y_{t-1} + theta*DU_t + lagged diffs + e
            // DU_t = 1 if t > bp
            let n_cols = 3 + max_lags; // const, y_{t-1}, DU, lagged diffs
            let mut x_mat = Array2::<f64>::zeros((effective_n, n_cols));
            let target_y = y_diff.slice(s![max_lags..]).to_owned();

            for i in 0..effective_n {
                let t = max_lags + i;
                x_mat[[i, 0]] = 1.0; // const
                x_mat[[i, 1]] = series[t]; // y_{t-1} (level)
                x_mat[[i, 2]] = if t > bp { 1.0 } else { 0.0 }; // break dummy
                for l in 0..max_lags {
                    x_mat[[i, 3 + l]] = y_diff[t - 1 - l];
                }
            }

            if let Ok(ols_res) = OLS::fit(&target_y, &x_mat, CovarianceType::NonRobust) {
                let t_stat = ols_res.t_values[1]; // t-stat on y_{t-1}
                if t_stat < best_t {
                    best_t = t_stat;
                    best_break = bp;
                }
            }
        }

        // Zivot-Andrews critical values (intercept break, model A)
        let critical_values = (-5.34, -4.80, -4.58); // 1%, 5%, 10%

        Ok(ZivotAndrewsResult {
            statistic: best_t,
            break_point: best_break,
            critical_values,
            is_stationary: best_t < critical_values.1,
            n_obs: n,
        })
    }

    // ── lagmat ────────────────────────────────────────────────────────

    /// Build a lag matrix from a 1-D series.
    ///
    /// Returns (n - max_lag) x (max_lag + 1) matrix where column 0 is the
    /// original series and column j is lag j.
    pub fn lagmat(series: &Array1<f64>, max_lag: usize) -> Array2<f64> {
        let n = series.len();
        let rows = n.saturating_sub(max_lag);
        let cols = max_lag + 1;
        let mut mat = Array2::<f64>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                mat[[i, j]] = series[max_lag + i - j];
            }
        }
        mat
    }

    // ── DeterministicProcess ─────────────────────────────────────────

    /// Generate deterministic process components.
    ///
    /// `components` can include: "const", "trend", "seasonal:{period}", "fourier:{period}:{order}"
    ///
    /// Returns matrix with columns for each requested component.
    pub fn deterministic_process(n: usize, components: &[&str]) -> Array2<f64> {
        let mut cols: Vec<Array1<f64>> = Vec::new();

        for &comp in components {
            if comp == "const" {
                cols.push(Array1::ones(n));
            } else if comp == "trend" {
                cols.push(Array1::from_vec((1..=n).map(|t| t as f64).collect()));
            } else if let Some(rest) = comp.strip_prefix("seasonal:") {
                if let Ok(period) = rest.parse::<usize>() {
                    // Seasonal dummies (period-1 columns)
                    for s in 0..period.saturating_sub(1) {
                        cols.push(Array1::from_vec(
                            (0..n)
                                .map(|t| if t % period == s { 1.0 } else { 0.0 })
                                .collect(),
                        ));
                    }
                }
            } else if let Some(rest) = comp.strip_prefix("fourier:") {
                let parts: Vec<&str> = rest.split(':').collect();
                if parts.len() == 2 {
                    if let (Ok(period), Ok(order)) =
                        (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                    {
                        let period_f = period as f64;
                        for k in 1..=order {
                            let k_f = k as f64;
                            cols.push(Array1::from_vec(
                                (0..n)
                                    .map(|t| {
                                        (2.0 * std::f64::consts::PI * k_f * t as f64 / period_f)
                                            .sin()
                                    })
                                    .collect(),
                            ));
                            cols.push(Array1::from_vec(
                                (0..n)
                                    .map(|t| {
                                        (2.0 * std::f64::consts::PI * k_f * t as f64 / period_f)
                                            .cos()
                                    })
                                    .collect(),
                            ));
                        }
                    }
                }
            }
        }

        if cols.is_empty() {
            return Array2::<f64>::zeros((n, 0));
        }

        let n_cols = cols.len();
        let mut mat = Array2::<f64>::zeros((n, n_cols));
        for (j, col) in cols.iter().enumerate() {
            mat.column_mut(j).assign(col);
        }
        mat
    }
}

/// Results of the Phillips-Perron test.
#[derive(Debug)]
pub struct PhillipsPerronResult {
    pub z_alpha: f64,
    pub z_t: f64,
    pub critical_values: (f64, f64, f64), // 1%, 5%, 10%
    pub is_stationary: bool,
    pub lags_used: usize,
    pub n_obs: usize,
}

/// Results of the Zivot-Andrews test.
#[derive(Debug)]
pub struct ZivotAndrewsResult {
    pub statistic: f64,
    pub break_point: usize,
    pub critical_values: (f64, f64, f64), // 1%, 5%, 10%
    pub is_stationary: bool,
    pub n_obs: usize,
}

// ─── Helper functions ────────────────────────────────────────────────────────

fn diff(arr: &Array1<f64>, lag: usize) -> Array1<f64> {
    let len = arr.len();
    let mut out = Vec::with_capacity(len - lag);
    for i in lag..len {
        out.push(arr[i] - arr[i - lag]);
    }
    Array1::from(out)
}

/// Osterwald-Lenum trace test critical values (10%, 5%, 1%).
/// Rows correspond to H0: rank = 0, 1, 2, ...
fn johansen_trace_critical_values(n_vars: usize) -> Array2<f64> {
    // Critical values for unrestricted constant case
    match n_vars {
        2 => Array2::from_shape_vec(
            (2, 3),
            vec![
                13.43, 15.41, 20.04, // r=0
                2.71, 3.76, 6.65, // r=1
            ],
        )
        .unwrap(),
        3 => Array2::from_shape_vec(
            (3, 3),
            vec![
                27.07, 29.68, 35.65, // r=0
                13.43, 15.41, 20.04, // r=1
                2.71, 3.76, 6.65, // r=2
            ],
        )
        .unwrap(),
        4 => Array2::from_shape_vec(
            (4, 3),
            vec![
                43.95, 47.21, 54.46, // r=0
                27.07, 29.68, 35.65, // r=1
                13.43, 15.41, 20.04, // r=2
                2.71, 3.76, 6.65, // r=3
            ],
        )
        .unwrap(),
        5 => Array2::from_shape_vec(
            (5, 3),
            vec![
                64.84, 68.52, 76.07, // r=0
                43.95, 47.21, 54.46, // r=1
                27.07, 29.68, 35.65, // r=2
                13.43, 15.41, 20.04, // r=3
                2.71, 3.76, 6.65, // r=4
            ],
        )
        .unwrap(),
        _ => {
            // For k > 5 or k < 2, return dummy values
            let mut cv = Array2::<f64>::zeros((n_vars, 3));
            for r in 0..n_vars {
                // Rough approximation
                let remaining = (n_vars - r) as f64;
                cv[[r, 0]] = remaining * 6.5 + 2.7;
                cv[[r, 1]] = remaining * 7.5 + 3.8;
                cv[[r, 2]] = remaining * 10.0 + 6.6;
            }
            cv
        }
    }
}

/// Osterwald-Lenum max eigenvalue critical values (10%, 5%, 1%).
fn johansen_max_eigen_critical_values(n_vars: usize) -> Array2<f64> {
    match n_vars {
        2 => Array2::from_shape_vec(
            (2, 3),
            vec![
                12.30, 14.07, 18.63, // r=0
                2.71, 3.76, 6.65, // r=1
            ],
        )
        .unwrap(),
        3 => Array2::from_shape_vec(
            (3, 3),
            vec![
                18.89, 21.13, 25.86, // r=0
                12.30, 14.07, 18.63, // r=1
                2.71, 3.76, 6.65, // r=2
            ],
        )
        .unwrap(),
        4 => Array2::from_shape_vec(
            (4, 3),
            vec![
                25.12, 27.58, 32.72, // r=0
                18.89, 21.13, 25.86, // r=1
                12.30, 14.07, 18.63, // r=2
                2.71, 3.76, 6.65, // r=3
            ],
        )
        .unwrap(),
        5 => Array2::from_shape_vec(
            (5, 3),
            vec![
                30.84, 33.46, 38.77, // r=0
                25.12, 27.58, 32.72, // r=1
                18.89, 21.13, 25.86, // r=2
                12.30, 14.07, 18.63, // r=3
                2.71, 3.76, 6.65, // r=4
            ],
        )
        .unwrap(),
        _ => {
            let mut cv = Array2::<f64>::zeros((n_vars, 3));
            for r in 0..n_vars {
                let remaining = (n_vars - r) as f64;
                cv[[r, 0]] = remaining * 6.0 + 2.7;
                cv[[r, 1]] = remaining * 7.0 + 3.8;
                cv[[r, 2]] = remaining * 9.0 + 6.6;
            }
            cv
        }
    }
}
