//! Mixed-Frequency VAR (MF-VAR) with MIDAS-style aggregation.
//!
//! Foroni, Ghysels & Marcellino (2013). Allows combining variables
//! observed at different frequencies (e.g., monthly GDP + daily
//! interest rates) in a single VAR framework.
//!
//! Approach: aggregate high-frequency variables to low frequency
//! via MIDAS (MiXed DAta Sampling) polynomial weights, then
//! estimate a standard VAR on the aggregated data.
//!
//! MIDAS aggregation: x_L,t = sum_{j=0}^{J-1} w(j; theta) * x_H,t*J-j
//! where w(j; theta) are exponential Almon weights:
//!   w(j; theta) = exp(theta_1 * j + theta_2 * j^2) / sum(exp(...))

use crate::linalg::{LinalgDeterminant as _, LinalgInverse as _};
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of MF-VAR estimation.
#[derive(Debug)]
pub struct MfVarResult {
    /// VAR coefficients (k x (k*p)), each row = equation
    pub coeffs: Array2<f64>,
    /// Standard errors
    pub std_errors: Array2<f64>,
    /// t-values
    pub t_values: Array2<f64>,
    /// p-values
    pub p_values: Array2<f64>,
    /// MIDAS aggregation weights (for high-freq variables)
    pub midas_weights: Array1<f64>,
    /// MIDAS parameters theta
    pub midas_theta: Array1<f64>,
    /// Aggregated high-frequency series (n_low x n_hf_vars)
    pub aggregated: Array2<f64>,
    /// Residual covariance
    pub resid_cov: Array2<f64>,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of low-frequency observations
    pub n_obs: usize,
    /// Number of variables (low + high freq aggregated)
    pub n_vars: usize,
    /// VAR lag order
    pub lags: usize,
    /// Aggregation ratio (high freq periods per low freq period)
    pub agg_ratio: usize,
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for MfVarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Mixed-Frequency VAR (MF-VAR) ")?;
        writeln!(f, "Foroni, Ghysels & Marcellino (2013)")?;
        writeln!(f, "{:<20} {:>12}", "Low-freq obs:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;
        writeln!(f, "{:<20} {:>12}", "Agg. ratio:", self.agg_ratio)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        // MIDAS weights
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  MIDAS aggregation weights (exponential Almon):")?;
        writeln!(
            f,
            "  theta = [{:.4}, {:.4}]",
            self.midas_theta[0], self.midas_theta[1]
        )?;
        for (j, &w) in self.midas_weights.iter().enumerate() {
            writeln!(f, "  w({}) = {:.6}", j, w)?;
        }

        // VAR coefficients
        let k = self.n_vars;
        let p = self.lags;
        for eq in 0..k {
            let eq_name = self
                .var_names
                .get(eq)
                .cloned()
                .unwrap_or_else(|| format!("y{}", eq));
            writeln!(f, "\n{:-^78}", format!(" Equation: {} ", eq_name))?;
            writeln!(
                f,
                "{:<14} {:>12} {:>12} {:>10} {:>10}",
                "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
            )?;
            for lag in 0..p {
                for j in 0..k {
                    let var_name = self
                        .var_names
                        .get(j)
                        .cloned()
                        .unwrap_or_else(|| format!("y{}", j));
                    let col = lag * k + j;
                    let label = format!("L{}.{}", lag + 1, var_name);
                    writeln!(
                        f,
                        "{:<14} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                        label,
                        self.coeffs[(eq, col)],
                        self.std_errors[(eq, col)],
                        self.t_values[(eq, col)],
                        self.p_values[(eq, col)]
                    )?;
                }
            }
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct MFVAR;

impl MFVAR {
    /// Estimate MF-VAR with MIDAS aggregation.
    ///
    /// # Arguments
    /// * `y_low` - Low-frequency variables (T_low x k_low)
    /// * `y_high` - High-frequency variables (T_high x k_high)
    ///   where T_high = T_low * agg_ratio
    /// * `agg_ratio` - High-freq periods per low-freq period (e.g., 3 for monthly→quarterly)
    /// * `lags` - VAR lag order
    /// * `var_names_low` - Names for low-freq variables
    /// * `var_names_high` - Names for high-freq variables
    pub fn fit(
        y_low: &Array2<f64>,
        y_high: &Array2<f64>,
        agg_ratio: usize,
        lags: usize,
        var_names_low: Option<Vec<String>>,
        var_names_high: Option<Vec<String>>,
    ) -> Result<MfVarResult, GreenersError> {
        let t_low = y_low.nrows();
        let k_low = y_low.ncols();
        let t_high = y_high.nrows();
        let k_high = y_high.ncols();

        if agg_ratio == 0 {
            return Err(GreenersError::InvalidOperation(
                "MFVAR: agg_ratio must be >= 1".into(),
            ));
        }
        if t_high < t_low * agg_ratio {
            return Err(GreenersError::ShapeMismatch(
                "MFVAR: y_high too short for given agg_ratio".into(),
            ));
        }
        if lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "MFVAR: lags must be >= 1".into(),
            ));
        }

        let names_low =
            var_names_low.unwrap_or_else(|| (0..k_low).map(|i| format!("y_low{}", i)).collect());
        let names_high =
            var_names_high.unwrap_or_else(|| (0..k_high).map(|i| format!("y_high{}", i)).collect());

        // Step 1: Estimate MIDAS weights via grid search on theta
        // For simplicity, use uniform weights as starting point and optimize
        let (midas_weights, midas_theta, aggregated) =
            Self::midas_aggregate(y_high, t_low, k_high, agg_ratio)?;

        // Step 2: Combine low-freq + aggregated high-freq into combined matrix
        let k_total = k_low + k_high;
        let mut y_combined = Array2::zeros((t_low, k_total));
        for i in 0..t_low {
            for j in 0..k_low {
                y_combined[(i, j)] = y_low[(i, j)];
            }
            for j in 0..k_high {
                y_combined[(i, k_low + j)] = aggregated[(i, j)];
            }
        }

        let mut all_names = names_low.clone();
        all_names.extend(names_high);

        // Step 3: Estimate VAR(p) on combined data
        let n_eff = t_low - lags;
        let n_reg = k_total * lags;
        let mut x = Array2::zeros((n_eff, n_reg));
        let mut y_dep = Array2::zeros((n_eff, k_total));

        for i in 0..n_eff {
            let t_i = lags + i;
            for j in 0..k_total {
                y_dep[(i, j)] = y_combined[(t_i, j)];
            }
            for lag in 0..lags {
                for j in 0..k_total {
                    x[(i, lag * k_total + j)] = y_combined[(t_i - 1 - lag, j)];
                }
            }
        }

        // OLS
        let xt = x.t();
        let xtx = xt.dot(&x);
        let xtx_inv = (&xtx + Array2::<f64>::eye(n_reg) * 1e-8).inv()?;
        let xty = xt.dot(&y_dep);
        let coeffs_mat = xtx_inv.dot(&xty); // (n_reg x k_total)

        let residuals = &y_dep - x.dot(&coeffs_mat);
        let resid_cov = residuals.t().dot(&residuals) / n_eff as f64;

        // SE, t, p
        let mut se = Array2::zeros((k_total, n_reg));
        let mut tv = Array2::zeros((k_total, n_reg));
        let mut pv = Array2::zeros((k_total, n_reg));
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        for eq in 0..k_total {
            let sigma2 = resid_cov[(eq, eq)].max(1e-10);
            for col in 0..n_reg {
                let se_val = (sigma2 * xtx_inv[(col, col)]).sqrt();
                let coef_val = coeffs_mat[(col, eq)];
                se[(eq, col)] = se_val;
                tv[(eq, col)] = if se_val > 1e-10 {
                    coef_val / se_val
                } else {
                    0.0
                };
                pv[(eq, col)] = 2.0 * (1.0 - normal.cdf(tv[(eq, col)].abs()));
            }
        }

        let coeffs = coeffs_mat.t().to_owned();
        let n_params = k_total * k_total * lags;
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let log_lik = -0.5 * n_eff as f64 * k_total as f64 * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * n_eff as f64 * resid_cov.det().unwrap_or(1e-300).ln().max(-300.0)
            - 0.5 * rss / resid_cov.det().unwrap_or(1e-10).max(1e-10);

        let aic = -2.0 * log_lik + 2.0 * n_params as f64;
        let bic = -2.0 * log_lik + (n_eff as f64) * n_params as f64;

        Ok(MfVarResult {
            coeffs,
            std_errors: se,
            t_values: tv,
            p_values: pv,
            midas_weights,
            midas_theta,
            aggregated,
            resid_cov,
            aic,
            bic,
            n_obs: n_eff,
            n_vars: k_total,
            lags,
            agg_ratio,
            var_names: all_names,
        })
    }

    /// Aggregate high-frequency data to low frequency using MIDAS exponential Almon weights.
    #[allow(clippy::type_complexity)]
    fn midas_aggregate(
        y_high: &Array2<f64>,
        t_low: usize,
        k_high: usize,
        agg_ratio: usize,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>), GreenersError> {
        // Grid search over theta = (theta1, theta2)
        // Start with uniform weights, then optimize
        let mut best_theta = Array1::zeros(2);
        let mut best_weights = Array1::ones(agg_ratio) / agg_ratio as f64;
        let mut best_agg = Array2::zeros((t_low, k_high));

        // Initialize with simple average
        for i in 0..t_low {
            for j in 0..k_high {
                let mut sum = 0.0;
                for h in 0..agg_ratio {
                    let idx = i * agg_ratio + h;
                    if idx < y_high.nrows() {
                        sum += y_high[(idx, j)];
                    }
                }
                best_agg[(i, j)] = sum / agg_ratio as f64;
            }
        }

        // Grid search over theta to minimize variance of aggregated series
        let mut best_var = f64::INFINITY;
        let n_grid = 11;
        for i in 0..n_grid {
            for j in 0..n_grid {
                let t1 = -2.0 + 4.0 * i as f64 / (n_grid - 1) as f64;
                let t2 = -1.0 + 2.0 * j as f64 / (n_grid - 1) as f64;

                // Compute weights
                let mut raw_w = Array1::zeros(agg_ratio);
                let mut sum_w = 0.0;
                for h in 0..agg_ratio {
                    let j_f = h as f64;
                    raw_w[h] = (t1 * j_f + t2 * j_f * j_f).exp();
                    sum_w += raw_w[h];
                }
                if sum_w < 1e-10 {
                    continue;
                }
                let weights = raw_w / sum_w;

                // Aggregate
                let mut agg = Array2::zeros((t_low, k_high));
                for ti in 0..t_low {
                    for jj in 0..k_high {
                        let mut s = 0.0;
                        for h in 0..agg_ratio {
                            let idx = ti * agg_ratio + h;
                            if idx < y_high.nrows() {
                                s += weights[h] * y_high[(idx, jj)];
                            }
                        }
                        agg[(ti, jj)] = s;
                    }
                }

                // Objective: minimize total variance (prefer smooth aggregation)
                let var: f64 = (0..k_high)
                    .map(|jj| {
                        let col = agg.column(jj);
                        let mean = col.mean().unwrap_or(0.0);
                        col.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    })
                    .sum();

                if var < best_var {
                    best_var = var;
                    best_theta = Array1::from_vec(vec![t1, t2]);
                    best_weights = weights;
                    best_agg = agg;
                }
            }
        }

        Ok((best_weights, best_theta, best_agg))
    }
}
