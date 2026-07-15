//! Threshold VAR (TVAR) — regime switching via threshold variable.
//!
//! Tong (1978), Tsay (1998). Extends VAR to allow different
//! coefficient regimes depending on a threshold variable q_t:
//!
//! Regime 1 (q_t <= c):  y_t = A_1 * y_{t-1} + ... + A_p * y_{t-p} + eps_1
//! Regime 2 (q_t >  c):  y_t = A_2 * y_{t-1} + ... + A_p * y_{t-p} + eps_2
//!
//! The threshold c and lag d are estimated via grid search over
//! possible threshold values (sorted q_t percentiles) and delay
//! parameters. For each (c, d), estimate two VAR(p) models and
//! compute the combined RSS. Select (c, d) minimizing RSS.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of TVAR estimation.
#[derive(Debug)]
pub struct TvarResult {
    /// Threshold value c
    pub threshold: f64,
    /// Delay parameter d (lag of threshold variable)
    pub delay: usize,
    /// VAR coefficients in regime 1 (k x (k*p)) — low regime
    pub coeffs_low: Array2<f64>,
    /// VAR coefficients in regime 2 (k x (k*p)) — high regime
    pub coeffs_high: Array2<f64>,
    /// Standard errors regime 1
    pub se_low: Array2<f64>,
    /// Standard errors regime 2
    pub se_high: Array2<f64>,
    /// t-values regime 1
    pub t_low: Array2<f64>,
    /// t-values regime 2
    pub t_high: Array2<f64>,
    /// p-values regime 1
    pub p_low: Array2<f64>,
    /// p-values regime 2
    pub p_high: Array2<f64>,
    /// Residual covariance regime 1 (k x k)
    pub cov_low: Array2<f64>,
    /// Residual covariance regime 2 (k x k)
    pub cov_high: Array2<f64>,
    /// Combined RSS
    pub rss: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations in regime 1
    pub n_low: usize,
    /// Number of observations in regime 2
    pub n_high: usize,
    /// Number of variables
    pub n_vars: usize,
    /// VAR lag order
    pub lags: usize,
    /// Threshold variable name
    pub threshold_var: String,
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for TvarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Threshold VAR (TVAR) ")?;
        writeln!(f, "Tong (1978), Tsay (1998)")?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;
        writeln!(f, "{:<20} {:>12.6}", "Threshold (c):", self.threshold)?;
        writeln!(f, "{:<20} {:>12}", "Delay (d):", self.delay)?;
        writeln!(f, "{:<20} {:>12}", "Obs in low regime:", self.n_low)?;
        writeln!(f, "{:<20} {:>12}", "Obs in high regime:", self.n_high)?;
        writeln!(f, "{:<20} {:>12.6}", "RSS:", self.rss)?;
        writeln!(f, "{:<20} {:>12.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        let k = self.n_vars;
        let p = self.lags;
        for (regime_name, coeffs, se, tv, pv, n_reg) in [
            (
                "Low (q <= c)",
                &self.coeffs_low,
                &self.se_low,
                &self.t_low,
                &self.p_low,
                self.n_low,
            ),
            (
                "High (q > c)",
                &self.coeffs_high,
                &self.se_high,
                &self.t_high,
                &self.p_high,
                self.n_high,
            ),
        ] {
            writeln!(
                f,
                "\n{:-^78}",
                format!(" Regime: {} (n={}) ", regime_name, n_reg)
            )?;
            for eq in 0..k {
                let eq_name = self
                    .var_names
                    .get(eq)
                    .cloned()
                    .unwrap_or_else(|| format!("y{}", eq));
                writeln!(f, "  Equation: {}", eq_name)?;
                writeln!(
                    f,
                    "  {:<14} {:>12} {:>12} {:>10} {:>10}",
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
                            "  {:<14} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                            label,
                            coeffs[(eq, col)],
                            se[(eq, col)],
                            tv[(eq, col)],
                            pv[(eq, col)]
                        )?;
                    }
                }
                writeln!(f)?;
            }
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct TVAR;

impl TVAR {
    /// Estimate TVAR via grid search over threshold and delay.
    ///
    /// # Arguments
    /// * `y` - Data matrix (T x k)
    /// * `q` - Threshold variable (T)
    /// * `lags` - VAR lag order p
    /// * `max_delay` - Maximum delay parameter to search (1..=max_delay)
    /// * `var_names` - Optional variable names
    pub fn fit(
        y: &Array2<f64>,
        q: &Array1<f64>,
        lags: usize,
        max_delay: usize,
        var_names: Option<Vec<String>>,
    ) -> Result<TvarResult, GreenersError> {
        let t = y.nrows();
        let k = y.ncols();
        if q.len() != t {
            return Err(GreenersError::ShapeMismatch(
                "TVAR: y and q must have same length".into(),
            ));
        }
        if lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "TVAR: lags must be >= 1".into(),
            ));
        }
        if t < (lags + max_delay + 1) * 2 {
            return Err(GreenersError::InvalidOperation(
                "TVAR: too few observations".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("y{}", i)).collect());

        // Grid of threshold values: percentiles of q (15th to 85th)
        let mut q_sorted: Vec<f64> = q.iter().copied().collect();
        q_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_thresholds = 15;
        let threshold_grid: Vec<f64> = (0..n_thresholds)
            .map(|i| {
                let frac = 0.15 + 0.70 * i as f64 / (n_thresholds - 1) as f64;
                let idx = (frac * (t - 1) as f64) as usize;
                q_sorted[idx]
            })
            .collect();

        let mut best_rss = f64::INFINITY;
        let mut best_c = 0.0_f64;
        let mut best_d = 1_usize;
        let mut best_low = Array2::<f64>::zeros((k, k * lags));
        let mut best_high = Array2::<f64>::zeros((k, k * lags));
        let mut best_se_low = Array2::<f64>::zeros((k, k * lags));
        let mut best_se_high = Array2::<f64>::zeros((k, k * lags));
        let mut best_t_low = Array2::<f64>::zeros((k, k * lags));
        let mut best_t_high = Array2::<f64>::zeros((k, k * lags));
        let mut best_p_low = Array2::<f64>::zeros((k, k * lags));
        let mut best_p_high = Array2::<f64>::zeros((k, k * lags));
        let mut best_cov_low = Array2::<f64>::eye(k);
        let mut best_cov_high = Array2::<f64>::eye(k);
        let mut best_n_low = 0;
        let mut best_n_high = 0;

        for d in 1..=max_delay.min(lags) {
            for &c in &threshold_grid {
                // Split observations into regimes based on q_{t-d}
                let n_eff = t - lags;
                let mut low_indices: Vec<usize> = Vec::new();
                let mut high_indices: Vec<usize> = Vec::new();

                for i in 0..n_eff {
                    let t_i = lags + i;
                    let q_delayed = q[t_i - d];
                    if q_delayed <= c {
                        low_indices.push(t_i);
                    } else {
                        high_indices.push(t_i);
                    }
                }

                // Need minimum observations per regime
                let min_obs = k * lags + k + 1;
                if low_indices.len() < min_obs || high_indices.len() < min_obs {
                    continue;
                }

                // Estimate VAR for each regime
                let (coeffs_l, se_l, t_l, p_l, cov_l, rss_l) =
                    Self::var_regime(y, &low_indices, lags, k)?;
                let (coeffs_h, se_h, t_h, p_h, cov_h, rss_h) =
                    Self::var_regime(y, &high_indices, lags, k)?;

                let total_rss = rss_l + rss_h;
                if total_rss < best_rss {
                    best_rss = total_rss;
                    best_c = c;
                    best_d = d;
                    best_low = coeffs_l;
                    best_high = coeffs_h;
                    best_se_low = se_l;
                    best_se_high = se_h;
                    best_t_low = t_l;
                    best_t_high = t_h;
                    best_p_low = p_l;
                    best_p_high = p_h;
                    best_cov_low = cov_l;
                    best_cov_high = cov_h;
                    best_n_low = low_indices.len();
                    best_n_high = high_indices.len();
                }
            }
        }

        if best_rss == f64::INFINITY {
            return Err(GreenersError::InvalidOperation(
                "TVAR: no valid threshold found (insufficient obs per regime)".into(),
            ));
        }

        // Log-likelihood (multivariate normal)
        let n_total = best_n_low + best_n_high;
        let det_l = Self::det_2d(&best_cov_low).max(1e-300);
        let det_h = Self::det_2d(&best_cov_high).max(1e-300);
        let ll = -0.5 * n_total as f64 * k as f64 * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * best_n_low as f64 * det_l.ln()
            - 0.5 * best_n_high as f64 * det_h.ln()
            - 0.5 * best_rss; // simplified

        let n_params = 2 * k * k * lags; // two regimes
        let aic = -2.0 * ll + 2.0 * n_params as f64;
        let bic = -2.0 * ll + (n_total as f64) * n_params as f64;

        Ok(TvarResult {
            threshold: best_c,
            delay: best_d,
            coeffs_low: best_low,
            coeffs_high: best_high,
            se_low: best_se_low,
            se_high: best_se_high,
            t_low: best_t_low,
            t_high: best_t_high,
            p_low: best_p_low,
            p_high: best_p_high,
            cov_low: best_cov_low,
            cov_high: best_cov_high,
            rss: best_rss,
            log_likelihood: ll,
            aic,
            bic,
            n_low: best_n_low,
            n_high: best_n_high,
            n_vars: k,
            lags,
            threshold_var: "q".to_string(),
            var_names: names,
        })
    }

    /// Estimate VAR(p) for a single regime given observation indices.
    #[allow(clippy::type_complexity)]
    fn var_regime(
        y: &Array2<f64>,
        indices: &[usize],
        lags: usize,
        k: usize,
    ) -> Result<
        (
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
            f64,
        ),
        GreenersError,
    > {
        let n = indices.len();
        let n_reg = k * lags;
        let mut x = Array2::zeros((n, n_reg));
        let mut y_dep = Array2::zeros((n, k));

        for (i, &t_i) in indices.iter().enumerate() {
            for j in 0..k {
                y_dep[(i, j)] = y[(t_i, j)];
            }
            for lag in 0..lags {
                for j in 0..k {
                    x[(i, lag * k + j)] = y[(t_i - 1 - lag, j)];
                }
            }
        }

        // OLS: B = (X'X)^{-1} X'Y
        let xt = x.t();
        let xtx = xt.dot(&x);
        let xtx_inv = (&xtx + Array2::<f64>::eye(n_reg) * 1e-8).inv()?;
        let xty = xt.dot(&y_dep);
        let coeffs = xtx_inv.dot(&xty); // (n_reg x k)

        let residuals = &y_dep - x.dot(&coeffs);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // Residual covariance
        let cov = residuals.t().dot(&residuals) / n as f64;

        // Standard errors per equation
        let _sigma2 = rss / (n - n_reg) as f64;
        let mut se = Array2::zeros((k, n_reg));
        let mut tv = Array2::zeros((k, n_reg));
        let mut pv = Array2::zeros((k, n_reg));
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        for eq in 0..k {
            let cov_eq = cov[(eq, eq)].max(1e-10);
            for col in 0..n_reg {
                let se_val = (cov_eq * xtx_inv[(col, col)]).sqrt();
                let coef_val = coeffs[(col, eq)];
                se[(eq, col)] = se_val;
                tv[(eq, col)] = if se_val > 1e-10 {
                    coef_val / se_val
                } else {
                    0.0
                };
                pv[(eq, col)] = 2.0 * (1.0 - normal.cdf(tv[(eq, col)].abs()));
            }
        }

        // Transpose coeffs from (n_reg x k) to (k x n_reg)
        let coeffs_t = coeffs.t().to_owned();

        Ok((coeffs_t, se, tv, pv, cov, rss))
    }

    fn det_2d(m: &Array2<f64>) -> f64 {
        let n = m.nrows();
        if n == 1 {
            return m[(0, 0)];
        }
        if n == 2 {
            return m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
        }
        // LU decomposition
        let mut a = m.clone();
        let mut det = 1.0;
        for i in 0..n {
            let mut max_row = i;
            for r in (i + 1)..n {
                if a[(r, i)].abs() > a[(max_row, i)].abs() {
                    max_row = r;
                }
            }
            if max_row != i {
                for j in 0..n {
                    let tmp = a[(i, j)];
                    a[(i, j)] = a[(max_row, j)];
                    a[(max_row, j)] = tmp;
                }
                det = -det;
            }
            if a[(i, i)].abs() < 1e-300 {
                return 0.0;
            }
            det *= a[(i, i)];
            for r in (i + 1)..n {
                let factor = a[(r, i)] / a[(i, i)];
                for j in i..n {
                    a[(r, j)] -= factor * a[(i, j)];
                }
            }
        }
        det
    }
}
