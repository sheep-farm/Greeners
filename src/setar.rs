//! SETAR (Self-Exciting Threshold Autoregressive) model.
//!
//! y_t = {
//!   α₁ + Σ φ₁ⱼ y_{t-j} + ε_t   if y_{t-d} ≤ threshold
//!   α₂ + Σ φ₂ⱼ y_{t-j} + ε_t   if y_{t-d} > threshold
//! }
//!
//! where d is the delay parameter and the threshold is estimated
//! by minimizing the RSS over all possible thresholds (conditional
//! on the delay and AR order).

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of SETAR estimation.
#[derive(Debug)]
pub struct SetarResult {
    /// Regime 1 coefficients (intercept + AR terms)
    pub beta_low: Array1<f64>,
    /// Regime 2 coefficients (intercept + AR terms)
    pub beta_high: Array1<f64>,
    /// SE of regime 1 coefficients
    pub se_low: Array1<f64>,
    /// SE of regime 2 coefficients
    pub se_high: Array1<f64>,
    /// t-values of regime 1
    pub t_low: Array1<f64>,
    /// t-values of regime 2
    pub t_high: Array1<f64>,
    /// p-values of regime 1
    pub p_low: Array1<f64>,
    /// p-values of regime 2
    pub p_high: Array1<f64>,
    /// Estimated threshold
    pub threshold: f64,
    /// Delay parameter
    pub delay: usize,
    /// AR order
    pub ar_order: usize,
    /// Number of obs in regime 1 (low)
    pub n_low: usize,
    /// Number of obs in regime 2 (high)
    pub n_high: usize,
    /// R-squared
    pub r_squared: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Sigma (residual std dev, pooled)
    pub sigma: f64,
    /// Number of observations
    pub n_obs: usize,
}

impl fmt::Display for SetarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " SETAR (Threshold AR) ")?;
        writeln!(f, "Model: y_t = a_i + sum phi_ij y_{{t-j}} + eps_t")?;
        writeln!(
            f,
            "  Regime 1 (low):  y_{{t-{d}}} <= {thr:.6}",
            d = self.delay,
            thr = self.threshold
        )?;
        writeln!(
            f,
            "  Regime 2 (high): y_{{t-{d}}} >  {thr:.6}",
            d = self.delay,
            thr = self.threshold
        )?;
        writeln!(f, "{:<20} {:>12}", "AR order:", self.ar_order)?;
        writeln!(f, "{:<20} {:>12}", "Delay:", self.delay)?;
        writeln!(
            f,
            "{:<20} {:>12}",
            "Obs (low/high):",
            format!("{}/{}", self.n_low, self.n_high)
        )?;
        writeln!(f, "{:<20} {:>12.6}", "Threshold:", self.threshold)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "Sigma:", self.sigma)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;

        // Regime 1
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Regime 1 (low): {n} observations", n = self.n_low)?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.beta_low.len() {
            let name = if i == 0 {
                "const".to_string()
            } else {
                format!("y_{{t-{i}}}")
            };
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.beta_low[i], self.se_low[i], self.t_low[i], self.p_low[i]
            )?;
        }

        // Regime 2
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Regime 2 (high): {n} observations", n = self.n_high)?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.beta_high.len() {
            let name = if i == 0 {
                "const".to_string()
            } else {
                format!("y_{{t-{i}}}")
            };
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.beta_high[i], self.se_high[i], self.t_high[i], self.p_high[i]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct SETAR;

impl SETAR {
    /// Estimate SETAR model by grid search over threshold.
    ///
    /// # Arguments
    /// * `y` - Time series (T)
    /// * `ar_order` - AR order p
    /// * `delay` - Delay parameter d (which lag determines regime)
    pub fn fit(
        y: &Array1<f64>,
        ar_order: usize,
        delay: usize,
    ) -> Result<SetarResult, GreenersError> {
        let t = y.len();
        if t < (ar_order + delay) * 3 {
            return Err(GreenersError::InvalidOperation(
                "SETAR: too few observations for given AR order and delay".into(),
            ));
        }

        // Build AR design matrix: y_t = α + Σ φ_j y_{t-j}
        // We need y_{t-d} as the threshold variable.
        let start = ar_order.max(delay);
        let n_eff = t - start;

        let mut x = Array2::zeros((n_eff, ar_order + 1));
        let mut y_dep = Array1::zeros(n_eff);
        let mut thresh_var = Array1::zeros(n_eff);

        for i in 0..n_eff {
            let t_i = start + i;
            y_dep[i] = y[t_i];
            thresh_var[i] = y[t_i - delay];
            x[(i, 0)] = 1.0; // intercept
            for j in 1..=ar_order {
                x[(i, j)] = y[t_i - j];
            }
        }

        // Grid search over threshold (percentiles of thresh_var)
        let mut sorted_thresh: Vec<f64> = thresh_var.iter().copied().collect();
        sorted_thresh.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Trim 15% from each end to ensure enough obs in each regime
        let trim = (n_eff * 15 / 100).max(ar_order + 2);
        let n_candidates = n_eff - 2 * trim;

        let mut best_sse = f64::INFINITY;
        let mut best_threshold = 0.0_f64;
        let mut best_beta_low = Array1::zeros(ar_order + 1);
        let mut best_beta_high = Array1::zeros(ar_order + 1);
        let mut best_n_low = 0;
        let mut best_n_high = 0;
        let mut best_se_low = Array1::zeros(ar_order + 1);
        let mut best_se_high = Array1::zeros(ar_order + 1);

        for i in 0..n_candidates {
            let idx = trim + i;
            let thr = sorted_thresh[idx];

            let low_mask: Vec<bool> = thresh_var.iter().map(|&v| v <= thr).collect();
            let high_mask: Vec<bool> = thresh_var.iter().map(|&v| v > thr).collect();

            let n_low = low_mask.iter().filter(|&&b| b).count();
            let n_high = high_mask.iter().filter(|&&b| b).count();

            if n_low < ar_order + 2 || n_high < ar_order + 2 {
                continue;
            }

            // Regime 1 (low) OLS
            let (beta_low, sse_low, se_low) =
                Self::ols_subset(&y_dep, &x, &low_mask, ar_order + 1)?;
            // Regime 2 (high) OLS
            let (beta_high, sse_high, se_high) =
                Self::ols_subset(&y_dep, &x, &high_mask, ar_order + 1)?;

            let sse = sse_low + sse_high;
            if sse < best_sse {
                best_sse = sse;
                best_threshold = thr;
                best_beta_low = beta_low;
                best_beta_high = beta_high;
                best_n_low = n_low;
                best_n_high = n_high;
                best_se_low = se_low;
                best_se_high = se_high;
            }
        }

        if best_sse == f64::INFINITY {
            return Err(GreenersError::InvalidOperation(
                "SETAR: no valid threshold found (insufficient obs in regimes)".into(),
            ));
        }

        // Stats
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let t_low = &best_beta_low / &best_se_low;
        let t_high = &best_beta_high / &best_se_high;
        let p_low = t_low.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));
        let p_high = t_high.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        let y_mean = y_dep.mean().unwrap_or(0.0);
        let tss = y_dep.mapv(|v| (v - y_mean).powi(2)).sum();
        let r_squared = if tss > 1e-15 {
            1.0 - best_sse / tss
        } else {
            0.0
        };

        let sigma2 = best_sse / (n_eff - 2 * (ar_order + 1)) as f64;
        let sigma = sigma2.sqrt();
        let log_likelihood = -(n_eff as f64) / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln()
            - best_sse / (2.0 * sigma2);

        Ok(SetarResult {
            beta_low: best_beta_low,
            beta_high: best_beta_high,
            se_low: best_se_low,
            se_high: best_se_high,
            t_low,
            t_high,
            p_low,
            p_high,
            threshold: best_threshold,
            delay,
            ar_order,
            n_low: best_n_low,
            n_high: best_n_high,
            r_squared,
            log_likelihood,
            sigma,
            n_obs: n_eff,
        })
    }

    fn ols_subset(
        y: &Array1<f64>,
        x: &Array2<f64>,
        mask: &[bool],
        k: usize,
    ) -> Result<(Array1<f64>, f64, Array1<f64>), GreenersError> {
        let n_sub = mask.iter().filter(|&&b| b).count();
        let mut x_sub = Array2::zeros((n_sub, k));
        let mut y_sub = Array1::zeros(n_sub);
        let mut idx = 0;
        for i in 0..y.len() {
            if mask[i] {
                for j in 0..k {
                    x_sub[(idx, j)] = x[(i, j)];
                }
                y_sub[idx] = y[i];
                idx += 1;
            }
        }

        let xt = x_sub.t();
        let xtx = xt.dot(&x_sub);
        let xtx_reg = &xtx + Array2::eye(k) * 1e-10;
        let xtx_inv = xtx_reg.inv()?;
        let xty = xt.dot(&y_sub);
        let beta: Array1<f64> = xtx_inv.dot(&xty);

        let residuals = &y_sub - x_sub.dot(&beta);
        let sse = residuals.dot(&residuals);
        let sigma2 = sse / (n_sub - k) as f64;
        let se = (xtx_inv * sigma2).diag().mapv(|v| v.sqrt());

        Ok((beta, sse, se))
    }
}
