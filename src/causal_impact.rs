//! Causal Impact (Brodersen, Gallus, Henderson & Orban 2015).
//!
//! Bayesian structural time series for causal inference via
//! counterfactual prediction. Originally implemented in Google's
//! CausalImpact R package.
//!
//! Model:
//!   y_t = mu_t + tau_t + beta * x_t + epsilon_t
//!   mu_t = mu_{t-1} + delta_{t-1} + eta_t  (local linear trend)
//!   delta_t = delta_{t-1} + nu_t
//!
//! Where x_t are control series (not affected by intervention).
//! The intervention effect is:
//!   tau_t = y_t - y_t^{counterfactual}
//!
//! This simplified implementation uses:
//!   1. OLS regression of y on controls (pre-treatment period)
//!   2. Kalman-filter-like state space model for local level
//!   3. Bayesian posterior for counterfactual prediction
//!   4. Cumulative effect and posterior probability

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of Causal Impact estimation.
#[derive(Debug)]
pub struct CausalImpactResult {
    /// Observed y (n)
    pub y: Array1<f64>,
    /// Predicted counterfactual (n)
    pub counterfactual: Array1<f64>,
    /// Counterfactual standard deviation (n)
    pub counterfactual_sd: Array1<f64>,
    /// Pointwise effect: y - counterfactual (n)
    pub pointwise_effect: Array1<f64>,
    /// Cumulative effect (n)
    pub cumulative_effect: Array1<f64>,
    /// Posterior mean of average effect (post-treatment)
    pub avg_effect: f64,
    /// SD of average effect
    pub avg_effect_sd: f64,
    /// 95% CI for average effect
    pub avg_effect_ci: [f64; 2],
    /// Posterior probability that effect > 0
    pub p_effect_positive: f64,
    /// Total cumulative effect
    pub total_effect: f64,
    /// SD of total effect
    pub total_effect_sd: f64,
    /// 95% CI for total effect
    pub total_effect_ci: [f64; 2],
    /// Pre-treatment period length
    pub n_pre: usize,
    /// Post-treatment period length
    pub n_post: usize,
    /// Regression coefficients on controls
    pub coefficients: Array1<f64>,
    /// Control variable names
    pub control_names: Vec<String>,
    /// Pre-treatment R-squared
    pub pre_r_squared: f64,
}

impl fmt::Display for CausalImpactResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Causal Impact ")?;
        writeln!(f, "Brodersen, Gallus, Henderson & Orban (2015)")?;
        writeln!(f, "Bayesian structural time series")?;
        writeln!(f, "{:<20} {:>12}", "Pre-treatment:", self.n_pre)?;
        writeln!(f, "{:<20} {:>12}", "Post-treatment:", self.n_post)?;
        writeln!(f, "{:<20} {:>12.6}", "Pre-R²:", self.pre_r_squared)?;

        // Average effect
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Average effect (post-treatment):")?;
        writeln!(f, "  {:<20} {:>12.6}", "Posterior mean:", self.avg_effect)?;
        writeln!(f, "  {:<20} {:>12.6}", "SD:", self.avg_effect_sd)?;
        writeln!(
            f,
            "  {:<20} [{:.4}, {:.4}]",
            "95% CI:", self.avg_effect_ci[0], self.avg_effect_ci[1]
        )?;
        writeln!(
            f,
            "  {:<20} {:>12.4}",
            "P(effect > 0):", self.p_effect_positive
        )?;

        // Total cumulative effect
        writeln!(f, "\n  Cumulative effect:")?;
        writeln!(f, "  {:<20} {:>12.6}", "Total:", self.total_effect)?;
        writeln!(f, "  {:<20} {:>12.6}", "SD:", self.total_effect_sd)?;
        writeln!(
            f,
            "  {:<20} [{:.4}, {:.4}]",
            "95% CI:", self.total_effect_ci[0], self.total_effect_ci[1]
        )?;

        // Coefficients
        writeln!(f, "\n  Control coefficients:")?;
        writeln!(f, "  {:<14} {:>12}", "Variable", "Coef")?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(f, "  {:<14} {:>12.6}", "Intercept", self.coefficients[0])?;
        for (j, name) in self.control_names.iter().enumerate() {
            if j + 1 < self.coefficients.len() {
                writeln!(f, "  {:<14} {:>12.6}", name, self.coefficients[j + 1])?;
            }
        }

        // Post-treatment summary
        writeln!(f, "\n  Post-treatment pointwise effects:")?;
        let n_show = self.n_post.min(5);
        writeln!(
            f,
            "  {:<6} {:>12} {:>12} {:>12}",
            "t", "Observed", "Predicted", "Effect"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..n_show {
            let idx = self.n_pre + i;
            writeln!(
                f,
                "  {:<6} {:>12.4} {:>12.4} {:>12.4}",
                idx + 1,
                self.y[idx],
                self.counterfactual[idx],
                self.pointwise_effect[idx]
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct CausalImpact;

impl CausalImpact {
    /// Estimate causal impact via Bayesian structural time series.
    ///
    /// # Arguments
    /// * `y` - Treated series (n)
    /// * `controls` - Control series (n x k), not affected by intervention
    /// * `treatment_period` - Index where treatment starts (pre-period = 0..treatment_period)
    /// * `control_names` - Optional names for control variables
    pub fn fit(
        y: &Array1<f64>,
        controls: &Array2<f64>,
        treatment_period: usize,
        control_names: Option<Vec<String>>,
    ) -> Result<CausalImpactResult, GreenersError> {
        let n = y.len();
        let k = controls.ncols();
        if controls.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "CausalImpact: y and controls must have same n".into(),
            ));
        }
        if treatment_period >= n - 2 {
            return Err(GreenersError::InvalidOperation(
                "CausalImpact: treatment_period must leave at least 2 post-period obs".into(),
            ));
        }
        if treatment_period < k + 2 {
            return Err(GreenersError::InvalidOperation(
                "CausalImpact: pre-treatment period too short for controls".into(),
            ));
        }

        let names = control_names.unwrap_or_else(|| (0..k).map(|i| format!("c{}", i)).collect());
        let n_pre = treatment_period;
        let n_post = n - treatment_period;

        // Step 1: OLS regression on pre-treatment period
        // y_pre = [1, controls_pre] * beta + epsilon
        let mut x_pre = Array2::zeros((n_pre, k + 1));
        let mut y_pre = Array1::zeros(n_pre);
        for i in 0..n_pre {
            x_pre[(i, 0)] = 1.0;
            for j in 0..k {
                x_pre[(i, j + 1)] = controls[(i, j)];
            }
            y_pre[i] = y[i];
        }

        let xt = x_pre.t();
        let xtx = xt.dot(&x_pre);
        let xtx_inv = (&xtx + Array2::<f64>::eye(k + 1) * 1e-8).inv()?;
        let xty = xt.dot(&y_pre);
        let beta: Array1<f64> = xtx_inv.dot(&xty);

        // Residuals and sigma^2
        let y_pre_hat = x_pre.dot(&beta);
        let residuals_pre = &y_pre - &y_pre_hat;
        let sigma2 = residuals_pre.mapv(|r| r * r).sum() / (n_pre - k - 1) as f64;
        let _sigma = sigma2.sqrt();

        // Pre-treatment R-squared
        let y_pre_mean = y_pre.mean().unwrap_or(0.0);
        let tss = y_pre.mapv(|v| (v - y_pre_mean).powi(2)).sum();
        let sse = residuals_pre.mapv(|r| r * r).sum();
        let pre_r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        // Step 2: Predict counterfactual for all periods
        let mut counterfactual = Array1::zeros(n);
        let mut counterfactual_sd = Array1::zeros(n);
        for i in 0..n {
            let mut x_i = Array1::zeros(k + 1);
            x_i[0] = 1.0;
            for j in 0..k {
                x_i[j + 1] = controls[(i, j)];
            }
            counterfactual[i] = beta.dot(&x_i);

            // Prediction variance: x' * (X'X)^{-1} * x * sigma^2 + sigma^2
            let pred_var = x_i.dot(&xtx_inv).dot(&x_i) * sigma2 + sigma2;
            counterfactual_sd[i] = pred_var.sqrt();
        }

        // Step 3: Add local level component (random walk state)
        // Estimate local level variance from pre-treatment residuals
        let level_var = sigma2 * 0.1; // shrinkage

        // Adjust counterfactual with local level (cumulative residual adjustment)
        let mut level = 0.0;
        for i in 0..n {
            if i < n_pre {
                // Update level from pre-treatment residuals
                let resid = y[i] - counterfactual[i];
                level = level * 0.9 + resid * 0.1;
            }
            // Add level to counterfactual
            counterfactual[i] += level;
            // Add level variance to prediction SD
            counterfactual_sd[i] = (counterfactual_sd[i].powi(2) + level_var).sqrt();
        }

        // Step 4: Compute effects
        let pointwise_effect = y - &counterfactual;
        let mut cumulative_effect = Array1::zeros(n);
        let mut cumsum = 0.0;
        for i in 0..n {
            if i >= n_pre {
                cumsum += pointwise_effect[i];
            }
            cumulative_effect[i] = cumsum;
        }

        // Average effect (post-treatment)
        let post_effects: Vec<f64> = (n_pre..n).map(|i| pointwise_effect[i]).collect();
        let avg_effect: f64 = post_effects.iter().sum::<f64>() / n_post as f64;

        // SD of average effect
        let post_sds: Vec<f64> = (n_pre..n).map(|i| counterfactual_sd[i]).collect();
        let avg_var: f64 = post_sds.iter().map(|s| s * s).sum::<f64>() / (n_post * n_post) as f64;
        let avg_effect_sd = avg_var.sqrt();

        // Total effect
        let total_effect: f64 = post_effects.iter().sum();
        let total_var: f64 = post_sds.iter().map(|s| s * s).sum();
        let total_effect_sd = total_var.sqrt();

        // CIs and p-values
        let z = 1.959964;
        let avg_effect_ci = [
            avg_effect - z * avg_effect_sd,
            avg_effect + z * avg_effect_sd,
        ];
        let total_effect_ci = [
            total_effect - z * total_effect_sd,
            total_effect + z * total_effect_sd,
        ];

        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_effect_positive = 1.0 - normal.cdf(avg_effect / avg_effect_sd.max(1e-10));

        Ok(CausalImpactResult {
            y: y.clone(),
            counterfactual,
            counterfactual_sd,
            pointwise_effect,
            cumulative_effect,
            avg_effect,
            avg_effect_sd,
            avg_effect_ci,
            p_effect_positive,
            total_effect,
            total_effect_sd,
            total_effect_ci,
            n_pre,
            n_post,
            coefficients: beta,
            control_names: names,
            pre_r_squared,
        })
    }
}
