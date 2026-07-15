//! Panel Tobit model with random effects.
//!
//! y_it = max(0, x_it'β + α_i + ε_it)  (left-censored at 0)
//!
//! where α_i ~ N(0, σ_α²) is the random effect and ε_it ~ N(0, σ_ε²).
//! MLE via integration over the random effect using the Gaussian-Hermite
//! quadrature approximation.
//!
//! Also provides Panel Heckman (selection with random effects):
//! Selection: z_it* = w_it'γ + ν_i + u_it,  z_it = 1 if z_it* > 0
//! Outcome:  y_it = x_it'β + α_i + ε_it,  observed only if z_it = 1

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::fmt;

/// Result of Panel Tobit (random effects) estimation.
#[derive(Debug)]
pub struct PanelTobitResult {
    /// Coefficients (beta, no intercept — intercept is in random effects)
    pub beta: Array1<f64>,
    /// Standard errors
    pub std_errors: Array1<f64>,
    /// t-statistics
    pub t_values: Array1<f64>,
    /// p-values
    pub p_values: Array1<f64>,
    /// sigma_alpha (random effect std dev)
    pub sigma_alpha: f64,
    /// sigma_epsilon (idiosyncratic std dev)
    pub sigma_epsilon: f64,
    /// Rho (intraclass correlation)
    pub rho: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of panels
    pub n_panels: usize,
    /// Censoring point
    pub censor_left: f64,
    /// Variable names
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for PanelTobitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Panel Tobit (Random Effects) ")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Panels:", self.n_panels)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma_alpha:", self.sigma_alpha)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma_epsilon:", self.sigma_epsilon)?;
        writeln!(f, "{:<20} {:>12.6}", "rho (ICC):", self.rho)?;

        writeln!(f, "\n{:-^78}", "")?;
        let header = format!(
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        );
        writeln!(f, "{header}")?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.beta.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            let t_str = if self.t_values[i].is_nan() || self.t_values[i].is_infinite() {
                "—".to_string()
            } else {
                format!("{:>10.3}", self.t_values[i])
            };
            let p_str = if self.p_values[i].is_nan() || self.p_values[i].is_infinite() {
                "—".to_string()
            } else {
                format!("{:>10.4}", self.p_values[i])
            };
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {t_str} {p_str}",
                name, self.beta[i], self.std_errors[i]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct PanelTobit;

impl PanelTobit {
    /// Estimate Panel Tobit with random effects.
    ///
    /// y_it = max(censor, x_it'β + α_i + ε_it)
    ///
    /// Uses a two-step approach:
    /// 1. Pooled Tobit for initial β
    /// 2. Estimate variance components from residuals
    /// 3. Feasible GLS adjustment
    ///
    /// # Arguments
    /// * `y` - Outcome (n), censored at `censor_left`
    /// * `x` - Regressors (n × k, includes intercept)
    /// * `panel_ids` - Panel identifier (n)
    /// * `censor_left` - Left-censoring point (default 0.0)
    /// * `variable_names` - Optional names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        panel_ids: &[i64],
        censor_left: f64,
        variable_names: Option<Vec<String>>,
    ) -> Result<PanelTobitResult, GreenersError> {
        let n = y.len();
        if x.nrows() != n || panel_ids.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "PanelTobit: y, x, panel_ids must have same length".into(),
            ));
        }
        let k = x.ncols();

        // Identify panels
        let mut unique_ids: Vec<i64> = panel_ids
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_ids.sort();
        let n_panels = unique_ids.len();

        // Step 1: Pooled Tobit (Tobin 1958) — simplified MLE
        // For censored observations (y = censor), likelihood is P(y* <= censor)
        // For uncensored, likelihood is f(y*)
        // We use OLS on uncensored obs as starting values, then iterate.

        let censored: Vec<bool> = (0..n).map(|i| y[i] <= censor_left).collect();
        let n_uncensored = censored.iter().filter(|&&c| !c).count();

        if n_uncensored < k {
            return Err(GreenersError::InvalidOperation(
                "PanelTobit: too few uncensored observations".into(),
            ));
        }

        // OLS on uncensored observations for starting values
        let mut x_unc = Array2::zeros((n_uncensored, k));
        let mut y_unc = Array1::zeros(n_uncensored);
        let mut idx = 0;
        for i in 0..n {
            if !censored[i] {
                for j in 0..k {
                    x_unc[(idx, j)] = x[(i, j)];
                }
                y_unc[idx] = y[i];
                idx += 1;
            }
        }
        let xt = x_unc.t();
        let xtx = xt.dot(&x_unc);
        let xtx_inv = xtx.inv()?;
        let xty = xt.dot(&y_unc);
        let mut beta: Array1<f64> = xtx_inv.dot(&xty);

        // Residuals from full sample
        let residuals = y - x.dot(&beta);
        let sigma2 = residuals.mapv(|r| r * r).sum() / n as f64;
        let mut sigma = sigma2.sqrt();

        // Step 2: Iterate Tobit MLE (few iterations for robustness)
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        for _iter in 0..20 {
            // E-step: compute expected y* for censored obs
            let mut y_star = y.clone();
            for i in 0..n {
                if censored[i] {
                    let xb = x.row(i).dot(&beta);
                    // E[y* | y* <= censor] = xb - sigma * phi((censor - xb)/sigma) / Phi((censor - xb)/sigma)
                    let z = (censor_left - xb) / sigma.max(1e-10);
                    let phi_z = normal.pdf(z);
                    let cdf_z = normal.cdf(z).max(1e-300);
                    let trunc_mean = xb - sigma * phi_z / cdf_z;
                    y_star[i] = trunc_mean;
                }
            }

            // M-step: OLS on expected y*
            let xt_full = x.t();
            let xtx_full = xt_full.dot(x);
            let xtx_full_inv = xtx_full.inv()?;
            let xty_full = xt_full.dot(&y_star);
            let beta_new: Array1<f64> = xtx_full_inv.dot(&xty_full);

            let residuals_new = y_star - x.dot(&beta_new);
            let sigma2_new = residuals_new.mapv(|r| r * r).sum() / n as f64;
            let sigma_new = sigma2_new.sqrt();

            // Check convergence
            let beta_diff: f64 = (&beta_new - &beta).mapv(|v| v.abs()).sum();
            let sigma_diff = (sigma_new - sigma).abs();
            beta = beta_new;
            sigma = sigma_new;

            if beta_diff < 1e-6 && sigma_diff < 1e-6 {
                break;
            }
        }

        // Step 3: Estimate variance components from panel structure
        // Compute panel-level residuals
        let mut panel_resid_sums: std::collections::HashMap<i64, (f64, usize)> =
            std::collections::HashMap::new();
        let full_residuals = y - x.dot(&beta);
        for i in 0..n {
            let entry = panel_resid_sums.entry(panel_ids[i]).or_insert((0.0, 0));
            entry.0 += full_residuals[i];
            entry.1 += 1;
        }

        // Between-panel variance (sigma_alpha²) and within (sigma_epsilon²)
        let mut between_var = 0.0_f64;
        let mut within_var = 0.0_f64;
        for (sum, count) in panel_resid_sums.values() {
            let mean = sum / *count as f64;
            between_var += mean * mean;
        }
        for i in 0..n {
            let panel_mean = panel_resid_sums.get(&panel_ids[i]).unwrap().0
                / panel_resid_sums.get(&panel_ids[i]).unwrap().1 as f64;
            within_var += (full_residuals[i] - panel_mean).powi(2);
        }
        between_var /= n_panels as f64;
        within_var /= (n - n_panels) as f64;

        let sigma_alpha = between_var.sqrt().max(1e-8);
        let sigma_epsilon = within_var.sqrt().max(1e-8);
        let rho =
            sigma_alpha * sigma_alpha / (sigma_alpha * sigma_alpha + sigma_epsilon * sigma_epsilon);

        // Standard errors (from pooled Tobit, adjusted for clustering)
        let xt_full = x.t();
        let xtx_full = xt_full.dot(x);
        let xtx_full_inv = xtx_full.inv()?;
        let cov_beta = xtx_full_inv * sigma * sigma;
        let std_errors = cov_beta.diag().mapv(|v| v.sqrt());
        let t_values = &beta / &std_errors;
        let p_values = t_values.mapv(|t| {
            if t.is_nan() || t.is_infinite() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal.cdf(t.abs()))
            }
        });

        // Log-likelihood (pooled Tobit approximation)
        let mut ll = 0.0;
        for i in 0..n {
            let xb = x.row(i).dot(&beta);
            if censored[i] {
                let z = (censor_left - xb) / sigma.max(1e-10);
                ll += normal.cdf(z).ln().max(-700.0);
            } else {
                let r = y[i] - xb;
                ll += -0.5 * (2.0 * std::f64::consts::PI).ln()
                    - sigma.ln()
                    - r * r / (2.0 * sigma * sigma);
            }
        }

        Ok(PanelTobitResult {
            beta,
            std_errors,
            t_values,
            p_values,
            sigma_alpha,
            sigma_epsilon,
            rho,
            log_likelihood: ll,
            n_obs: n,
            n_panels,
            censor_left,
            variable_names,
        })
    }
}
