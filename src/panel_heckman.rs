//! Panel Heckman (selection model with random effects).
//!
//! Selection equation:  z_it* = w_it'γ + ν_i + u_it,   z_it = 1 if z_it* > 0
//! Outcome equation:    y_it = x_it'β + α_i + ε_it,    observed only if z_it = 1
//!
//! where (ν_i, α_i) are panel random effects and (u_it, ε_it) are
//! idiosyncratic errors. The errors are correlated: corr(u, ε) = ρ.
//!
//! Estimation: two-step Heckman with panel-adjusted standard errors.
//! Step 1: Probit on selection equation (pooled, with cluster-robust SE).
//! Step 2: OLS on outcome equation with inverse Mills ratio (IMR) added.
//! Variance components estimated from panel residuals.

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::fmt;

/// Result of Panel Heckman estimation.
#[derive(Debug)]
pub struct PanelHeckmanResult {
    /// Selection equation coefficients (gamma)
    pub gamma: Array1<f64>,
    /// Outcome equation coefficients (beta)
    pub beta: Array1<f64>,
    /// Selection SE
    pub gamma_se: Array1<f64>,
    /// Outcome SE
    pub beta_se: Array1<f64>,
    /// Selection t-values
    pub gamma_t: Array1<f64>,
    /// Outcome t-values
    pub beta_t: Array1<f64>,
    /// Selection p-values
    pub gamma_p: Array1<f64>,
    /// Outcome p-values
    pub beta_p: Array1<f64>,
    /// Rho (correlation between selection and outcome errors)
    pub rho: f64,
    /// Sigma (outcome error std dev)
    pub sigma: f64,
    /// Sigma_alpha (outcome random effect std dev)
    pub sigma_alpha: f64,
    /// Sigma_nu (selection random effect std dev)
    pub sigma_nu: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Mean inverse Mills ratio for selected observations
    pub imr_mean: f64,
    /// IMR coefficient in outcome equation
    pub imr_coef: f64,
    /// Number of observations (total)
    pub n_obs: usize,
    /// Number of selected observations
    pub n_selected: usize,
    /// Number of panels
    pub n_panels: usize,
    /// Variable names (selection)
    pub sel_names: Option<Vec<String>>,
    /// Variable names (outcome)
    pub out_names: Option<Vec<String>>,
}

impl fmt::Display for PanelHeckmanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Panel Heckman (Random Effects) ")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Selected:", self.n_selected)?;
        writeln!(f, "{:<20} {:>12}", "Panels:", self.n_panels)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.6}", "rho (corr):", self.rho)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma:", self.sigma)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma_alpha:", self.sigma_alpha)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma_nu:", self.sigma_nu)?;
        writeln!(f, "{:<20} {:>12.6}", "IMR mean:", self.imr_mean)?;

        // Selection equation
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Selection equation")?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.gamma.len() {
            let name = if i == 0 {
                "_cons".to_string()
            } else {
                self.sel_names
                    .as_ref()
                    .and_then(|n| n.get(i - 1).cloned())
                    .unwrap_or_else(|| format!("w{}", i))
            };
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.gamma[i], self.gamma_se[i], self.gamma_t[i], self.gamma_p[i]
            )?;
        }

        // Outcome equation
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Outcome equation")?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.beta.len() {
            let name = if i == 0 {
                "_cons".to_string()
            } else {
                self.out_names
                    .as_ref()
                    .and_then(|n| n.get(i - 1).cloned())
                    .unwrap_or_else(|| format!("x{}", i))
            };
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.beta[i], self.beta_se[i], self.beta_t[i], self.beta_p[i]
            )?;
        }
        // Print IMR coefficient on its own row (standard error not computed here).
        writeln!(
            f,
            "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            "lambda", self.imr_coef, 0.0, 0.0, 1.0
        )?;
        write!(f, "{:=^78}", "")
    }
}

pub struct PanelHeckman;

impl PanelHeckman {
    /// Estimate Panel Heckman (two-step with random effects).
    ///
    /// # Arguments
    /// * `z` - Selection indicator (1 = observed, 0 = not)
    /// * `y` - Outcome (only meaningful when z=1)
    /// * `w` - Selection regressors (n × k_w, includes intercept)
    /// * `x` - Outcome regressors (n × k_x, includes intercept)
    /// * `panel_ids` - Panel identifier (n)
    /// * `sel_names` - Optional names for selection variables
    /// * `out_names` - Optional names for outcome variables
    pub fn fit(
        z: &[bool],
        y: &Array1<f64>,
        w: &Array2<f64>,
        x: &Array2<f64>,
        panel_ids: &[i64],
        sel_names: Option<Vec<String>>,
        out_names: Option<Vec<String>>,
    ) -> Result<PanelHeckmanResult, GreenersError> {
        let n = z.len();
        if y.len() != n || w.nrows() != n || x.nrows() != n || panel_ids.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "PanelHeckman: all inputs must have same length".into(),
            ));
        }

        let n_selected = z.iter().filter(|&&c| c).count();
        if n_selected < x.ncols() {
            return Err(GreenersError::InvalidOperation(
                "PanelHeckman: too few selected observations".into(),
            ));
        }

        // Count panels
        let mut unique_ids: Vec<i64> = panel_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_ids.sort();
        let n_panels = unique_ids.len();

        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        // ── Step 1: Probit on selection equation ──
        // Pooled probit with Newton-Raphson
        let k_w = w.ncols();
        let mut gamma = Array1::zeros(k_w);

        for _iter in 0..50 {
            let eta = w.dot(&gamma);
            let mut score = Array1::zeros(k_w);
            let mut hessian: Array2<f64> = Array2::zeros((k_w, k_w));

            for i in 0..n {
                let z_i = if z[i] { 1.0 } else { 0.0 };
                let phi = normal.pdf(eta[i]);
                let cdf = normal.cdf(eta[i]).clamp(1e-300, 1.0 - 1e-300);
                let one_minus_cdf = 1.0 - cdf;
                let pdf_over_cdf = (phi / cdf).min(1e6);
                let pdf_over_1mcdf = (phi / one_minus_cdf).min(1e6);
                let d_loglik = z_i * pdf_over_cdf - (1.0 - z_i) * pdf_over_1mcdf;

                for j in 0..k_w {
                    score[j] += d_loglik * w[(i, j)];
                }

                // d²loglik/dη² = -η*d_loglik - z*(phi/cdf)^2 - (1-z)*(phi/(1-cdf))^2
                let d2 = -eta[i] * d_loglik
                    - z_i * pdf_over_cdf * pdf_over_cdf
                    - (1.0 - z_i) * pdf_over_1mcdf * pdf_over_1mcdf;
                for j in 0..k_w {
                    for k in 0..k_w {
                        hessian[(j, k)] += d2 * w[(i, j)] * w[(i, k)];
                    }
                }
            }

            let hessian_reg = &hessian + Array2::eye(k_w) * 1e-4;
            let hessian_inv = hessian_reg.inv()?;
            let mut delta = hessian_inv.dot(&score);
            // Limit the Newton step to avoid overshooting into extreme eta values.
            let max_step = 1.0;
            let delta_norm = delta.mapv(|v| v.abs()).sum();
            if delta_norm > max_step {
                delta = &delta * (max_step / delta_norm);
            }
            gamma = &gamma - &delta;

            if delta.mapv(|v| v.abs()).sum() < 1e-8 {
                break;
            }
        }

        // Selection SE (cluster-robust: sum over panels)
        let eta = w.dot(&gamma);
        let mut bread: Array2<f64> = Array2::zeros((k_w, k_w));
        let mut meat: Array2<f64> = Array2::zeros((k_w, k_w));

        // Group by panel
        let mut panel_scores: std::collections::HashMap<i64, Array1<f64>> =
            std::collections::HashMap::new();

        for i in 0..n {
            let z_i = if z[i] { 1.0 } else { 0.0 };
            let phi = normal.pdf(eta[i]);
            let cdf = normal.cdf(eta[i]).clamp(1e-300, 1.0 - 1e-300);
            let one_minus_cdf = 1.0 - cdf;
            let pdf_over_cdf = (phi / cdf).min(1e6);
            let pdf_over_1mcdf = (phi / one_minus_cdf).min(1e6);
            let d_loglik = z_i * pdf_over_cdf - (1.0 - z_i) * pdf_over_1mcdf;
            let d2 = -eta[i] * d_loglik
                - z_i * pdf_over_cdf * pdf_over_cdf
                - (1.0 - z_i) * pdf_over_1mcdf * pdf_over_1mcdf;

            for j in 0..k_w {
                for k in 0..k_w {
                    bread[(j, k)] += d2 * w[(i, j)] * w[(i, k)];
                }
            }

            let entry = panel_scores
                .entry(panel_ids[i])
                .or_insert_with(|| Array1::zeros(k_w));
            for j in 0..k_w {
                entry[j] += d_loglik * w[(i, j)];
            }
        }

        let bread_inv = (&bread + Array2::eye(k_w) * 1e-6).inv()?;
        for s in panel_scores.values() {
            for j in 0..k_w {
                for k in 0..k_w {
                    meat[(j, k)] += s[j] * s[k];
                }
            }
        }
        let cluster_vcov = &bread_inv * &meat * &bread_inv;
        let gamma_se = cluster_vcov.diag().mapv(|v| v.sqrt().max(0.0));

        let gamma_t = &gamma / &gamma_se;
        let gamma_p = gamma_t.mapv(|t| {
            if t.is_nan() || t.is_infinite() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal.cdf(t.abs()))
            }
        });

        // ── Compute IMR (inverse Mills ratio) for selected obs ──
        let mut imr = Array1::zeros(n);
        let mut ll_sel = 0.0;
        for i in 0..n {
            let phi = normal.pdf(eta[i]);
            let cdf = normal.cdf(eta[i]).max(1e-300);
            if z[i] {
                imr[i] = phi / cdf; // lambda = phi(η) / Φ(η)
                ll_sel += normal.cdf(eta[i]).ln().max(-700.0);
            } else {
                ll_sel += (1.0 - normal.cdf(eta[i])).ln().max(-700.0);
            }
        }

        // ── Step 2: OLS on outcome with IMR (selected obs only) ──
        let k_x = x.ncols();
        let mut x_aug = Array2::zeros((n_selected, k_x + 1));
        let mut y_sel = Array1::zeros(n_selected);
        let mut idx = 0;
        for i in 0..n {
            if z[i] {
                for j in 0..k_x {
                    x_aug[(idx, j)] = x[(i, j)];
                }
                x_aug[(idx, k_x)] = imr[i]; // IMR column
                y_sel[idx] = y[i];
                idx += 1;
            }
        }

        let xt = x_aug.t();
        let xtx = xt.dot(&x_aug);
        let xtx_inv = (&xtx + Array2::eye(k_x + 1) * 1e-8).inv()?;
        let xty = xt.dot(&y_sel);
        let beta_aug: Array1<f64> = xtx_inv.dot(&xty);

        let beta = beta_aug.slice(ndarray::s![0..k_x]).to_owned();
        let imr_coef = beta_aug[k_x];

        // Residuals
        let fitted = x_aug.dot(&beta_aug);
        let residuals = &y_sel - &fitted;
        let sigma2 = residuals.dot(&residuals) / (n_selected - k_x - 1) as f64;
        let sigma = sigma2.sqrt();

        // Outcome SE (cluster-robust)
        let mut beta_se = Array1::zeros(k_x);
        let mut bread_out: Array2<f64> = Array2::zeros((k_x + 1, k_x + 1));
        let mut meat_out: Array2<f64> = Array2::zeros((k_x + 1, k_x + 1));
        let mut panel_scores_out: std::collections::HashMap<i64, Array1<f64>> =
            std::collections::HashMap::new();

        idx = 0;
        for i in 0..n {
            if z[i] {
                let r = residuals[idx];
                for j in 0..(k_x + 1) {
                    for k in 0..(k_x + 1) {
                        bread_out[(j, k)] += x_aug[(idx, j)] * x_aug[(idx, k)];
                    }
                }
                let entry = panel_scores_out
                    .entry(panel_ids[i])
                    .or_insert_with(|| Array1::zeros(k_x + 1));
                for j in 0..(k_x + 1) {
                    entry[j] += r * x_aug[(idx, j)];
                }
                idx += 1;
            }
        }

        let bread_out_inv = (&bread_out + Array2::eye(k_x + 1) * 1e-8).inv()?;
        for s in panel_scores_out.values() {
            for j in 0..(k_x + 1) {
                for k in 0..(k_x + 1) {
                    meat_out[(j, k)] += s[j] * s[k];
                }
            }
        }
        let cluster_vcov_out = &bread_out_inv * &meat_out * &bread_out_inv;
        let full_se = cluster_vcov_out.diag().mapv(|v| v.sqrt().max(0.0));
        for j in 0..k_x {
            beta_se[j] = full_se[j];
        }

        let beta_t = &beta / &beta_se;
        let beta_p = beta_t.mapv(|t| {
            if t.is_nan() || t.is_infinite() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal.cdf(t.abs()))
            }
        });

        // Mean inverse Mills ratio for selected observations
        let imr_sum: f64 = (0..n).filter(|&i| z[i]).map(|i| imr[i]).sum();
        let imr_mean = imr_sum / n_selected as f64;

        // ── Rho: correlation between selection and outcome errors ──
        // rho ≈ imr_coef / sigma (Heckman two-step approximation)
        let rho = (imr_coef / sigma).clamp(-0.99, 0.99);

        // ── Variance components from panel structure ──
        let mut panel_resid_sums: std::collections::HashMap<i64, (f64, usize)> =
            std::collections::HashMap::new();
        idx = 0;
        for i in 0..n {
            if z[i] {
                let entry = panel_resid_sums.entry(panel_ids[i]).or_insert((0.0, 0));
                entry.0 += residuals[idx];
                entry.1 += 1;
                idx += 1;
            }
        }

        let mut between_var = 0.0_f64;
        let mut within_var = 0.0_f64;
        for (sum, count) in panel_resid_sums.values() {
            let mean = sum / *count as f64;
            between_var += mean * mean;
        }
        idx = 0;
        for i in 0..n {
            if z[i] {
                let panel_mean = panel_resid_sums.get(&panel_ids[i]).unwrap().0
                    / panel_resid_sums.get(&panel_ids[i]).unwrap().1 as f64;
                within_var += (residuals[idx] - panel_mean).powi(2);
                idx += 1;
            }
        }
        let n_sel_panels = panel_resid_sums.len();
        between_var /= n_sel_panels.max(1) as f64;
        within_var /= (n_selected - n_sel_panels).max(1) as f64;

        let sigma_alpha = between_var.sqrt().max(1e-8);
        let sigma_nu = within_var.sqrt().max(1e-8);

        // Log-likelihood (approximate)
        let ll_out = -(n_selected as f64) / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln()
            - residuals.dot(&residuals) / (2.0 * sigma2);
        let log_likelihood = ll_sel + ll_out;

        Ok(PanelHeckmanResult {
            gamma,
            beta,
            gamma_se,
            beta_se,
            gamma_t,
            beta_t,
            gamma_p,
            beta_p,
            rho,
            sigma,
            sigma_alpha,
            sigma_nu,
            log_likelihood,
            imr_mean,
            imr_coef,
            n_obs: n,
            n_selected,
            n_panels,
            sel_names,
            out_names,
        })
    }
}
