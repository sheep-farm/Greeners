//! Spatial panel models: SAR and SEM with fixed or random effects.
//!
//! SAR panel:  y_it = ρ·W·y_it + x_it'β + μ_i + ε_it
//! SEM panel:  y_it = x_it'β + μ_i + u_it,  u_it = λ·W·u_it + ε_it
//!
//! where μ_i is the entity fixed effect, W is the spatial weights
//! matrix, and ε_it ~ N(0, σ²).
//!
//! Estimation: within transformation (demeaning by entity) removes
//! fixed effects, then grid search + golden section for spatial
//! parameter, OLS for β.

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of spatial panel estimation.
#[derive(Debug)]
pub struct SpatialPanelResult {
    /// Model type: "sar" or "sem"
    pub model_type: String,
    /// Spatial parameter (rho for SAR, lambda for SEM)
    pub spatial_param: f64,
    /// SE of spatial parameter
    pub spatial_se: f64,
    /// t-stat of spatial parameter
    pub spatial_t: f64,
    /// p-value of spatial parameter
    pub spatial_p: f64,
    /// Beta coefficients
    pub beta: Array1<f64>,
    /// Standard errors
    pub std_errors: Array1<f64>,
    /// t-statistics
    pub t_values: Array1<f64>,
    /// p-values
    pub p_values: Array1<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Sigma (residual std dev)
    pub sigma: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Number of time periods
    pub n_periods: usize,
    /// Variable names
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for SpatialPanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title = if self.model_type == "sar" {
            " Spatial Panel SAR (Fixed Effects) "
        } else {
            " Spatial Panel SEM (Fixed Effects) "
        };
        writeln!(f, "\n{:=^78}", title)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Entities:", self.n_entities)?;
        writeln!(f, "{:<20} {:>12}", "Periods:", self.n_periods)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.6}", "Sigma:", self.sigma)?;

        let spatial_name = if self.model_type == "sar" {
            "rho (spatial lag)"
        } else {
            "lambda (spatial error)"
        };
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            spatial_name, self.spatial_param, self.spatial_se, self.spatial_t, self.spatial_p
        )?;

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

pub struct SpatialPanel;

impl SpatialPanel {
    /// Estimate spatial panel SAR with fixed effects.
    ///
    /// y_it = ρ·W·y_it + x_it'β + μ_i + ε_it
    pub fn fit_sar(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        entity_ids: &[i64],
        variable_names: Option<Vec<String>>,
    ) -> Result<SpatialPanelResult, GreenersError> {
        Self::fit(y, x, w, entity_ids, "sar", variable_names)
    }

    /// Estimate spatial panel SEM with fixed effects.
    ///
    /// y_it = x_it'β + μ_i + u_it, u_it = λ·W·u_it + ε_it
    pub fn fit_sem(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        entity_ids: &[i64],
        variable_names: Option<Vec<String>>,
    ) -> Result<SpatialPanelResult, GreenersError> {
        Self::fit(y, x, w, entity_ids, "sem", variable_names)
    }

    fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        entity_ids: &[i64],
        model_type: &str,
        variable_names: Option<Vec<String>>,
    ) -> Result<SpatialPanelResult, GreenersError> {
        let n = y.len();
        if x.nrows() != n || entity_ids.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "SpatialPanel: dimension mismatch".into(),
            ));
        }

        // Identify entities
        let mut unique_ids: Vec<i64> = entity_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_ids.sort();
        let n_entities = unique_ids.len();
        let n_periods = n / n_entities;

        // Build full n×n W matrix (block diagonal: W repeated for each period)
        let w_full = if w.nrows() == n {
            w.clone()
        } else if w.nrows() == n_entities {
            // Block diagonal: W ⊗ I_T
            let mut w_block = Array2::zeros((n, n));
            for t in 0..n_periods {
                for i in 0..n_entities {
                    for j in 0..n_entities {
                        w_block[(t * n_entities + i, t * n_entities + j)] = w[(i, j)];
                    }
                }
            }
            w_block
        } else {
            return Err(GreenersError::ShapeMismatch(format!(
                "SpatialPanel: W must be {n}×{n} or {n_entities}×{n_entities}, got {}×{}",
                w.nrows(),
                w.ncols()
            )));
        };

        // Compute entity means for within transformation
        let mut entity_map: std::collections::HashMap<i64, (f64, usize)> =
            std::collections::HashMap::new();
        for i in 0..n {
            let entry = entity_map.entry(entity_ids[i]).or_insert((0.0, 0));
            entry.0 += y[i];
            entry.1 += 1;
        }
        let entity_means: std::collections::HashMap<i64, f64> = entity_map
            .iter()
            .map(|(&k, &(sum, count))| (k, sum / count as f64))
            .collect();

        // Demean y and x by entity
        let mut y_dm = Array1::zeros(n);
        let mut x_dm = Array2::zeros((n, x.ncols()));
        for i in 0..n {
            let mean = entity_means[&entity_ids[i]];
            y_dm[i] = y[i] - mean;
            for j in 0..x.ncols() {
                x_dm[(i, j)] = x[(i, j)];
            }
        }
        // Demean x by entity too
        for j in 0..x.ncols() {
            let mut col_means: std::collections::HashMap<i64, (f64, usize)> =
                std::collections::HashMap::new();
            for i in 0..n {
                let entry = col_means.entry(entity_ids[i]).or_insert((0.0, 0));
                entry.0 += x[(i, j)];
                entry.1 += 1;
            }
            for i in 0..n {
                let cm = col_means[&entity_ids[i]];
                x_dm[(i, j)] = x[(i, j)] - cm.0 / cm.1 as f64;
            }
        }

        // Grid search for spatial parameter
        let mut best_sp = 0.0_f64;
        let mut best_ll = f64::NEG_INFINITY;

        let n_grid = 199;
        for i in 0..n_grid {
            let sp = -0.99 + 1.98 * i as f64 / (n_grid - 1) as f64;
            let ll = Self::log_likelihood(&y_dm, &x_dm, &w_full, sp, model_type)?;
            if ll > best_ll {
                best_ll = ll;
                best_sp = sp;
            }
        }

        // Golden section refinement
        let golden = 0.6180339887498949;
        let mut a = best_sp - 0.05;
        let mut b = best_sp + 0.05;
        let mut c = b - golden * (b - a);
        let mut d = a + golden * (b - a);
        let mut fc = Self::log_likelihood(&y_dm, &x_dm, &w_full, c, model_type)?;
        let mut fd = Self::log_likelihood(&y_dm, &x_dm, &w_full, d, model_type)?;
        for _ in 0..50 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - golden * (b - a);
                fc = Self::log_likelihood(&y_dm, &x_dm, &w_full, c, model_type)?;
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + golden * (b - a);
                fd = Self::log_likelihood(&y_dm, &x_dm, &w_full, d, model_type)?;
            }
        }
        best_sp = if fc > fd { c } else { d };
        best_ll = if fc > fd { fc } else { fd };

        // Compute beta at optimal spatial parameter
        let k = x.ncols();
        let (beta, residuals, sigma2) = if model_type == "sar" {
            let wy = w_full.dot(&y_dm);
            let y_star = y_dm.clone() - best_sp * &wy;
            let xt = x_dm.t();
            let xtx = xt.dot(&x_dm);
            let xtx_reg = &xtx + Array2::eye(k) * 1e-8;
            let xtx_inv = xtx_reg.inv()?;
            let xty = xt.dot(&y_star);
            let beta: Array1<f64> = xtx_inv.dot(&xty);
            let fitted = x_dm.dot(&beta) + best_sp * &wy;
            let res = y_dm.clone() - fitted;
            let s2 = res.dot(&res) / (n - n_entities) as f64;
            (beta, res, s2)
        } else {
            // SEM: transform y and x by (I - λW)
            let i_minus_lw = Array2::eye(n) - best_sp * &w_full;
            let x_trans = i_minus_lw.dot(&x_dm);
            let y_trans = i_minus_lw.dot(&y_dm);
            let xt = x_trans.t();
            let xtx = xt.dot(&x_trans);
            let xtx_reg = &xtx + Array2::eye(k) * 1e-8;
            let xtx_inv = xtx_reg.inv()?;
            let xty = xt.dot(&y_trans);
            let beta: Array1<f64> = xtx_inv.dot(&xty);
            let res = y_dm.clone() - x_dm.dot(&beta);
            let s2 = res.dot(&res) / (n - n_entities) as f64;
            (beta, res, s2)
        };

        let sigma = sigma2.sqrt();

        // Standard errors for beta
        let xt = x_dm.t();
        let xtx = xt.dot(&x_dm);
        let xtx_inv = (&xtx + Array2::eye(k) * 1e-8).inv()?;
        let cov_beta = xtx_inv * sigma2;
        let std_errors = cov_beta.diag().mapv(|v| v.sqrt());
        let t_values = &beta / &std_errors;

        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values = t_values.mapv(|t| {
            if t.is_nan() || t.is_infinite() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal.cdf(t.abs()))
            }
        });

        // SE for spatial parameter (from Hessian)
        let sp_se = {
            let h = 0.01;
            let ll_p = Self::log_likelihood(&y_dm, &x_dm, &w_full, best_sp + h, model_type)?;
            let ll_m = Self::log_likelihood(&y_dm, &x_dm, &w_full, best_sp - h, model_type)?;
            let second_deriv = (ll_p - 2.0 * best_ll + ll_m) / (h * h);
            if second_deriv < 0.0 {
                (-1.0 / second_deriv).sqrt()
            } else {
                f64::NAN
            }
        };
        let sp_t = best_sp / sp_se;
        let sp_p = if sp_t.is_nan() || sp_t.is_infinite() {
            f64::NAN
        } else {
            2.0 * (1.0 - normal.cdf(sp_t.abs()))
        };

        // R-squared (within)
        let y_mean = y_dm.mean().unwrap_or(0.0);
        let tss = y_dm.mapv(|v| (v - y_mean).powi(2)).sum();
        let rss = residuals.dot(&residuals);
        let r_squared = if tss > 1e-15 { 1.0 - rss / tss } else { 0.0 };

        // Estimate n_periods (balanced panel assumption)
        let n_periods = n / n_entities;

        Ok(SpatialPanelResult {
            model_type: model_type.to_string(),
            spatial_param: best_sp,
            spatial_se: sp_se,
            spatial_t: sp_t,
            spatial_p: sp_p,
            beta,
            std_errors,
            t_values,
            p_values,
            r_squared,
            log_likelihood: best_ll,
            sigma,
            n_obs: n,
            n_entities,
            n_periods,
            variable_names,
        })
    }

    fn log_likelihood(
        y_dm: &Array1<f64>,
        x_dm: &Array2<f64>,
        w: &Array2<f64>,
        sp: f64,
        model_type: &str,
    ) -> Result<f64, GreenersError> {
        let n = y_dm.len();
        let k = x_dm.ncols();

        let (_beta, residuals) = if model_type == "sar" {
            let wy = w.dot(y_dm);
            let y_star = y_dm.clone() - sp * &wy;
            let xt = x_dm.t();
            let xtx = xt.dot(x_dm);
            // Regularized inverse to avoid singularity
            let xtx_reg = &xtx + Array2::eye(k) * 1e-8;
            let xtx_inv = xtx_reg.inv()?;
            let xty = xt.dot(&y_star);
            let beta: Array1<f64> = xtx_inv.dot(&xty);
            let fitted = x_dm.dot(&beta) + sp * &wy;
            let res = y_dm.clone() - fitted;
            (beta, res)
        } else {
            let i_minus_lw = Array2::eye(n) - sp * w;
            let x_trans = i_minus_lw.dot(x_dm);
            let y_trans = i_minus_lw.dot(y_dm);
            let xt = x_trans.t();
            let xtx = xt.dot(&x_trans);
            let xtx_reg = &xtx + Array2::eye(k) * 1e-8;
            let xtx_inv = xtx_reg.inv()?;
            let xty = xt.dot(&y_trans);
            let beta: Array1<f64> = xtx_inv.dot(&xty);
            let res = y_dm.clone() - x_dm.dot(&beta);
            (beta, res)
        };

        let rss = residuals.dot(&residuals);
        let sigma2 = rss / n as f64;

        let log_det = -(n as f64) * (1.0 - sp * sp).max(1e-10).ln() / 2.0;
        let ll = log_det
            - n as f64 / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln()
            - rss / (2.0 * sigma2);

        Ok(ll)
    }
}
