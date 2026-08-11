//! Spatial Durbin Model (SDM) for panel data with fixed effects.
//!
//! y_it = rho * W * y_it + x_it' * beta + W * x_it' * theta + mu_i + eps_it
//!
//! SDM nests SAR (theta=0) and SLX (rho=0). The spatially lagged
//! regressors W*X capture spatial spillovers from neighbors' covariates.
//!
//! Estimation: within transformation (demean by entity) + grid search
//! for rho, with W*X included as additional regressors.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of spatial panel Durbin model.
#[derive(Debug)]
pub struct SpatialDurbinResult {
    /// Spatial autoregressive parameter (rho)
    pub rho: f64,
    /// Direct effects (beta)
    pub beta: Array1<f64>,
    /// Indirect/spillover effects (theta)
    pub theta: Array1<f64>,
    /// SE of beta
    pub beta_se: Array1<f64>,
    /// SE of theta
    pub theta_se: Array1<f64>,
    /// t-values of beta
    pub beta_t: Array1<f64>,
    /// t-values of theta
    pub theta_t: Array1<f64>,
    /// p-values of beta
    pub beta_p: Array1<f64>,
    /// p-values of theta
    pub theta_p: Array1<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Number of regressors
    pub n_regressors: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for SpatialDurbinResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            " Spatial Durbin Model (Panel, Fixed Effects) "
        )?;
        writeln!(f, "y = rho*W*y + X*beta + W*X*theta + FE + eps")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Entities:", self.n_entities)?;
        writeln!(f, "{:<20} {:>12.6}", "rho (spatial AR):", self.rho)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<14} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Direct(beta)", "Coef.", "Std.Err.", "t", "P>|t|", "Indirect"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.beta.len() {
            let name = self
                .variable_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{i}"));
            writeln!(
                f,
                "{:<14} {:>10.6} {:>10.6} {:>10.3} {:>10.4} {:>10.6}",
                name, self.beta[i], self.beta_se[i], self.beta_t[i], self.beta_p[i], self.theta[i]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct SpatialDurbin;

impl SpatialDurbin {
    /// Estimate spatial panel Durbin model with fixed effects.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Regressors (n × k)
    /// * `w` - Spatial weights matrix (n_entities × n_entities or n × n)
    /// * `entity_ids` - Entity identifier (n)
    /// * `variable_names` - Optional names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        entity_ids: &[i64],
        variable_names: Option<Vec<String>>,
    ) -> Result<SpatialDurbinResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if x.nrows() != n || entity_ids.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "SpatialDurbin: dimension mismatch".into(),
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

        // Build full n×n W matrix (block diagonal)
        let w_full = if w.nrows() == n {
            w.clone()
        } else if w.nrows() == n_entities {
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
                "SpatialDurbin: W must be {n}x{n} or {n_entities}x{n_entities}, got {}x{}",
                w.nrows(),
                w.ncols()
            )));
        };

        // Compute W*X (spatially lagged regressors)
        let wx = w_full.dot(x);

        // Within transformation (demean by entity)
        let mut entity_sums: std::collections::HashMap<i64, (f64, usize)> =
            std::collections::HashMap::new();
        for i in 0..n {
            let entry = entity_sums.entry(entity_ids[i]).or_insert((0.0, 0));
            entry.0 += y[i];
            entry.1 += 1;
        }
        let entity_means: std::collections::HashMap<i64, f64> = entity_sums
            .iter()
            .map(|(&k, &(s, c))| (k, s / c as f64))
            .collect();

        let mut y_dm = Array1::zeros(n);
        for i in 0..n {
            y_dm[i] = y[i] - entity_means[&entity_ids[i]];
        }

        // Demean x and wx
        let mut x_dm = Array2::zeros((n, k));
        let mut wx_dm = Array2::zeros((n, k));
        for j in 0..k {
            let mut x_sums: std::collections::HashMap<i64, (f64, usize)> =
                std::collections::HashMap::new();
            let mut wx_sums: std::collections::HashMap<i64, (f64, usize)> =
                std::collections::HashMap::new();
            for i in 0..n {
                let xe = x_sums.entry(entity_ids[i]).or_insert((0.0, 0));
                xe.0 += x[(i, j)];
                xe.1 += 1;
                let we = wx_sums.entry(entity_ids[i]).or_insert((0.0, 0));
                we.0 += wx[(i, j)];
                we.1 += 1;
            }
            for i in 0..n {
                let xm = x_sums[&entity_ids[i]].0 / x_sums[&entity_ids[i]].1 as f64;
                let wm = wx_sums[&entity_ids[i]].0 / wx_sums[&entity_ids[i]].1 as f64;
                x_dm[(i, j)] = x[(i, j)] - xm;
                wx_dm[(i, j)] = wx[(i, j)] - wm;
            }
        }

        // Combined design matrix: [X_dm, WX_dm] (n x 2k)
        let mut x_combined = Array2::zeros((n, 2 * k));
        for i in 0..n {
            for j in 0..k {
                x_combined[(i, j)] = x_dm[(i, j)];
                x_combined[(i, k + j)] = wx_dm[(i, j)];
            }
        }

        // Grid search for rho
        let mut best_rho = 0.0_f64;
        let mut best_sse = f64::INFINITY;

        let n_grid = 41;
        for i in 0..n_grid {
            let rho = -1.0 + 2.0 * i as f64 / (n_grid - 1) as f64;
            let wy = w_full.dot(&y_dm);
            let y_star = &y_dm - rho * &wy;
            let xt = x_combined.t();
            let xtx = xt.dot(&x_combined);
            let xtx_reg = &xtx + Array2::eye(2 * k) * 1e-8;
            let xtx_inv = xtx_reg.inv()?;
            let xty = xt.dot(&y_star);
            let beta: Array1<f64> = xtx_inv.dot(&xty);
            let res = &y_star - x_combined.dot(&beta);
            let sse = res.dot(&res);
            if sse < best_sse {
                best_sse = sse;
                best_rho = rho;
            }
        }

        // Golden section refinement
        let golden = 0.6180339887498949;
        let mut a = (best_rho - 0.05).max(-0.99);
        let mut b = (best_rho + 0.05).min(0.99);
        let mut c = b - golden * (b - a);
        let mut d = a + golden * (b - a);
        for _ in 0..30 {
            let sse_c = Self::sse_at_rho(&y_dm, &x_combined, &w_full, c, 2 * k)?;
            let sse_d = Self::sse_at_rho(&y_dm, &x_combined, &w_full, d, 2 * k)?;
            if sse_c < sse_d {
                b = d;
                d = c;
                c = b - golden * (b - a);
            } else {
                a = c;
                c = d;
                d = a + golden * (b - a);
            }
        }
        best_rho = if Self::sse_at_rho(&y_dm, &x_combined, &w_full, c, 2 * k)?
            < Self::sse_at_rho(&y_dm, &x_combined, &w_full, d, 2 * k)?
        {
            c
        } else {
            d
        };

        // Final estimate at best_rho
        let wy = w_full.dot(&y_dm);
        let y_star = &y_dm - best_rho * &wy;
        let xt = x_combined.t();
        let xtx = xt.dot(&x_combined);
        let xtx_reg = &xtx + Array2::eye(2 * k) * 1e-8;
        let xtx_inv = xtx_reg.inv()?;
        let xty = xt.dot(&y_star);
        let best_beta = xtx_inv.dot(&xty);

        let residuals = &y_star - x_combined.dot(&best_beta);
        let sse = residuals.dot(&residuals);
        let sigma2 = sse / (n - n_entities - 2 * k) as f64;

        // SE
        let cov = &xtx_inv * sigma2;
        let std_errors = cov.diag().mapv(|v| v.sqrt());
        let t_values = &best_beta / &std_errors;
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        // Split beta into direct (beta) and indirect (theta)
        let beta = best_beta.slice(ndarray::s![0..k]).to_owned();
        let theta = best_beta.slice(ndarray::s![k..2 * k]).to_owned();
        let beta_se = std_errors.slice(ndarray::s![0..k]).to_owned();
        let theta_se = std_errors.slice(ndarray::s![k..2 * k]).to_owned();
        let beta_t = t_values.slice(ndarray::s![0..k]).to_owned();
        let theta_t = t_values.slice(ndarray::s![k..2 * k]).to_owned();
        let beta_p = p_values.slice(ndarray::s![0..k]).to_owned();
        let theta_p = p_values.slice(ndarray::s![k..2 * k]).to_owned();

        // R-squared
        let y_mean = y_dm.mean().unwrap_or(0.0);
        let tss = y_dm.mapv(|v| (v - y_mean).powi(2)).sum();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        let log_likelihood =
            -(n as f64) / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln() - sse / (2.0 * sigma2);

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{i}")).collect());

        Ok(SpatialDurbinResult {
            rho: best_rho,
            beta,
            theta,
            beta_se,
            theta_se,
            beta_t,
            theta_t,
            beta_p,
            theta_p,
            r_squared,
            log_likelihood,
            n_obs: n,
            n_entities,
            n_regressors: k,
            variable_names: names,
        })
    }

    fn sse_at_rho(
        y_dm: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        rho: f64,
        k: usize,
    ) -> Result<f64, GreenersError> {
        let wy = w.dot(y_dm);
        let y_star = y_dm - rho * &wy;
        let xt = x.t();
        let xtx = xt.dot(x);
        let xtx_reg = &xtx + Array2::eye(k) * 1e-8;
        let xtx_inv = xtx_reg.inv()?;
        let xty = xt.dot(&y_star);
        let beta = xtx_inv.dot(&xty);
        let res = &y_star - x.dot(&beta);
        Ok(res.dot(&res))
    }
}
