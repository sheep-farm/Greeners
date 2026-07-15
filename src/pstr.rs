//! Panel Smooth Transition Regression (PSTR) with fixed effects.
//!
//! y_it = mu_i + beta_0' * x_it + beta_1' * x_it * g(q_it; gamma, c) + eps_it
//!
//! where g(q_it; gamma, c) = 1 / (1 + exp(-gamma * (q_it - c)))
//! is the logistic transition function, q_it is the transition
//! variable, gamma > 0 is the smoothness parameter, and c is the
//! threshold.
//!
//! For gamma -> infinity, PSTR collapses to a panel threshold model
//! (PTR). For gamma -> 0, it collapses to a linear panel model with
//! homogeneous coefficients.
//!
//! Estimation: nonlinear least squares (NLS) via grid search on
//! (gamma, c) + within transformation for fixed effects.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of PSTR estimation.
#[derive(Debug)]
pub struct PstrResult {
    /// Smoothness parameter gamma
    pub gamma: f64,
    /// Threshold parameter c
    pub c: f64,
    /// Coefficients in regime 0 (linear part): beta_0
    pub beta0: Array1<f64>,
    /// Coefficients in regime 1 (nonlinear part): beta_1
    pub beta1: Array1<f64>,
    /// SE of beta_0
    pub beta0_se: Array1<f64>,
    /// SE of beta_1
    pub beta1_se: Array1<f64>,
    /// t-values of beta_0
    pub beta0_t: Array1<f64>,
    /// t-values of beta_1
    pub beta1_t: Array1<f64>,
    /// p-values of beta_0
    pub beta0_p: Array1<f64>,
    /// p-values of beta_1
    pub beta1_p: Array1<f64>,
    /// Transition variable name
    pub transition_var: String,
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

impl fmt::Display for PstrResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            " Panel Smooth Transition Regression (PSTR) "
        )?;
        writeln!(f, "y = mu_i + beta0'*x + beta1'*x*g(q;gamma,c) + eps")?;
        writeln!(f, "g(q) = 1/(1+exp(-gamma*(q-c)))")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Entities:", self.n_entities)?;
        writeln!(f, "{:<20} {:>12.6}", "gamma (smoothness):", self.gamma)?;
        writeln!(f, "{:<20} {:>12.6}", "c (threshold):", self.c)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "beta0", "SE0", "t0", "P>|t|0", "beta1", "SE1", "t1"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.beta0.len() {
            let name = self
                .variable_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<14} {:>10.6} {:>10.6} {:>10.3} {:>10.4} {:>10.6} {:>10.6} {:>10.3}",
                name,
                self.beta0[i],
                self.beta0_se[i],
                self.beta0_t[i],
                self.beta0_p[i],
                self.beta1[i],
                self.beta1_se[i],
                self.beta1_t[i]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct PSTR;

impl PSTR {
    /// Estimate PSTR with fixed effects via NLS + grid search.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Regressors (n x k)
    /// * `q` - Transition variable (n)
    /// * `entity_ids` - Entity identifier (n)
    /// * `variable_names` - Optional names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        q: &Array1<f64>,
        entity_ids: &[i64],
        variable_names: Option<Vec<String>>,
    ) -> Result<PstrResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if x.nrows() != n || q.len() != n || entity_ids.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "PSTR: dimension mismatch".into(),
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
            .map(|(&key, &(s, c))| (key, s / c as f64))
            .collect();

        let mut y_dm = Array1::zeros(n);
        for i in 0..n {
            y_dm[i] = y[i] - entity_means[&entity_ids[i]];
        }

        // Demean x and q
        let mut x_dm = Array2::zeros((n, k));
        let mut q_dm = Array1::zeros(n);
        for j in 0..k {
            let mut x_sums: std::collections::HashMap<i64, (f64, usize)> =
                std::collections::HashMap::new();
            for i in 0..n {
                let xe = x_sums.entry(entity_ids[i]).or_insert((0.0, 0));
                xe.0 += x[(i, j)];
                xe.1 += 1;
            }
            for i in 0..n {
                let xm = x_sums[&entity_ids[i]].0 / x_sums[&entity_ids[i]].1 as f64;
                x_dm[(i, j)] = x[(i, j)] - xm;
            }
        }
        // Demean q
        {
            let mut q_sums: std::collections::HashMap<i64, (f64, usize)> =
                std::collections::HashMap::new();
            for i in 0..n {
                let qe = q_sums.entry(entity_ids[i]).or_insert((0.0, 0));
                qe.0 += q[i];
                qe.1 += 1;
            }
            for i in 0..n {
                let qm = q_sums[&entity_ids[i]].0 / q_sums[&entity_ids[i]].1 as f64;
                q_dm[i] = q[i] - qm;
            }
        }

        // Grid search over (gamma, c)
        // c: percentiles of q_dm (15th to 85th)
        let mut q_sorted: Vec<f64> = q_dm.iter().copied().collect();
        q_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_c_grid = 9;
        let c_grid: Vec<f64> = (0..n_c_grid)
            .map(|i| {
                let idx = (q_sorted.len() as f64 * (0.15 + 0.7 * i as f64 / (n_c_grid - 1) as f64))
                    as usize;
                q_sorted[idx.min(q_sorted.len() - 1)]
            })
            .collect();

        let n_gamma_grid = 10;
        let gamma_grid: Vec<f64> = (0..n_gamma_grid)
            .map(|i| 0.5 + 10.0 * i as f64 / (n_gamma_grid - 1) as f64)
            .collect();

        let mut best_gamma = 1.0_f64;
        let mut best_c = 0.0_f64;
        let mut best_sse = f64::INFINITY;

        for &gamma in &gamma_grid {
            for &c in &c_grid {
                // Compute transition function g_t
                let g: Array1<f64> = (0..n)
                    .map(|i| 1.0 / (1.0 + (-gamma * (q_dm[i] - c)).exp()))
                    .collect();

                // Design matrix: [x_dm, x_dm * g_t] (n x 2k)
                let mut x_combined = Array2::zeros((n, 2 * k));
                for i in 0..n {
                    for j in 0..k {
                        x_combined[(i, j)] = x_dm[(i, j)];
                        x_combined[(i, k + j)] = x_dm[(i, j)] * g[i];
                    }
                }

                // OLS
                let xt = x_combined.t();
                let xtx = xt.dot(&x_combined);
                let xtx_reg = &xtx + Array2::eye(2 * k) * 1e-8;
                let xtx_inv = xtx_reg.inv()?;
                let xty = xt.dot(&y_dm);
                let beta: Array1<f64> = xtx_inv.dot(&xty);
                let res = &y_dm - x_combined.dot(&beta);
                let sse = res.dot(&res);

                if sse < best_sse {
                    best_sse = sse;
                    best_gamma = gamma;
                    best_c = c;
                }
            }
        }

        // Final estimate at best (gamma, c)
        let g: Array1<f64> = (0..n)
            .map(|i| 1.0 / (1.0 + (-best_gamma * (q_dm[i] - best_c)).exp()))
            .collect();

        let mut x_combined = Array2::zeros((n, 2 * k));
        for i in 0..n {
            for j in 0..k {
                x_combined[(i, j)] = x_dm[(i, j)];
                x_combined[(i, k + j)] = x_dm[(i, j)] * g[i];
            }
        }

        let xt = x_combined.t();
        let xtx = xt.dot(&x_combined);
        let xtx_reg = &xtx + Array2::eye(2 * k) * 1e-8;
        let xtx_inv = xtx_reg.inv()?;
        let xty = xt.dot(&y_dm);
        let beta_full: Array1<f64> = xtx_inv.dot(&xty);

        let residuals = &y_dm - x_combined.dot(&beta_full);
        let sse = residuals.dot(&residuals);
        let sigma2 = sse / (n - n_entities - 2 * k) as f64;

        // SE
        let cov = &xtx_inv * sigma2;
        let std_errors = cov.diag().mapv(|v| v.sqrt());
        let t_values = &beta_full / &std_errors;
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        // Split into beta0 and beta1
        let beta0 = beta_full.slice(ndarray::s![0..k]).to_owned();
        let beta1 = beta_full.slice(ndarray::s![k..2 * k]).to_owned();
        let beta0_se = std_errors.slice(ndarray::s![0..k]).to_owned();
        let beta1_se = std_errors.slice(ndarray::s![k..2 * k]).to_owned();
        let beta0_t = t_values.slice(ndarray::s![0..k]).to_owned();
        let beta1_t = t_values.slice(ndarray::s![k..2 * k]).to_owned();
        let beta0_p = p_values.slice(ndarray::s![0..k]).to_owned();
        let beta1_p = p_values.slice(ndarray::s![k..2 * k]).to_owned();

        // R-squared
        let y_mean = y_dm.mean().unwrap_or(0.0);
        let tss = y_dm.mapv(|v| (v - y_mean).powi(2)).sum();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        let log_likelihood =
            -(n as f64) / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln() - sse / (2.0 * sigma2);

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());

        Ok(PstrResult {
            gamma: best_gamma,
            c: best_c,
            beta0,
            beta1,
            beta0_se,
            beta1_se,
            beta0_t,
            beta1_t,
            beta0_p,
            beta1_p,
            transition_var: "q".to_string(),
            r_squared,
            log_likelihood,
            n_obs: n,
            n_entities,
            n_regressors: k,
            variable_names: names,
        })
    }
}
