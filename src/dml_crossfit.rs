//! Double Machine Learning with cross-fitting (Chernozhukov et al. 2018).
//!
//! Estimates causal effects in the presence of high-dimensional
//! confounders by using ML to nuisance parameters, with cross-
//! fitting to avoid overfitting bias.
//!
//! Partially linear model:
//!   Y = theta * D + g(X) + epsilon
//!   D = m(X) + eta
//!
//! where g(X) and m(X) are nuisance functions estimated via ML.
//!
//! Cross-fitting procedure:
//! 1. Split sample into K folds
//! 2. For each fold k:
//!    a. Train g_hat and m_hat on the other K-1 folds
//!    b. Compute residuals on fold k: Y - g_hat(X), D - m_hat(X)
//! 3. Estimate theta via OLS of Y residuals on D residuals
//!
//! This avoids the "own observation" bias of standard DML.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of DML cross-fitting estimation.
#[derive(Debug)]
pub struct DmlResult {
    /// Causal effect estimate (theta)
    pub theta: f64,
    /// Standard error
    pub se: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value (two-sided)
    pub p_value: f64,
    /// 95% confidence interval
    pub ci: [f64; 2],
    /// Number of folds
    pub n_folds: usize,
    /// Nuisure MSE for g(X) (outcome model)
    pub g_mse: f64,
    /// Nuisure MSE for m(X) (treatment model)
    pub m_mse: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of confounders
    pub n_confounders: usize,
}

impl fmt::Display for DmlResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Double ML (Cross-fitting) ")?;
        writeln!(f, "Chernozhukov et al. (2018)")?;
        writeln!(f, "Partially linear model: Y = theta*D + g(X) + eps")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Confounders:", self.n_confounders)?;
        writeln!(f, "{:<20} {:>12}", "Folds:", self.n_folds)?;
        writeln!(f, "{:<20} {:>12.6}", "theta (causal effect):", self.theta)?;
        writeln!(f, "{:<20} {:>12.6}", "Std. Error:", self.se)?;
        writeln!(f, "{:<20} {:>12.3}", "t-statistic:", self.t_stat)?;
        writeln!(f, "{:<20} {:>12.4}", "p-value:", self.p_value)?;
        writeln!(
            f,
            "{:<20} [{:.4}, {:.4}]",
            "95% CI:", self.ci[0], self.ci[1]
        )?;
        writeln!(f, "{:<20} {:>12.6}", "g(X) nuisance MSE:", self.g_mse)?;
        writeln!(f, "{:<20} {:>12.6}", "m(X) nuisance MSE:", self.m_mse)?;

        write!(f, "{:=^78}", "")
    }
}

pub struct DML;

impl DML {
    /// Estimate DML with cross-fitting.
    ///
    /// # Arguments
    /// * `y` - Outcome (n)
    /// * `d` - Treatment variable (n)
    /// * `x` - Confounders (n x k)
    /// * `n_folds` - Number of cross-fitting folds (default 5)
    pub fn fit(
        y: &Array1<f64>,
        d: &Array1<f64>,
        x: &Array2<f64>,
        n_folds: Option<usize>,
    ) -> Result<DmlResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if d.len() != n || x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "DML: dimension mismatch".into(),
            ));
        }
        if n < 20 {
            return Err(GreenersError::InvalidOperation(
                "DML: need at least 20 observations".into(),
            ));
        }

        let folds = n_folds.unwrap_or(5).min(n / 5).max(2);

        // Create fold assignments (shuffle then split)
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..n {
            let j = i + Self::rand_int(n - i);
            indices.swap(i, j);
        }
        let fold_size = n / folds;
        let fold_assignment: Vec<usize> = (0..n)
            .map(|i| (i / fold_size.max(1)).min(folds - 1))
            .collect();

        // Reorder fold assignment to match shuffled indices
        let mut fold_of: Vec<usize> = vec![0; n];
        for (pos, &orig_idx) in indices.iter().enumerate() {
            fold_of[orig_idx] = fold_assignment[pos];
        }

        // Cross-fitting
        let mut y_resid = Array1::zeros(n);
        let mut d_resid = Array1::zeros(n);
        let mut g_mse_sum = 0.0_f64;
        let mut m_mse_sum = 0.0_f64;

        for fold in 0..folds {
            // Split into train (other folds) and test (this fold)
            let train_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] != fold).collect();
            let test_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] == fold).collect();

            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }

            // Train g_hat(X) -> Y on train, predict on test
            let g_hat = Self::fit_ols_nuisance(y, x, &train_idx)?;
            let m_hat = Self::fit_ols_nuisance(d, x, &train_idx)?;

            // Predict on test fold
            for &i in &test_idx {
                let x_i = x.row(i).to_owned();
                let g_pred = Self::predict_ols(&g_hat, &x_i);
                let m_pred = Self::predict_ols(&m_hat, &x_i);
                y_resid[i] = y[i] - g_pred;
                d_resid[i] = d[i] - m_pred;

                g_mse_sum += (y[i] - g_pred).powi(2);
                m_mse_sum += (d[i] - m_pred).powi(2);
            }
        }

        let g_mse = g_mse_sum / n as f64;
        let m_mse = m_mse_sum / n as f64;

        // Estimate theta: OLS of y_resid on d_resid
        // theta = sum(d_resid * y_resid) / sum(d_resid^2)
        let dd: f64 = d_resid.iter().map(|d| d * d).sum();
        let dy: f64 = d_resid.iter().zip(y_resid.iter()).map(|(d, y)| d * y).sum();

        if dd < 1e-15 {
            return Err(GreenersError::InvalidOperation(
                "DML: treatment residuals have zero variance".into(),
            ));
        }

        let theta = dy / dd;

        // SE: sqrt(1/n * var(eps) / var(d_resid))
        // where eps = y_resid - theta * d_resid
        let residuals: Vec<f64> = (0..n).map(|i| y_resid[i] - theta * d_resid[i]).collect();
        let var_eps = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
        let var_d = dd / n as f64;
        let se = (var_eps / (n as f64 * var_d)).sqrt();

        let t_stat = if se > 1e-10 { theta / se } else { 0.0 };
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));
        let z_crit = 1.959964;
        let ci = [theta - z_crit * se, theta + z_crit * se];

        Ok(DmlResult {
            theta,
            se,
            t_stat,
            p_value,
            ci,
            n_folds: folds,
            g_mse,
            m_mse,
            n_obs: n,
            n_confounders: k,
        })
    }

    /// Fit OLS for nuisance function (y ~ X with intercept).
    fn fit_ols_nuisance(
        y: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
    ) -> Result<Array1<f64>, GreenersError> {
        let n = indices.len();
        let k = x.ncols();
        let mut x_full = Array2::zeros((n, k + 1));
        let mut y_sub = Array1::zeros(n);
        for (i, &idx) in indices.iter().enumerate() {
            x_full[(i, 0)] = 1.0;
            for j in 0..k {
                x_full[(i, j + 1)] = x[(idx, j)];
            }
            y_sub[i] = y[idx];
        }

        let xt = x_full.t();
        let xtx = xt.dot(&x_full);
        let xtx_inv = (&xtx + Array2::<f64>::eye(k + 1) * 1e-8).inv()?;
        let xty = xt.dot(&y_sub);
        Ok(xtx_inv.dot(&xty))
    }

    /// Predict from OLS coefficients.
    fn predict_ols(beta: &Array1<f64>, x: &Array1<f64>) -> f64 {
        let mut pred = beta[0]; // intercept
        for j in 0..x.len() {
            pred += beta[j + 1] * x[j];
        }
        pred
    }

    fn rand_int(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (Self::rand_uniform() * n as f64) as usize
    }

    fn rand_uniform() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(2718281828) };
        }
        STATE.with(|s| {
            let mut state = s.get();
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.set(state);
            ((state >> 11) as f64) / (1u64 << 53) as f64
        })
    }
}
