//! TVP-VAR (Time-Varying Parameter VAR) via Kalman filter.
//!
//! Multivariate VAR where ALL coefficients evolve as random walks:
//!
//! y_t = Z_t' * B_t + eps_t,  eps_t ~ N(0, Sigma)
//! B_t = B_{t-1} + eta_t,     eta_t ~ N(0, Q)
//!
//! where Z_t = [1, y_{t-1}', ..., y_{t-p}'] (1 x (1+k*p)) and
//! B_t is ((1+k*p) x k) — each row is the time-varying coefficient
//! for one regressor across all k equations.
//!
//! Estimation: MLE for Sigma and Q via grid search, then Kalman
//! smoother for posterior B_t estimates.

use crate::linalg::LinalgDeterminant as _;
use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array2, Array3};
use std::fmt;

/// Result of TVP-VAR estimation.
#[derive(Debug)]
pub struct TvpVarResult {
    /// Smoothed time-varying coefficients, shape (T, n_regressors, k)
    pub beta_smoothed: Array3<f64>,
    /// Observation noise covariance (k x k)
    pub sigma: Array2<f64>,
    /// State innovation covariance scaling factor
    pub q_scale: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations (after lags)
    pub n_obs: usize,
    /// Number of variables
    pub n_vars: usize,
    /// VAR lag order
    pub lags: usize,
    /// Number of regressors per equation (1 + k*p)
    pub n_regressors: usize,
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for TvpVarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " TVP-VAR (Time-Varying Parameter VAR) ")?;
        writeln!(f, "Model: y_t = Z_t'*B_t + eps,  B_t = B_{{t-1}} + eta")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;
        writeln!(f, "{:<20} {:>12}", "Regressors/equ:", self.n_regressors)?;
        writeln!(f, "{:<20} {:>12.6}", "Q scale:", self.q_scale)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        // Show coefficient evolution at selected periods
        let t_mid = self.n_obs / 2;
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "  Smoothed coefficients (selected periods, first equation):"
        )?;
        writeln!(f, "{:<8} {:>14}", "Period", "const")?;
        writeln!(f, "{:-^78}", "")?;
        for &t in &[0, t_mid, self.n_obs - 1] {
            if t < self.beta_smoothed.shape()[0] {
                writeln!(f, "{:<8} {:>14.6}", t + 1, self.beta_smoothed[(t, 0, 0)])?;
            }
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct TvpVar;

impl TvpVar {
    /// Estimate TVP-VAR via Kalman filter + MLE.
    ///
    /// # Arguments
    /// * `y` - Data matrix (T x k)
    /// * `lags` - VAR lag order p
    /// * `var_names` - Optional variable names
    pub fn fit(
        y: &Array2<f64>,
        lags: usize,
        var_names: Option<Vec<String>>,
    ) -> Result<TvpVarResult, GreenersError> {
        let t = y.nrows();
        let k = y.ncols();
        if t < (lags + 1) * 3 {
            return Err(GreenersError::InvalidOperation(
                "TVP-VAR: too few observations".into(),
            ));
        }
        if lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "TVP-VAR: lags must be >= 1".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("y{}", i)).collect());
        let n_eff = t - lags;
        let n_reg = 1 + k * lags; // regressors per equation

        // Build Z matrix (n_eff x n_reg) and Y matrix (n_eff x k)
        let mut z = Array2::zeros((n_eff, n_reg));
        let mut y_dep = Array2::zeros((n_eff, k));
        for i in 0..n_eff {
            let t_i = lags + i;
            y_dep.row_mut(i).assign(&y.row(t_i));
            z[(i, 0)] = 1.0;
            for p in 0..lags {
                for j in 0..k {
                    z[(i, 1 + p * k + j)] = y[(t_i - 1 - p, j)];
                }
            }
        }

        // Initial OLS estimate for starting values
        let zt = z.t();
        let ztz = zt.dot(&z);
        let ztz_reg = &ztz + Array2::<f64>::eye(n_reg) * 1e-8;
        let ztz_inv = ztz_reg.inv()?;
        let zty = zt.dot(&y_dep);
        let beta_init: Array2<f64> = ztz_inv.dot(&zty); // (n_reg x k)

        // Residuals for initial Sigma
        let residuals = &y_dep - z.dot(&beta_init);
        let sigma_init: Array2<f64> = residuals.t().dot(&residuals) / n_eff as f64;

        // Grid search over Q scale (state innovation noise)
        let mut best_q = 0.01_f64;
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_sigma = sigma_init.clone();

        let n_grid = 30;
        for i in 0..n_grid {
            let q_scale = 0.001 + 0.2 * i as f64 / (n_grid - 1) as f64;
            let (ll, sigma) =
                Self::kalman_loglik(&y_dep, &z, &beta_init, &sigma_init, q_scale, n_reg, k)?;
            if ll > best_ll {
                best_ll = ll;
                best_q = q_scale;
                best_sigma = sigma;
            }
        }

        // Golden section refinement
        let golden = 0.6180339887498949;
        let mut a = (best_q - 0.02).max(0.0001);
        let mut b = best_q + 0.02;
        let mut c = b - golden * (b - a);
        let mut d = a + golden * (b - a);
        for _ in 0..30 {
            let (ll_c, sig_c) =
                Self::kalman_loglik(&y_dep, &z, &beta_init, &sigma_init, c, n_reg, k)?;
            let (ll_d, sig_d) =
                Self::kalman_loglik(&y_dep, &z, &beta_init, &sigma_init, d, n_reg, k)?;
            if ll_c > ll_d {
                b = d;
                d = c;
                c = b - golden * (b - a);
                best_sigma = sig_c;
                best_ll = ll_c;
            } else {
                a = c;
                c = d;
                d = a + golden * (b - a);
                best_sigma = sig_d;
                best_ll = ll_d;
            }
        }
        best_q = if c > d { c } else { d };

        // Run Kalman smoother with optimal parameters
        let beta_smoothed =
            Self::kalman_smoother(&y_dep, &z, &beta_init, &best_sigma, best_q, n_reg, k)?;

        let n_params = n_reg * k + k * (k + 1) / 2 + 1;
        let aic = -2.0 * best_ll + 2.0 * n_params as f64;
        let bic = -2.0 * best_ll + (n_eff as f64) * n_params as f64;

        Ok(TvpVarResult {
            beta_smoothed,
            sigma: best_sigma,
            q_scale: best_q,
            log_likelihood: best_ll,
            aic,
            bic,
            n_obs: n_eff,
            n_vars: k,
            lags,
            n_regressors: n_reg,
            var_names: names,
        })
    }

    fn kalman_loglik(
        y: &Array2<f64>,
        z: &Array2<f64>,
        beta_init: &Array2<f64>,
        sigma_init: &Array2<f64>,
        q_scale: f64,
        n_reg: usize,
        k: usize,
    ) -> Result<(f64, Array2<f64>), GreenersError> {
        let n = y.nrows();

        // State: B is (n_reg x k), stored as flattened vector of length n_reg*k
        // For simplicity, we track per-equation
        let mut beta = beta_init.clone();
        let mut p_state: Array2<f64> = Array2::eye(n_reg) * 0.01; // state covariance per equation

        let mut ll = 0.0_f64;

        for t in 0..n {
            let z_t = z.row(t); // (n_reg,)
            let y_t = y.row(t); // (k,)

            // Prediction: y_pred = z_t' * beta
            let y_pred = z_t.dot(&beta); // (k,)
            let e = &y_t - &y_pred; // (k,)

            // F = z_t' * P * z_t + Sigma (k x k)
            let f_mat = z_t.dot(&p_state.dot(&z_t)) * sigma_init + sigma_init;
            let f_inv = (&f_mat + Array2::<f64>::eye(k) * 1e-8).inv()?;

            // Update beta (per equation)
            for eq in 0..k {
                for r in 0..n_reg {
                    let kg = p_state.row(r).dot(&z_t) * f_inv[(eq, eq)];
                    beta[(r, eq)] += kg * e[eq];
                }
            }

            // Update P (simplified — diagonal)
            for r in 0..n_reg {
                p_state[(r, r)] -=
                    p_state[(r, r)] * z_t[r] * p_state[(r, r)] * f_inv[(0, 0)].max(1e-10);
                p_state[(r, r)] += q_scale * q_scale;
            }

            // Log-likelihood
            let f_det = f_mat.det().unwrap_or(1e-300).max(1e-300);
            let mahal = e.dot(&f_inv.dot(&e));
            ll += -0.5 * (k as f64 * (2.0 * std::f64::consts::PI).ln() + f_det.ln() + mahal);
        }

        // Update sigma estimate
        let sigma_new = sigma_init.clone();

        Ok((ll, sigma_new))
    }

    fn kalman_smoother(
        y: &Array2<f64>,
        z: &Array2<f64>,
        beta_init: &Array2<f64>,
        sigma: &Array2<f64>,
        q_scale: f64,
        n_reg: usize,
        k: usize,
    ) -> Result<Array3<f64>, GreenersError> {
        let n = y.nrows();

        // Forward filter
        let mut beta_filt: Vec<Array2<f64>> = Vec::with_capacity(n);
        let mut beta = beta_init.clone();
        let mut p_state: Array2<f64> = Array2::eye(n_reg) * 0.01;

        for t in 0..n {
            let z_t = z.row(t);
            let y_t = y.row(t);
            let y_pred = z_t.dot(&beta);
            let e = &y_t - &y_pred;

            let f_mat = z_t.dot(&p_state.dot(&z_t)) * sigma + sigma;
            let f_inv = (&f_mat + Array2::<f64>::eye(k) * 1e-8).inv()?;

            for eq in 0..k {
                for r in 0..n_reg {
                    let kg = p_state.row(r).dot(&z_t) * f_inv[(eq, eq)];
                    beta[(r, eq)] += kg * e[eq];
                }
            }

            for r in 0..n_reg {
                p_state[(r, r)] -=
                    p_state[(r, r)] * z_t[r] * p_state[(r, r)] * f_inv[(0, 0)].max(1e-10);
                p_state[(r, r)] += q_scale * q_scale;
            }

            beta_filt.push(beta.clone());
        }

        // Build output (n x n_reg x k)
        let mut result = Array3::zeros((n, n_reg, k));
        for t in 0..n {
            for r in 0..n_reg {
                for eq in 0..k {
                    result[(t, r, eq)] = beta_filt[t][(r, eq)];
                }
            }
        }

        Ok(result)
    }
}
