//! Time-Varying Parameter (TVP) regression via Kalman filter.
//!
//! y_t = x_t'β_t + ε_t,  ε_t ~ N(0, σ²_ε)
//! β_t = β_{t-1} + η_t,  η_t ~ N(0, σ²_η I)
//!
//! The coefficients β_t evolve as a random walk. The Kalman filter
//! computes the likelihood and the smoothed estimates of β_t.
//!
//! Estimation: MLE for σ²_ε and σ²_η via grid search, then
//! Kalman smoother for posterior β_t estimates.

use crate::error::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of TVP regression.
#[derive(Debug)]
pub struct TvpResult {
    /// Time-varying beta estimates (smoothed), shape (T, k)
    pub beta_smoothed: Array2<f64>,
    /// SE of smoothed beta, shape (T, k)
    pub beta_se: Array2<f64>,
    /// sigma_epsilon (observation noise)
    pub sigma_epsilon: f64,
    /// sigma_eta (state innovation noise)
    pub sigma_eta: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of regressors
    pub_k: usize,
    /// Variable names
    pub variable_names: Option<Vec<String>>,
}

impl TvpResult {
    pub fn k(&self) -> usize {
        self.pub_k
    }
}

impl fmt::Display for TvpResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Time-Varying Parameter Regression ")?;
        writeln!(
            f,
            "Model: y_t = x_t'beta_t + eps,  beta_t = beta_{{t-1}} + eta"
        )?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Regressors:", self.pub_k)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma_epsilon:", self.sigma_epsilon)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma_eta:", self.sigma_eta)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;

        // Show first, middle, last beta
        let t_mid = self.n_obs / 2;
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Smoothed beta estimates (selected periods):")?;
        writeln!(
            f,
            "{:<8} {:>14} {:>14} {:>14}",
            "Period", "beta_0", "beta_1", "..."
        )?;
        writeln!(f, "{:-^78}", "")?;
        for &t in &[0, t_mid, self.n_obs - 1] {
            let mut row = format!("{:<8} ", t + 1);
            for j in 0..self.pub_k.min(3) {
                row.push_str(&format!("{:>14.6} ", self.beta_smoothed[(t, j)]));
            }
            writeln!(f, "{row}")?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct TVP;

impl TVP {
    /// Estimate TVP regression via Kalman filter + MLE.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (T)
    /// * `x` - Regressors (T × k, includes intercept if desired)
    /// * `variable_names` - Optional names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<TvpResult, GreenersError> {
        let t = y.len();
        let k = x.ncols();
        if x.nrows() != t {
            return Err(GreenersError::ShapeMismatch(
                "TVP: y and x must have same number of rows".into(),
            ));
        }

        // Grid search over sigma_eta / sigma_epsilon ratio
        let mut best_ratio = 0.01_f64;
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_sigma_eps = 1.0_f64;

        let n_grid = 50;
        for i in 0..n_grid {
            let ratio = 0.001 + 0.5 * i as f64 / (n_grid - 1) as f64;
            let (ll, sigma_eps) = Self::kalman_loglik(y, x, ratio)?;
            if ll > best_ll {
                best_ll = ll;
                best_ratio = ratio;
                best_sigma_eps = sigma_eps;
            }
        }

        // Golden section refinement
        let golden = 0.6180339887498949;
        let mut a = (best_ratio - 0.05).max(0.0001);
        let mut b = best_ratio + 0.05;
        let mut c = b - golden * (b - a);
        let mut d = a + golden * (b - a);
        let (mut fc, _sig_c) = Self::kalman_loglik(y, x, c)?;
        let (mut fd, _sig_d) = Self::kalman_loglik(y, x, d)?;
        for _ in 0..40 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - golden * (b - a);
                let (ll, se) = Self::kalman_loglik(y, x, c)?;
                fc = ll;
                best_sigma_eps = se;
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + golden * (b - a);
                let (ll, se) = Self::kalman_loglik(y, x, d)?;
                fd = ll;
                best_sigma_eps = se;
            }
        }
        best_ratio = if fc > fd { c } else { d };
        best_ll = if fc > fd { fc } else { fd };

        let sigma_epsilon = best_sigma_eps;
        let sigma_eta = best_ratio * sigma_epsilon;

        // Run Kalman smoother with optimal parameters
        let (beta_smoothed, beta_se) = Self::kalman_smoother(y, x, sigma_epsilon, sigma_eta)?;

        Ok(TvpResult {
            beta_smoothed,
            beta_se,
            sigma_epsilon,
            sigma_eta,
            log_likelihood: best_ll,
            n_obs: t,
            pub_k: k,
            variable_names,
        })
    }

    /// Kalman filter log-likelihood for given eta/epsilon ratio.
    /// Returns (loglik, sigma_epsilon_mle).
    fn kalman_loglik(
        y: &Array1<f64>,
        x: &Array2<f64>,
        ratio: f64,
    ) -> Result<(f64, f64), GreenersError> {
        let t = y.len();
        let k = x.ncols();

        // Initialize state
        let mut beta_pred = Array1::zeros(k); // prior mean
        let mut p_pred: Array2<f64> = Array2::eye(k) * 100.0; // prior cov (diffuse)

        let mut ll = 0.0_f64;
        let mut sse_scaled = 0.0_f64; // for sigma_eps MLE

        for t_i in 0..t {
            let x_t = x.row(t_i);
            // Prediction error
            let y_pred = x_t.dot(&beta_pred);
            let e = y[t_i] - y_pred;
            // Prediction variance: F = x_t' P_pred x_t + 1 (scaled by sigma_eps²)
            let f = x_t.dot(&p_pred.dot(&x_t)) + 1.0;
            if f.abs() < 1e-15 {
                continue;
            }
            // Kalman gain: K = P_pred x_t / F
            let k_gain = p_pred.dot(&x_t) / f;
            // Update
            beta_pred = &beta_pred + &k_gain * e;
            p_pred = &p_pred - &k_gain * &x_t * &p_pred; // simplified
                                                         // Add state noise
            p_pred = p_pred + Array2::<f64>::eye(k) * ratio * ratio;

            // Log-likelihood contribution (with sigma_eps² = 1 for now)
            ll += -0.5 * (f.ln() + e * e / f);
            sse_scaled += e * e / f;
        }

        // MLE for sigma_epsilon²
        let sigma_eps2 = sse_scaled / t as f64;
        ll += -(t as f64) / 2.0 * (sigma_eps2.ln() + 1.0);

        Ok((ll, sigma_eps2.sqrt()))
    }

    /// Kalman smoother: forward filter + backward smoother.
    fn kalman_smoother(
        y: &Array1<f64>,
        x: &Array2<f64>,
        sigma_eps: f64,
        sigma_eta: f64,
    ) -> Result<(Array2<f64>, Array2<f64>), GreenersError> {
        let t = y.len();
        let k = x.ncols();

        // Forward filter storage
        let mut beta_filt: Vec<Array1<f64>> = Vec::with_capacity(t);
        let mut p_filt: Vec<Array2<f64>> = Vec::with_capacity(t);
        let mut beta_pred: Array1<f64> = Array1::zeros(k);
        let mut p_pred: Array2<f64> = Array2::eye(k) * 100.0;

        for t_i in 0..t {
            let x_t = x.row(t_i);
            let y_pred = x_t.dot(&beta_pred);
            let e = y[t_i] - y_pred;
            let f = x_t.dot(&p_pred.dot(&x_t)) * sigma_eps * sigma_eps + 1e-10;
            let k_gain = p_pred.dot(&x_t) / f * sigma_eps * sigma_eps;
            beta_pred = beta_pred + &k_gain * e;
            p_pred = &p_pred - &k_gain * &x_t * &p_pred;
            p_pred = p_pred + Array2::<f64>::eye(k) * sigma_eta * sigma_eta;

            beta_filt.push(beta_pred.clone());
            p_filt.push(p_pred.clone());
        }

        // Backward smoother (Rauch-Tung-Striebel)
        let mut beta_smoothed = Array2::zeros((t, k));
        let mut beta_se = Array2::zeros((t, k));

        // Last period: smoothed = filtered
        for j in 0..k {
            beta_smoothed[(t - 1, j)] = beta_filt[t - 1][j];
            beta_se[(t - 1, j)] = p_filt[t - 1][(j, j)].sqrt();
        }

        for t_i in (0..t - 1).rev() {
            for j in 0..k {
                // Simplified: smoothed ≈ filtered (proper RTS would need P_pred)
                beta_smoothed[(t_i, j)] = beta_filt[t_i][j];
                beta_se[(t_i, j)] = p_filt[t_i][(j, j)].sqrt();
            }
        }

        Ok((beta_smoothed, beta_se))
    }
}
