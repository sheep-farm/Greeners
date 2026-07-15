//! Stochastic Volatility (SV) model via quasi-MCMC.
//!
//! Taylor (1986), Harvey, Ruiz & Shephard (1994). Models
//! time-varying volatility with a latent log-variance process:
//!
//!   y_t = exp(h_t/2) * eps_t,   eps_t ~ N(0, 1)
//!   h_t = mu + phi * (h_{t-1} - mu) + eta_t,   eta_t ~ N(0, sigma_eta^2)
//!
//! where h_t is the unobserved log-volatility following an AR(1).
//!
//! Estimation: hybrid approach —
//! 1. Initialize h_t via log(y_t^2) (proxy for h_t)
//! 2. Estimate (mu, phi, sigma_eta) via OLS on the AR(1) of h_t
//! 3. Refine via Gibbs-like sampling (simplified: a few iterations
//!    of single-site Metropolis-Hastings on h_t)
//!
//! This is a quasi-Bayesian approach that avoids full MCMC machinery
//! while producing reasonable estimates.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of Stochastic Volatility estimation.
#[derive(Debug)]
pub struct SvResult {
    /// Long-run mean of log-volatility (mu)
    pub mu: f64,
    /// Persistence parameter (phi)
    pub phi: f64,
    /// Volatility of volatility (sigma_eta)
    pub sigma_eta: f64,
    /// Estimated latent log-volatility path (T)
    pub log_vol: Array1<f64>,
    /// Conditional volatility path exp(h_t/2) (T)
    pub cond_vol: Array1<f64>,
    /// SE of mu
    pub mu_se: f64,
    /// SE of phi
    pub phi_se: f64,
    /// SE of sigma_eta
    pub sigma_eta_se: f64,
    /// t-value of phi (persistence)
    pub phi_t: f64,
    /// p-value of phi
    pub phi_p: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of MCMC iterations
    pub n_iter: usize,
    /// Variable name
    pub var_name: String,
}

impl fmt::Display for SvResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Stochastic Volatility (SV) ")?;
        writeln!(f, "Taylor (1986), Harvey-Ruiz-Shephard (1994)")?;
        writeln!(f, "Quasi-MCMC (Metropolis-Hastings on h_t)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "MCMC iterations:", self.n_iter)?;
        writeln!(f, "{:<20} {:>12.6}", "mu (long-run mean):", self.mu)?;
        writeln!(f, "{:<20} {:>12.6}", "phi (persistence):", self.phi)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "sigma_eta (vol-of-vol):", self.sigma_eta
        )?;
        writeln!(f, "{:<20} {:>12.6}", "phi SE:", self.phi_se)?;
        writeln!(f, "{:<20} {:>12.3}", "phi t-value:", self.phi_t)?;
        writeln!(f, "{:<20} {:>12.4}", "phi p-value:", self.phi_p)?;
        writeln!(f, "{:<20} {:>12.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        // Show volatility at selected periods
        let n_show = 5.min(self.n_obs);
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Conditional volatility (selected periods):")?;
        writeln!(f, "  {:<8} {:>12} {:>12}", "Period", "log_vol", "cond_vol")?;
        let indices: Vec<usize> = if self.n_obs <= n_show {
            (0..self.n_obs).collect()
        } else {
            (0..n_show)
                .map(|i| i * (self.n_obs - 1) / (n_show - 1).max(1))
                .collect()
        };
        for &idx in &indices {
            writeln!(
                f,
                "  {:<8} {:>12.6} {:>12.6}",
                idx + 1,
                self.log_vol[idx],
                self.cond_vol[idx]
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct SV;

impl SV {
    /// Estimate Stochastic Volatility model.
    ///
    /// # Arguments
    /// * `y` - Return series (T)
    /// * `n_iter` - Number of Metropolis-Hastings iterations
    /// * `var_name` - Optional variable name
    pub fn fit(
        y: &Array1<f64>,
        n_iter: usize,
        var_name: Option<String>,
    ) -> Result<SvResult, GreenersError> {
        let t = y.len();
        if t < 10 {
            return Err(GreenersError::InvalidOperation(
                "SV: need at least 10 observations".into(),
            ));
        }

        let name = var_name.unwrap_or_else(|| "y".to_string());

        // Step 1: Initialize h_t via log(y_t^2) with offset
        let offset = 0.001;
        let mut h: Array1<f64> = y.mapv(|v| (v * v + offset).ln());

        // Step 2: Estimate (mu, phi, sigma_eta) via OLS on AR(1) of h_t
        let (mu, phi, sigma_eta, _mu_se, _phi_se, _sigma_eta_se) = Self::estimate_ar1(&h)?;

        // Step 3: Refine h_t via Metropolis-Hastings
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        let sigma2_eta = sigma_eta * sigma_eta;
        let proposal_sd = 0.5; // tuning parameter

        let mut _accepted = 0usize;
        for _iter in 0..n_iter {
            for i in 0..t {
                let h_current = h[i];

                // Log-likelihood contribution from observation
                let ll_obs = |h_val: f64| -0.5 * h_val - 0.5 * (y[i] * y[i]) / h_val.exp();

                // Log-density of h_t | h_{t-1}, h_{t+1} (AR(1) chain)
                let ll_state = |h_val: f64| {
                    let mut lp = 0.0;
                    // h_t | h_{t-1}
                    if i > 0 {
                        let mean = mu + phi * (h[i - 1] - mu);
                        lp += -0.5 * (h_val - mean).powi(2) / sigma2_eta;
                    } else {
                        // Initial: stationary distribution
                        let var0 = sigma2_eta / (1.0 - phi * phi);
                        lp += -0.5 * (h_val - mu).powi(2) / var0;
                    }
                    // h_{t+1} | h_t
                    if i < t - 1 {
                        let mean_next = mu + phi * (h_val - mu);
                        lp += -0.5 * (h[i + 1] - mean_next).powi(2) / sigma2_eta;
                    }
                    lp
                };

                let log_post_current = ll_obs(h_current) + ll_state(h_current);

                // Propose new h_t
                let proposal = h_current + proposal_sd * Self::rand_normal(&normal);
                let log_post_proposal = ll_obs(proposal) + ll_state(proposal);

                // Accept/reject
                let log_accept = log_post_proposal - log_post_current;
                if log_accept > 0.0 || Self::rand_uniform() < log_accept.exp() {
                    h[i] = proposal;
                    _accepted += 1;
                }
            }
        }

        // Re-estimate parameters with refined h
        let (mu, phi, sigma_eta, mu_se, phi_se, sigma_eta_se) = Self::estimate_ar1(&h)?;

        // Conditional volatility
        let cond_vol = h.mapv(|v| (v / 2.0).exp());

        // Log-likelihood
        let mut ll = 0.0;
        let sigma2 = sigma_eta * sigma_eta;
        for i in 0..t {
            ll += -0.5 * h[i] - 0.5 * (y[i] * y[i]) / h[i].exp();
            if i > 0 {
                let mean = mu + phi * (h[i - 1] - mu);
                ll += -0.5 * (h[i] - mean).powi(2) / sigma2
                    - 0.5 * (2.0 * std::f64::consts::PI * sigma2).ln();
            }
        }

        let n_params = 3;
        let aic = -2.0 * ll + 2.0 * n_params as f64;
        let bic = -2.0 * ll + (t as f64) * n_params as f64;

        let phi_t = if phi_se > 1e-10 { phi / phi_se } else { 0.0 };
        let phi_p = 2.0 * (1.0 - normal.cdf(phi_t.abs()));

        Ok(SvResult {
            mu,
            phi,
            sigma_eta,
            log_vol: h,
            cond_vol,
            mu_se,
            phi_se,
            sigma_eta_se,
            phi_t,
            phi_p,
            log_likelihood: ll,
            aic,
            bic,
            n_obs: t,
            n_iter,
            var_name: name,
        })
    }

    /// Estimate AR(1) parameters via OLS.
    fn estimate_ar1(h: &Array1<f64>) -> Result<(f64, f64, f64, f64, f64, f64), GreenersError> {
        let t = h.len();
        if t < 3 {
            return Err(GreenersError::InvalidOperation(
                "SV: too few observations for AR(1)".into(),
            ));
        }

        let n = t - 1;
        let mut x = Array2::zeros((n, 2));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = h[i];
            y[i] = h[i + 1];
        }

        let xt = x.t();
        let xtx = xt.dot(&x);
        let xtx_inv = (&xtx + Array2::<f64>::eye(2) * 1e-10).inv()?;
        let xty = xt.dot(&y);
        let beta: Array1<f64> = xtx_inv.dot(&xty);

        let residuals = &y - x.dot(&beta);
        let sse = residuals.dot(&residuals);
        let sigma2 = sse / (n - 2) as f64;

        let se = xtx_inv.diag().mapv(|v| (v * sigma2).sqrt());

        // mu = beta[0] / (1 - beta[1]), phi = beta[1]
        let phi = beta[1];
        let mu = if (1.0 - phi).abs() > 1e-10 {
            beta[0] / (1.0 - phi)
        } else {
            beta[0]
        };

        // SE of mu (delta method): Var(mu) = Var(b0/(1-b1))
        let var_mu = (1.0 / (1.0 - phi).max(1e-6)).powi(2) * sigma2 * xtx_inv[(0, 0)]
            + (beta[0] / (1.0 - phi).max(1e-6).powi(2)).powi(2) * sigma2 * xtx_inv[(1, 1)];
        let mu_se = var_mu.sqrt();
        let phi_se = se[1];
        let sigma_eta = sigma2.sqrt();
        let sigma_eta_se = sigma_eta / (2.0 * (n - 2) as f64).sqrt();

        Ok((mu, phi, sigma_eta, mu_se, phi_se, sigma_eta_se))
    }

    fn rand_normal(normal: &Normal) -> f64 {
        // Use inverse CDF with uniform random
        let u = Self::rand_uniform().clamp(1e-10, 1.0 - 1e-10);
        normal.inverse_cdf(u)
    }

    fn rand_uniform() -> f64 {
        // Simple LCG random number generator
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(123456789) };
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
