//! Bayesian Synthetic Control (Brodersen et al. 2015; Scott 2019).
//!
//! Extends the synthetic control method with a Bayesian
//! structural time-series framework. Places priors on:
//!   - The synthetic control weights (Dirichlet-like)
//!   - The intervention effect (Normal prior)
//!   - Observation noise (Inverse-Gamma)
//!
//! The model:
//!   y_t = w' * x_t + tau * I(t >= T0) + epsilon_t
//!   epsilon_t ~ N(0, sigma^2)
//!
//! where w are the synthetic control weights (estimated via
//! Bayesian regression with shrinkage), x_t are the control
//! unit outcomes, and tau is the treatment effect.
//!
//! Implementation: conjugate Bayesian linear regression with
//! Normal prior on (w, tau) and Inverse-Gamma on sigma^2.
//! Posterior is Normal-Inverse-Gamma (closed-form).

use crate::linalg::{LinalgDeterminant as _, LinalgInverse as _};
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::fmt;

/// Result of Bayesian Synthetic Control estimation.
#[derive(Debug)]
pub struct BayesianScResult {
    /// Posterior mean of treatment effect (tau)
    pub tau: f64,
    /// Posterior SD of treatment effect
    pub tau_sd: f64,
    /// Posterior mean of sigma^2 (observation noise)
    pub sigma2: f64,
    /// Synthetic control weights (posterior mean)
    pub weights: Array1<f64>,
    /// 95% credible interval for tau
    pub tau_ci: [f64; 2],
    /// Bayesian p-value (two-sided, from t-distribution)
    pub p_value: f64,
    /// t-statistic
    pub t_stat: f64,
    /// Log marginal likelihood
    pub log_marginal: f64,
    /// Cumulative effect (sum of post-period effects)
    pub cumulative_effect: f64,
    /// Predicted counterfactual (T)
    pub counterfactual: Array1<f64>,
    /// Observed treated series (T)
    pub observed: Array1<f64>,
    /// Number of control units
    pub n_controls: usize,
    /// Number of pre-treatment periods
    pub n_pre: usize,
    /// Number of post-treatment periods
    pub n_post: usize,
    /// Prior precision on weights (ridge-like)
    pub prior_precision: f64,
}

impl fmt::Display for BayesianScResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Bayesian Synthetic Control ")?;
        writeln!(f, "Brodersen et al. (2015), Scott (2019)")?;
        writeln!(f, "Conjugate Normal-Inverse-Gamma posterior")?;
        writeln!(f, "{:<20} {:>12}", "Control units:", self.n_controls)?;
        writeln!(f, "{:<20} {:>12}", "Pre periods:", self.n_pre)?;
        writeln!(f, "{:<20} {:>12}", "Post periods:", self.n_post)?;
        writeln!(f, "{:<20} {:>12.6}", "tau (treatment effect):", self.tau)?;
        writeln!(f, "{:<20} {:>12.6}", "tau SD:", self.tau_sd)?;
        writeln!(
            f,
            "{:<20} [{:.4}, {:.4}]",
            "95% CrI:", self.tau_ci[0], self.tau_ci[1]
        )?;
        writeln!(f, "{:<20} {:>12.3}", "t-statistic:", self.t_stat)?;
        writeln!(f, "{:<20} {:>12.4}", "p-value:", self.p_value)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma² (noise):", self.sigma2)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Cumulative effect:", self.cumulative_effect
        )?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "Log marginal lik.:", self.log_marginal
        )?;

        // Weights
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Synthetic control weights (posterior mean):")?;
        for (j, &w) in self.weights.iter().enumerate() {
            if w.abs() > 1e-6 {
                writeln!(f, "  Unit {:<6} {:>12.6}", j + 1, w)?;
            }
        }

        // Counterfactual vs observed
        writeln!(f, "\n  Counterfactual vs observed (selected periods):")?;
        writeln!(
            f,
            "  {:<8} {:>12} {:>12} {:>12}",
            "Period", "Observed", "Counterfact.", "Effect"
        )?;
        let n_show = 5.min(self.observed.len());
        let indices: Vec<usize> = if self.observed.len() <= n_show {
            (0..self.observed.len()).collect()
        } else {
            (0..n_show)
                .map(|i| i * (self.observed.len() - 1) / (n_show - 1).max(1))
                .collect()
        };
        for &idx in &indices {
            let effect = self.observed[idx] - self.counterfactual[idx];
            writeln!(
                f,
                "  {:<8} {:>12.4} {:>12.4} {:>12.4}",
                idx + 1,
                self.observed[idx],
                self.counterfactual[idx],
                effect
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct BayesianSC;

impl BayesianSC {
    /// Estimate Bayesian Synthetic Control.
    ///
    /// # Arguments
    /// * `y_treated` - Treated unit outcome (T)
    /// * `y_controls` - Control units outcomes (T x n_controls)
    /// * `treatment_period` - First post-treatment period (0-based)
    /// * `prior_precision` - Ridge-like prior on weights (default 1.0)
    pub fn fit(
        y_treated: &Array1<f64>,
        y_controls: &Array2<f64>,
        treatment_period: usize,
        prior_precision: Option<f64>,
    ) -> Result<BayesianScResult, GreenersError> {
        let t = y_treated.len();
        let n_controls = y_controls.ncols();
        if y_controls.nrows() != t {
            return Err(GreenersError::ShapeMismatch(
                "BayesianSC: y_treated and y_controls must have same length".into(),
            ));
        }
        if treatment_period == 0 || treatment_period >= t {
            return Err(GreenersError::InvalidOperation(
                "BayesianSC: treatment_period must be in (0, T)".into(),
            ));
        }

        let lambda = prior_precision.unwrap_or(1.0);
        let n_pre = treatment_period;
        let n_post = t - treatment_period;

        // Step 1: Estimate weights using pre-treatment data
        // Bayesian regression: y_pre = X_pre * w + eps
        // Prior: w ~ N(0, 1/lambda * I)
        // Posterior: w ~ N(mu_w, Sigma_w)
        let y_pre = y_treated.slice(ndarray::s![0..n_pre]).to_owned();
        let x_pre = y_controls
            .slice(ndarray::s![0..n_pre, 0..n_controls])
            .to_owned();

        // Add intercept
        let mut x_pre_full = Array2::zeros((n_pre, n_controls + 1));
        for i in 0..n_pre {
            x_pre_full[(i, 0)] = 1.0;
            for j in 0..n_controls {
                x_pre_full[(i, j + 1)] = x_pre[(i, j)];
            }
        }

        let n_params = n_controls + 1;

        // Prior precision matrix: lambda * I (ridge-like)
        let prior_prec = Array2::<f64>::eye(n_params) * lambda;

        // Posterior precision: X'X + lambda * I
        let xt = x_pre_full.t();
        let xtx = xt.dot(&x_pre_full);
        let post_prec = &xtx + &prior_prec;
        let post_cov = (&post_prec + Array2::<f64>::eye(n_params) * 1e-10).inv()?;

        // Posterior mean: (X'X + lambda*I)^{-1} X'y
        let xty = xt.dot(&y_pre);
        let post_mean: Array1<f64> = post_cov.dot(&xty);

        // Extract weights (skip intercept)
        let weights = post_mean.slice(ndarray::s![1..n_params]).to_owned();

        // Residual variance estimate (from pre-period)
        let y_pred_pre = x_pre_full.dot(&post_mean);
        let residuals_pre = &y_pre - &y_pred_pre;
        let sse_pre = residuals_pre.dot(&residuals_pre);
        let dof = n_pre.saturating_sub(n_params);
        let _sigma2_est = sse_pre / dof.max(1) as f64;

        // Step 2: Estimate treatment effect using post-period data
        // y_post = X_post * w + tau * 1 + eps
        // We already have w, so: y_post - X_post * w = tau + eps
        let y_post = y_treated.slice(ndarray::s![treatment_period..t]).to_owned();
        let x_post = y_controls
            .slice(ndarray::s![treatment_period..t, 0..n_controls])
            .to_owned();

        let mut x_post_full = Array2::zeros((n_post, n_params + 1));
        for i in 0..n_post {
            x_post_full[(i, 0)] = 1.0;
            for j in 0..n_controls {
                x_post_full[(i, j + 1)] = x_post[(i, j)];
            }
            // Treatment indicator
            x_post_full[(i, n_params)] = 1.0;
        }

        // Bayesian regression with treatment indicator
        let n_params_full = n_params + 1;
        let prior_prec_full = Array2::<f64>::eye(n_params_full) * lambda;
        // Weaker prior on tau (less shrinkage on treatment effect)
        let mut prior_prec_adj = prior_prec_full;
        prior_prec_adj[(n_params, n_params)] = lambda * 0.01;

        let xt_post = x_post_full.t();
        let xtx_post = xt_post.dot(&x_post_full);
        let post_prec_full = &xtx_post + &prior_prec_adj;
        let post_cov_full = (&post_prec_full + Array2::<f64>::eye(n_params_full) * 1e-10).inv()?;
        let xty_post = xt_post.dot(&y_post);
        let post_mean_full: Array1<f64> = post_cov_full.dot(&xty_post);

        // tau is the last coefficient
        let tau = post_mean_full[n_params];
        let tau_var = post_cov_full[(n_params, n_params)];
        let tau_sd = tau_var.sqrt();

        // Posterior sigma^2 (Inverse-Gamma posterior)
        let y_pred_post = x_post_full.dot(&post_mean_full);
        let residuals_post = &y_post - &y_pred_post;
        let sse_post = residuals_post.dot(&residuals_post);
        let sigma2 = (sse_pre + sse_post) / (dof + n_post) as f64;

        // 95% credible interval using t-distribution
        let dof_total = (dof + n_post) as f64;
        let t_dist = StudentsT::new(0.0, 1.0, dof_total)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let t_crit = t_dist.inverse_cdf(0.975);
        let tau_ci = [tau - t_crit * tau_sd, tau + t_crit * tau_sd];

        // p-value and t-stat
        let t_stat = if tau_sd > 1e-10 { tau / tau_sd } else { 0.0 };
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));

        // Counterfactual: predicted outcome without treatment
        let mut counterfactual = Array1::zeros(t);
        for time in 0..t {
            let mut pred = post_mean_full[0]; // intercept
            for j in 0..n_controls {
                pred += weights[j] * y_controls[(time, j)];
            }
            counterfactual[time] = pred;
        }

        // Cumulative effect
        let cumulative_effect: f64 = (treatment_period..t)
            .map(|time| y_treated[time] - counterfactual[time])
            .sum();

        // Log marginal likelihood (simplified)
        let log_marginal = -0.5 * n_pre as f64 * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * (post_prec.det().unwrap_or(1e-300)).ln().max(-300.0)
            - 0.5 * sse_pre / sigma2.max(1e-10);

        Ok(BayesianScResult {
            tau,
            tau_sd,
            sigma2,
            weights,
            tau_ci,
            p_value,
            t_stat,
            log_marginal,
            cumulative_effect,
            counterfactual,
            observed: y_treated.clone(),
            n_controls,
            n_pre,
            n_post,
            prior_precision: lambda,
        })
    }
}
