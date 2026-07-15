//! Bayesian Linear Regression with conjugate Normal-Inverse-Gamma prior.
//!
//! Model:
//!   y = X * beta + epsilon,  epsilon ~ N(0, sigma^2)
//!
//! Prior (conjugate):
//!   beta | sigma^2 ~ N(beta_0, sigma^2 * V_0)
//!   sigma^2 ~ Inverse-Gamma(a_0, b_0)
//!
//! Posterior (closed-form):
//!   V_n = (V_0^{-1} + X'X)^{-1}
//!   beta_n = V_n * (V_0^{-1} * beta_0 + X'y)
//!   a_n = a_0 + n/2
//!   b_n = b_0 + 0.5 * (y'y + beta_0' * V_0^{-1} * beta_0 - beta_n' * V_n^{-1} * beta_n)
//!
//! Default prior: weak (non-informative): beta_0 = 0, V_0 = 1000*I,
//! a_0 = 0.001, b_0 = 0.001.

use crate::linalg::{LinalgDeterminant as _, LinalgInverse as _};
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, InverseGamma, Normal, StudentsT};
use std::fmt;

/// Result of Bayesian linear regression.
#[derive(Debug)]
pub struct BayesianLinearResult {
    /// Posterior mean of coefficients
    pub beta: Array1<f64>,
    /// Posterior covariance of coefficients
    pub beta_cov: Array2<f64>,
    /// Posterior mean of sigma^2
    pub sigma2: f64,
    /// Posterior shape (a_n)
    pub sigma2_shape: f64,
    /// Posterior scale (b_n)
    pub sigma2_scale: f64,
    /// Prior mean of coefficients
    pub beta_prior: Array1<f64>,
    /// Prior covariance of coefficients
    pub v_prior: Array2<f64>,
    /// Prior shape (a_0)
    pub a_prior: f64,
    /// Prior scale (b_0)
    pub b_prior: f64,
    /// 95% credible intervals for coefficients
    pub beta_ci: Array2<f64>,
    /// Posterior probability that each coefficient > 0
    pub p_positive: Array1<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of predictors (including intercept)
    pub n_pred: usize,
    /// Marginal likelihood
    pub log_marginal: f64,
    /// In-sample R-squared
    pub r_squared: f64,
    /// Fitted values
    pub fitted: Array1<f64>,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for BayesianLinearResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Bayesian Linear Regression ")?;
        writeln!(f, "Conjugate Normal-Inverse-Gamma prior")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Predictors:", self.n_pred)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma² (posterior):", self.sigma2)?;
        writeln!(f, "{:<20} {:>12.6}", "R²:", self.r_squared)?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "Log marginal lik.:", self.log_marginal
        )?;

        // Coefficients
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "  {:<14} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Post. mean", "SD", "2.5%", "97.5%"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for (j, name) in self.variable_names.iter().enumerate() {
            let sd = self.beta_cov[(j, j)].sqrt();
            writeln!(
                f,
                "  {:<14} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                name,
                self.beta[j],
                sd,
                self.beta_ci[(j, 0)],
                self.beta_ci[(j, 1)]
            )?;
        }

        // Posterior probabilities
        writeln!(f, "\n  P(coef > 0):")?;
        for (j, name) in self.variable_names.iter().enumerate() {
            writeln!(f, "  {:<14} {:>10.4}", name, self.p_positive[j])?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct BayesianLinear;

impl BayesianLinear {
    /// Estimate Bayesian linear regression with conjugate prior.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Design matrix (n x k), WITHOUT intercept (added internally)
    /// * `variable_names` - Optional variable names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<BayesianLinearResult, GreenersError> {
        Self::fit_with_prior(y, x, None, None, None, None, variable_names)
    }

    /// Estimate with custom prior parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_with_prior(
        y: &Array1<f64>,
        x: &Array2<f64>,
        beta_prior: Option<&Array1<f64>>,
        v_prior: Option<&Array2<f64>>,
        a_prior: Option<f64>,
        b_prior: Option<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<BayesianLinearResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "BayesianLinear: y and x must have same n".into(),
            ));
        }
        if n < k + 2 {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: need more observations than predictors".into(),
            ));
        }

        // Add intercept column
        let p = k + 1; // intercept + k predictors
        let mut x_full = Array2::zeros((n, p));
        for i in 0..n {
            x_full[(i, 0)] = 1.0;
            for j in 0..k {
                x_full[(i, j + 1)] = x[(i, j)];
            }
        }

        let mut names =
            variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        names.insert(0, "Intercept".to_string());

        // Default prior: weak/non-informative
        let beta_0 = beta_prior.cloned().unwrap_or_else(|| Array1::zeros(p));
        let v_0 = v_prior
            .cloned()
            .unwrap_or_else(|| Array2::<f64>::eye(p) * 1000.0);
        let a_0 = a_prior.unwrap_or(0.001);
        let b_0 = b_prior.unwrap_or(0.001);

        if beta_0.len() != p || v_0.nrows() != p || v_0.ncols() != p {
            return Err(GreenersError::ShapeMismatch(
                "BayesianLinear: prior dimensions don't match".into(),
            ));
        }

        // Posterior computation
        let xt = x_full.t();
        let xtx = xt.dot(&x_full);
        let xty = xt.dot(y);
        let yty = y.dot(y);

        let v_0_inv = v_0.inv()?;
        let v_n_inv = &v_0_inv + &xtx;
        let v_n = v_n_inv.inv()?;

        // beta_n = V_n * (V_0^{-1} * beta_0 + X'y)
        let v0_inv_beta0 = v_0_inv.dot(&beta_0);
        let rhs = &v0_inv_beta0 + &xty;
        let beta_n = v_n.dot(&rhs);

        // a_n = a_0 + n/2
        let a_n = a_0 + n as f64 / 2.0;

        // b_n = b_0 + 0.5 * (y'y + beta_0' * V_0^{-1} * beta_0 - beta_n' * V_n^{-1} * beta_n)
        let beta0_v0inv_beta0 = beta_0.dot(&v0_inv_beta0);
        let betan_vninv_betan = beta_n.dot(&v_n_inv.dot(&beta_n));
        let b_n = b_0 + 0.5 * (yty + beta0_v0inv_beta0 - betan_vninv_betan);

        // Posterior mean of sigma^2 = b_n / (a_n - 1)
        let sigma2_post = if a_n > 1.0 {
            b_n / (a_n - 1.0)
        } else {
            b_n / a_n
        };

        // Credible intervals for beta
        // Marginal posterior: beta_j ~ t_{2*a_n}(beta_n_j, V_n_jj * b_n / a_n)
        let df = 2.0 * a_n;
        let t_dist = StudentsT::new(0.0, 1.0, df)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let t_975 = t_dist.inverse_cdf(0.975);

        let mut beta_ci = Array2::zeros((p, 2));
        let mut p_positive = Array1::zeros(p);
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        for j in 0..p {
            let sd_j = (v_n[(j, j)] * b_n / a_n).sqrt();
            beta_ci[(j, 0)] = beta_n[j] - t_975 * sd_j;
            beta_ci[(j, 1)] = beta_n[j] + t_975 * sd_j;

            // P(beta_j > 0) using t-distribution
            let t_stat = if sd_j > 1e-10 { beta_n[j] / sd_j } else { 0.0 };
            p_positive[j] = 1.0 - t_dist.cdf(t_stat);
        }

        // Fitted values
        let fitted = x_full.dot(&beta_n);

        // R-squared
        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let sse = y
            .iter()
            .zip(fitted.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        // Log marginal likelihood (approximation)
        let log_marginal = Self::log_marginal_likelihood(
            &xtx, &xty, yty, n, p, &v_0, &v_0_inv, &beta_0, a_0, b_0,
        )?;

        // Suppress unused warning for normal
        let _ = normal;

        Ok(BayesianLinearResult {
            beta: beta_n,
            beta_cov: v_n,
            sigma2: sigma2_post,
            sigma2_shape: a_n,
            sigma2_scale: b_n,
            beta_prior: beta_0,
            v_prior: v_0,
            a_prior: a_0,
            b_prior: b_0,
            beta_ci,
            p_positive,
            n_obs: n,
            n_pred: p,
            log_marginal,
            r_squared,
            fitted,
            variable_names: names,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn log_marginal_likelihood(
        xtx: &Array2<f64>,
        xty: &Array1<f64>,
        yty: f64,
        n: usize,
        _p: usize,
        v_0: &Array2<f64>,
        v_0_inv: &Array2<f64>,
        beta_0: &Array1<f64>,
        a_0: f64,
        b_0: f64,
    ) -> Result<f64, GreenersError> {
        let v_n_inv = v_0_inv + xtx;
        let v_n = v_n_inv.inv()?;
        let v0_inv_beta0 = v_0_inv.dot(beta_0);
        let rhs = &v0_inv_beta0 + xty;
        let beta_n = v_n.dot(&rhs);

        let beta0_v0inv_beta0 = beta_0.dot(&v0_inv_beta0);
        let betan_vninv_betan = beta_n.dot(&v_n_inv.dot(&beta_n));

        let a_n = a_0 + n as f64 / 2.0;
        let b_n = b_0 + 0.5 * (yty + beta0_v0inv_beta0 - betan_vninv_betan);

        // Log marginal = log Gamma(a_n) - log Gamma(a_0)
        //   - (n/2) * log(2*pi) + 0.5 * log(|V_0|/|V_n|)
        //   + a_0 * log(b_0) - a_n * log(b_n)

        let log_gamma = |x: f64| {
            // Stirling's approximation for log Gamma
            if x > 1.0 {
                (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
            } else {
                0.0
            }
        };

        let det_v0 = v_0.det().unwrap_or(1e-300).ln().max(-300.0);
        let det_vn = v_n.det().unwrap_or(1e-300).ln().max(-300.0);

        let lml =
            log_gamma(a_n) - log_gamma(a_0) - (n as f64 / 2.0) * (2.0 * std::f64::consts::PI).ln()
                + 0.5 * (det_v0 - det_vn)
                + a_0 * b_0.ln().max(-300.0)
                - a_n * b_n.ln().max(-300.0);

        // Suppress unused InverseGamma
        let _ = InverseGamma::new(a_n, b_n);

        Ok(lml)
    }
}
