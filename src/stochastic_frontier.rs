//! Stochastic Frontier Analysis (SFA).
//!
//! Production frontier: y = α + β'x + v - u
//! where v ~ N(0, σ_v²) is statistical noise and u ~ N⁺(0, σ_u²)
//! is the one-sided inefficiency term (half-normal).
//!
//! Cost frontier: y = α + β'x + v + u (u ≥ 0 increases cost).
//!
//! MLE via grid search over λ = σ_u/σ_v, then closed-form for β.
//! Technical efficiency: TE_i = E[u_i | ε_i] (Jondrow et al. 1982).

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::f64::consts;
use std::fmt;

/// Result of stochastic frontier estimation.
#[derive(Debug)]
pub struct SfaResult {
    /// Model type: "production" or "cost"
    pub model_type: String,
    /// Coefficients (intercept + beta)
    pub beta: Array1<f64>,
    /// Standard errors
    pub std_errors: Array1<f64>,
    /// t-statistics
    pub t_values: Array1<f64>,
    /// p-values
    pub p_values: Array1<f64>,
    /// sigma_v (noise std dev)
    pub sigma_v: f64,
    /// sigma_u (inefficiency std dev)
    pub sigma_u: f64,
    /// sigma = sqrt(sigma_v² + sigma_u²)
    pub sigma: f64,
    /// lambda = sigma_u / sigma_v
    pub lambda: f64,
    /// gamma = sigma_u² / sigma² (variance share of inefficiency)
    pub gamma: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Technical efficiency per observation (TE = exp(-u_i) for production)
    pub efficiency: Array1<f64>,
    /// Mean efficiency
    pub mean_efficiency: f64,
    /// Variable names
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for SfaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title = if self.model_type == "production" {
            " Stochastic Production Frontier "
        } else {
            " Stochastic Cost Frontier "
        };
        writeln!(f, "\n{:=^78}", title)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma_v (noise):", self.sigma_v)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma_u (ineffic.):", self.sigma_u)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma:", self.sigma)?;
        writeln!(f, "{:<20} {:>12.6}", "lambda:", self.lambda)?;
        writeln!(f, "{:<20} {:>12.6}", "gamma:", self.gamma)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Mean efficiency:", self.mean_efficiency
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

pub struct StochasticFrontier;

impl StochasticFrontier {
    /// Estimate stochastic production frontier: y = α + β'x + v - u
    ///
    /// # Arguments
    /// * `y` - Output (log scale recommended)
    /// * `x` - Inputs matrix (n × k, includes intercept)
    /// * `variable_names` - Optional names
    pub fn fit_production(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<SfaResult, GreenersError> {
        Self::fit(y, x, variable_names, "production")
    }

    /// Estimate stochastic cost frontier: y = α + β'x + v + u
    pub fn fit_cost(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<SfaResult, GreenersError> {
        Self::fit(y, x, variable_names, "cost")
    }

    fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
        model_type: &str,
    ) -> Result<SfaResult, GreenersError> {
        let n = y.len();
        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "SFA: y and x must have same number of rows".into(),
            ));
        }

        // OLS for initial beta
        let xt = x.t();
        let xtx = xt.dot(x);
        let xtx_inv = xtx.inv()?;
        let xty = xt.dot(y);
        let beta_ols: Array1<f64> = xtx_inv.dot(&xty);
        let residuals = y - x.dot(&beta_ols);

        // For production: residual = v - u (left-skewed)
        // For cost: residual = v + u (right-skewed)
        // We search over lambda = sigma_u / sigma_v

        let mut best_lambda = 0.5_f64;
        let mut best_ll = f64::NEG_INFINITY;

        // Grid search over lambda in [0.01, 10]
        let n_grid = 200;
        for i in 0..n_grid {
            let lam = 0.01 + 9.99 * i as f64 / (n_grid - 1) as f64;
            let ll = Self::log_likelihood(&residuals, lam, model_type);
            if ll > best_ll {
                best_ll = ll;
                best_lambda = lam;
            }
        }

        // Golden section refinement
        let golden = 0.6180339887498949;
        let mut a = (best_lambda - 0.1).max(0.01);
        let mut b = best_lambda + 0.1;
        let mut c = b - golden * (b - a);
        let mut d = a + golden * (b - a);
        let mut fc = Self::log_likelihood(&residuals, c, model_type);
        let mut fd = Self::log_likelihood(&residuals, d, model_type);
        for _ in 0..60 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - golden * (b - a);
                fc = Self::log_likelihood(&residuals, c, model_type);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + golden * (b - a);
                fd = Self::log_likelihood(&residuals, d, model_type);
            }
        }
        best_lambda = if fc > fd { c } else { d };
        best_ll = if fc > fd { fc } else { fd };

        // Compute sigma_v and sigma_u from lambda and residuals
        // sigma² = var(residuals) (from OLS)
        let mean_res = residuals.mean().unwrap_or(0.0);
        let var_res = residuals.mapv(|r| (r - mean_res).powi(2)).sum() / n as f64;
        let sigma2 = var_res;
        let lambda2 = best_lambda * best_lambda;
        let sigma_v2 = sigma2 / (1.0 + lambda2);
        let sigma_u2 = sigma2 * lambda2 / (1.0 + lambda2);
        let sigma_v = sigma_v2.sqrt();
        let sigma_u = sigma_u2.sqrt();
        let sigma = sigma2.sqrt();
        let gamma = sigma_u2 / sigma2;

        // Re-estimate beta with MLE correction
        // For SFA, the MLE beta is the OLS beta (the skewness only affects
        // the intercept in the composite error). We adjust the intercept.
        let mut beta = beta_ols.clone();
        // Mean of u (half-normal): E[u] = sigma_u * sqrt(2/pi)
        let eu = sigma_u * (2.0 / consts::PI).sqrt();
        if model_type == "production" {
            // y = alpha + beta'x + v - u => E[y] = alpha - E[u] + beta'x
            // OLS intercept estimates alpha - E[u], so alpha = intercept + E[u]
            beta[0] += eu;
        } else {
            // y = alpha + beta'x + v + u => E[y] = alpha + E[u] + beta'x
            // OLS intercept estimates alpha + E[u], so alpha = intercept - E[u]
            beta[0] -= eu;
        }

        // Standard errors (from OLS, adjusted)
        let sigma_ols2 = var_res;
        let cov_beta = xtx_inv * sigma_ols2;
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

        // Technical efficiency: TE_i = exp(-E[u_i | eps_i])
        // Jondrow et al. (1982): E[u | eps] = sigma_u * sigma_v / sigma * [phi(eps*lam/sigma) / Phi(eps*lam/sigma) + eps*lam/sigma]
        // For production: eps = residual (v - u), sign is negative
        // For cost: eps = residual (v + u), sign is positive
        let mut efficiency = Array1::zeros(n);
        let normal_dist =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        for i in 0..n {
            let eps = if model_type == "production" {
                -residuals[i] // negate because u is subtracted
            } else {
                residuals[i]
            };
            let mu_star = eps * best_lambda / (1.0 + lambda2);
            let sigma_star = sigma_v * sigma_u / sigma;
            let ratio = mu_star / sigma_star.max(1e-10);
            // E[u | eps] = sigma_star * [phi(ratio)/Phi(ratio) + ratio]
            let phi_ratio = normal_dist.pdf(ratio);
            let cdf_ratio = normal_dist.cdf(ratio).max(1e-300);
            let e_u = sigma_star * (phi_ratio / cdf_ratio + ratio);
            efficiency[i] = (-e_u).exp();
        }

        let mean_efficiency = efficiency.mean().unwrap_or(0.0);

        Ok(SfaResult {
            model_type: model_type.to_string(),
            beta,
            std_errors,
            t_values,
            p_values,
            sigma_v,
            sigma_u,
            sigma,
            lambda: best_lambda,
            gamma,
            log_likelihood: best_ll,
            n_obs: n,
            efficiency,
            mean_efficiency,
            variable_names,
        })
    }

    /// Log-likelihood for the half-normal model.
    /// For production: eps = v - u (composite error, left-skewed)
    /// For cost: eps = v + u (composite error, right-skewed)
    /// We use the standard formulation for production (eps < 0 skew):
    /// ln L = -n/2 * ln(2π) - n/2 * ln(σ²) + Σ ln[Φ(-ε_i λ/σ)] + Σ[-ε_i²/(2σ²) * (1/(1+λ²)) ...]
    /// Simplified: ln f(ε) = -ln(σ) - 0.5*ln(2π) - ε²/(2σ²) + ln[Φ(-ελ/σ)]
    /// where σ² = σ_v²(1+λ²)
    fn log_likelihood(residuals: &Array1<f64>, lambda: f64, model_type: &str) -> f64 {
        let n = residuals.len();
        let normal = match Normal::new(0.0, 1.0) {
            Ok(d) => d,
            Err(_) => return f64::NEG_INFINITY,
        };

        let mean_res = residuals.mean().unwrap_or(0.0);
        let var_res = residuals.mapv(|r| (r - mean_res).powi(2)).sum() / n as f64;
        let sigma2 = var_res * (1.0 + lambda * lambda);
        let sigma = sigma2.sqrt();

        let mut ll = 0.0;
        for i in 0..n {
            let eps = if model_type == "production" {
                residuals[i]
            } else {
                -residuals[i]
            };
            // ln f(eps) = -ln(sigma) - 0.5*ln(2*pi) - eps^2/(2*sigma^2) + ln(Phi(-eps*lambda/sigma))
            let z = -eps * lambda / sigma;
            let cdf_val = normal.cdf(z).max(1e-300);
            let ln_f = -sigma.ln() - 0.5 * (2.0 * consts::PI).ln() - eps * eps / (2.0 * sigma2)
                + cdf_val.ln();
            ll += ln_f;
        }
        ll
    }
}
