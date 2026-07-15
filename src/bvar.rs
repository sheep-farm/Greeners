//! Bayesian VAR (BVAR) with Minnesota prior.
//!
//! Litterman (1986), Doan, Litterman & Sims (1984). Combines OLS
//! likelihood with a Minnesota (Litterman) prior to regularize VAR
//! coefficients. The Minnesota prior centers each equation around
//! a random walk (coefficient = 1 on own lag 1, 0 otherwise) and
//! shrinks lags more aggressively.
//!
//! Prior hyperparameters:
//!   - lambda_1: tightness on own lag (default 0.1)
//!   - lambda_2: tightness on other lags (default 0.2)
//!   - lambda_3: tightness on higher lags (default 1.0)
//!   - mu: prior mean on first own lag (default 1.0 for random walk)
//!
//! Implementation: conjugate Normal prior + Normal likelihood =>
//! posterior is Normal. Closed-form posterior mean and covariance.

use crate::linalg::{LinalgDeterminant as _, LinalgInverse as _};
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of BVAR estimation.
#[derive(Debug)]
pub struct BvarResult {
    /// Posterior mean coefficients (k x (k*p)), each row = equation
    pub coeffs: Array2<f64>,
    /// Posterior standard errors
    pub std_errors: Array2<f64>,
    /// t-values (posterior mean / SE)
    pub t_values: Array2<f64>,
    /// p-values (two-sided)
    pub p_values: Array2<f64>,
    /// Prior hyperparameters: [lambda1, lambda2, lambda3, mu]
    pub hyperparams: [f64; 4],
    /// Posterior residual covariance (k x k)
    pub resid_cov: Array2<f64>,
    /// Log marginal likelihood
    pub log_marginal: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of variables
    pub n_vars: usize,
    /// VAR lag order
    pub lags: usize,
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for BvarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Bayesian VAR (Minnesota prior) ")?;
        writeln!(f, "Litterman (1986), Doan-Litterman-Sims (1984)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "lambda_1 (own lag):", self.hyperparams[0]
        )?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "lambda_2 (cross lag):", self.hyperparams[1]
        )?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "lambda_3 (higher lag):", self.hyperparams[2]
        )?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "mu (RW prior mean):", self.hyperparams[3]
        )?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "Log marginal lik.:", self.log_marginal
        )?;

        let k = self.n_vars;
        let p = self.lags;
        for eq in 0..k {
            let eq_name = self
                .var_names
                .get(eq)
                .cloned()
                .unwrap_or_else(|| format!("y{}", eq));
            writeln!(f, "\n{:-^78}", format!(" Equation: {} ", eq_name))?;
            writeln!(
                f,
                "{:<14} {:>12} {:>12} {:>10} {:>10}",
                "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
            )?;
            writeln!(f, "{:-^78}", "")?;
            for lag in 0..p {
                for j in 0..k {
                    let var_name = self
                        .var_names
                        .get(j)
                        .cloned()
                        .unwrap_or_else(|| format!("y{}", j));
                    let col = lag * k + j;
                    let label = format!("L{}.{}", lag + 1, var_name);
                    writeln!(
                        f,
                        "{:<14} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                        label,
                        self.coeffs[(eq, col)],
                        self.std_errors[(eq, col)],
                        self.t_values[(eq, col)],
                        self.p_values[(eq, col)]
                    )?;
                }
            }
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct BVAR;

impl BVAR {
    /// Estimate BVAR with Minnesota prior.
    ///
    /// # Arguments
    /// * `y` - Data matrix (T x k)
    /// * `lags` - VAR lag order p
    /// * `lambda1` - Tightness on own lag (default 0.1)
    /// * `lambda2` - Tightness on cross-variable lags (default 0.2)
    /// * `lambda3` - Decay for higher lags (default 1.0)
    /// * `var_names` - Optional variable names
    pub fn fit(
        y: &Array2<f64>,
        lags: usize,
        lambda1: Option<f64>,
        lambda2: Option<f64>,
        lambda3: Option<f64>,
        var_names: Option<Vec<String>>,
    ) -> Result<BvarResult, GreenersError> {
        let t = y.nrows();
        let k = y.ncols();
        if lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "BVAR: lags must be >= 1".into(),
            ));
        }
        if t < lags + k + 1 {
            return Err(GreenersError::InvalidOperation(
                "BVAR: too few observations".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("y{}", i)).collect());
        let l1 = lambda1.unwrap_or(0.1);
        let l2 = lambda2.unwrap_or(0.2);
        let l3 = lambda3.unwrap_or(1.0);
        let mu = 1.0; // random walk prior

        // Build VAR design matrix
        let n_eff = t - lags;
        let n_reg = k * lags;
        let mut x = Array2::zeros((n_eff, n_reg));
        let mut y_dep = Array2::zeros((n_eff, k));

        for i in 0..n_eff {
            let t_i = lags + i;
            for j in 0..k {
                y_dep[(i, j)] = y[(t_i, j)];
            }
            for lag in 0..lags {
                for j in 0..k {
                    x[(i, lag * k + j)] = y[(t_i - 1 - lag, j)];
                }
            }
        }

        // OLS estimates
        let xt = x.t();
        let xtx = xt.dot(&x);
        let xty = xt.dot(&y_dep);
        let xtx_inv = (&xtx + Array2::<f64>::eye(n_reg) * 1e-10).inv()?;
        let ols_beta = xtx_inv.dot(&xty); // (n_reg x k)

        // Residual covariance (OLS)
        let residuals = &y_dep - x.dot(&ols_beta);
        let sigma2_ols: Array1<f64> = (0..k)
            .map(|j| {
                let col = residuals.column(j);
                col.iter().map(|r| r * r).sum::<f64>() / (n_eff - n_reg) as f64
            })
            .collect();

        // Minnesota prior: mean and variance for each coefficient
        // Prior mean: 1.0 on own lag 1, 0 elsewhere
        // (computed per-equation below as b0 and v0)

        // Posterior: combine prior + likelihood
        // For each equation j, the prior mean has mu on the own-lag-1 coefficient
        // Posterior: (X'X + V0^{-1})^{-1} (X'y + V0^{-1} * b0)
        let mut post_coeffs = Array2::zeros((k, n_reg));
        let mut post_se = Array2::zeros((k, n_reg));
        let mut post_t = Array2::zeros((k, n_reg));
        let mut post_p = Array2::zeros((k, n_reg));

        for eq in 0..k {
            // Set prior mean: mu on own lag 1 for this equation
            let mut b0 = Array1::zeros(n_reg);
            b0[eq] = mu; // own first lag

            // Adjust prior variance for this equation (own vs cross)
            let mut v0 = Array2::zeros((n_reg, n_reg));
            for lag in 0..lags {
                for j in 0..k {
                    let idx = lag * k + j;
                    let lag_decay = (lag as f64 + 1.0).powf(-l3);
                    if j == eq {
                        // Own lag
                        v0[(idx, idx)] = l1.powi(2) * lag_decay * sigma2_ols[eq].max(1e-10);
                    } else {
                        // Cross lag
                        let sigma_ratio = sigma2_ols[j].max(1e-10) / sigma2_ols[eq].max(1e-10);
                        v0[(idx, idx)] =
                            l2.powi(2) * lag_decay * sigma_ratio * sigma2_ols[eq].max(1e-10);
                    }
                }
            }

            // Prior precision
            let v0_inv = {
                let mut inv = Array2::zeros((n_reg, n_reg));
                for i in 0..n_reg {
                    inv[(i, i)] = 1.0 / v0[(i, i)].max(1e-10);
                }
                inv
            };

            // Posterior precision: X'X + V0^{-1}
            let post_prec = &xtx + &v0_inv;
            let post_cov = (&post_prec + Array2::<f64>::eye(n_reg) * 1e-10).inv()?;

            // Posterior mean: (X'X + V0^{-1})^{-1} (X'y + V0^{-1} * b0)
            let xty_eq = xty.column(eq).to_owned();
            let prior_term = v0_inv.dot(&b0);
            let post_rhs = &xty_eq + &prior_term;
            let post_mean: Array1<f64> = post_cov.dot(&post_rhs);

            // SE
            let se = post_cov.diag().mapv(|v| v.sqrt());
            let tv = &post_mean / &se;
            let normal = Normal::new(0.0, 1.0)
                .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
            let pv = tv.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

            for col in 0..n_reg {
                post_coeffs[(eq, col)] = post_mean[col];
                post_se[(eq, col)] = se[col];
                post_t[(eq, col)] = tv[col];
                post_p[(eq, col)] = pv[col];
            }
        }

        // Posterior residual covariance
        let post_resid = &y_dep - x.dot(&post_coeffs.t());
        let resid_cov = post_resid.t().dot(&post_resid) / n_eff as f64;

        // Log marginal likelihood (Laplace approximation, simplified)
        let log_marginal = -0.5 * n_eff as f64 * k as f64 * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * resid_cov.det().unwrap_or(1e-300).ln().max(-300.0)
            - 0.5 * post_resid.iter().map(|r| r * r).sum::<f64>();

        Ok(BvarResult {
            coeffs: post_coeffs,
            std_errors: post_se,
            t_values: post_t,
            p_values: post_p,
            hyperparams: [l1, l2, l3, mu],
            resid_cov,
            log_marginal,
            n_obs: n_eff,
            n_vars: k,
            lags,
            var_names: names,
        })
    }
}
