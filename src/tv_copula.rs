//! Time-varying copula with GARCH dynamics.
//!
//! Patton (2006), Creal, Koopman & Lucas (2013). Extends static
//! copula to allow the dependence parameter to evolve over time
//! via a GARCH-like or ARMA-like process.
//!
//! For Gaussian copula with time-varying correlation:
//!   rho_t = Lambda(a + b * rho_{t-1} + c * psi_{t-1})
//! where Lambda is the tanh inverse transform to keep rho in (-1, 1),
//! and psi_{t-1} is a forcing variable (e.g., lagged cross-product
//! of standardized residuals).
//!
//! For Clayton/Gumbel, the parameter evolves similarly with an
//! appropriate transform to maintain the valid range.

use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Copula types for time-varying estimation.
#[derive(Debug, Clone, Copy)]
pub enum TvCopulaType {
    Gaussian,
    Clayton,
    Gumbel,
}

/// Result of time-varying copula estimation.
#[derive(Debug)]
pub struct TvCopulaResult {
    /// Copula type
    pub copula_type: TvCopulaType,
    /// Time-varying dependence parameter (T)
    pub theta_path: Array1<f64>,
    /// Dynamics parameters: [a (intercept), b (AR), c (forcing)]
    pub dynamics_params: Array1<f64>,
    /// Kendall's tau path (T)
    pub kendall_tau_path: Array1<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Variable names
    pub var_names: Vec<String>,
    /// Mean theta
    pub mean_theta: f64,
    /// Std of theta
    pub std_theta: f64,
}

impl fmt::Display for TvCopulaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Time-Varying Copula ")?;
        writeln!(f, "Patton (2006) — GARCH-like dynamics")?;
        let copula_str = match self.copula_type {
            TvCopulaType::Gaussian => "Gaussian",
            TvCopulaType::Clayton => "Clayton",
            TvCopulaType::Gumbel => "Gumbel",
        };
        writeln!(f, "{:<20} {}", "Copula type:", copula_str)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12.6}", "Mean theta:", self.mean_theta)?;
        writeln!(f, "{:<20} {:>12.6}", "Std theta:", self.std_theta)?;

        writeln!(f, "\n  Dynamics parameters:")?;
        writeln!(f, "  a (intercept):  {:>12.6}", self.dynamics_params[0])?;
        writeln!(f, "  b (AR):         {:>12.6}", self.dynamics_params[1])?;
        writeln!(f, "  c (forcing):    {:>12.6}", self.dynamics_params[2])?;

        writeln!(
            f,
            "\n{:<20} {:>12.4}",
            "Log-likelihood:", self.log_likelihood
        )?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        // Show theta at selected periods
        let n_show = 5.min(self.n_obs);
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Time-varying theta (selected periods):")?;
        writeln!(f, "  {:<8} {:>12} {:>12}", "Period", "theta", "Kendall tau")?;
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
                self.theta_path[idx],
                self.kendall_tau_path[idx]
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct TvCopula;

impl TvCopula {
    /// Estimate time-varying copula with GARCH-like dynamics.
    ///
    /// # Arguments
    /// * `x` - Data matrix (T x k), bivariate (k=2) recommended
    /// * `copula_type` - Type of copula
    /// * `var_names` - Optional variable names
    pub fn fit(
        x: &Array2<f64>,
        copula_type: TvCopulaType,
        var_names: Option<Vec<String>>,
    ) -> Result<TvCopulaResult, GreenersError> {
        let t = x.nrows();
        let k = x.ncols();
        if k < 2 {
            return Err(GreenersError::InvalidOperation(
                "TvCopula: need at least 2 variables".into(),
            ));
        }
        if t < 10 {
            return Err(GreenersError::InvalidOperation(
                "TvCopula: need at least 10 observations".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());

        // Step 1: Transform to uniform via empirical CDF (using first 2 variables)
        let u1 = Self::empirical_cdf(&x.column(0));
        let u2 = Self::empirical_cdf(&x.column(1));

        // Step 2: Compute forcing variable psi_t = |u1_t - 0.5| * |u2_t - 0.5| * 4
        // (a measure of joint deviation from independence)
        let psi: Array1<f64> = (0..t)
            .map(|i| (u1[i] - 0.5).abs() * (u2[i] - 0.5).abs() * 4.0)
            .collect();

        // Step 3: Grid search over dynamics parameters [a, b, c]
        // theta_t = transform(a + b * theta_{t-1} + c * psi_{t-1})
        let mut best_params = Array1::zeros(3);
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_theta_path = Array1::zeros(t);

        let n_grid = 7;
        for ai in 0..n_grid {
            for bi in 0..n_grid {
                for ci in 0..n_grid {
                    let a = -1.0 + 2.0 * ai as f64 / (n_grid - 1) as f64;
                    let b = 0.0 + 0.95 * bi as f64 / (n_grid - 1) as f64;
                    let c = -1.0 + 2.0 * ci as f64 / (n_grid - 1) as f64;

                    // Skip if b too high (instability)
                    if b > 0.98 {
                        continue;
                    }

                    let (theta_path, ll) =
                        Self::evolve_and_loglik(&u1, &u2, copula_type, a, b, c, &psi, t);

                    if ll > best_ll {
                        best_ll = ll;
                        best_params = Array1::from_vec(vec![a, b, c]);
                        best_theta_path = theta_path;
                    }
                }
            }
        }

        // Compute Kendall's tau path
        let kendall_tau_path = match copula_type {
            TvCopulaType::Gaussian => {
                best_theta_path.mapv(|rho| (2.0 / std::f64::consts::PI) * rho.asin())
            }
            TvCopulaType::Clayton => best_theta_path.mapv(|th| th / (th + 2.0)),
            TvCopulaType::Gumbel => best_theta_path.mapv(|th| 1.0 - 1.0 / th),
        };

        let n_params = 3;
        let aic = -2.0 * best_ll + 2.0 * n_params as f64;
        let bic = -2.0 * best_ll + (t as f64) * n_params as f64;

        let mean_theta = best_theta_path.mean().unwrap_or(0.0);
        let std_theta = best_theta_path.std(0.0);

        Ok(TvCopulaResult {
            copula_type,
            theta_path: best_theta_path,
            dynamics_params: best_params,
            kendall_tau_path,
            log_likelihood: best_ll,
            aic,
            bic,
            n_obs: t,
            n_vars: k,
            var_names: names,
            mean_theta,
            std_theta,
        })
    }

    /// Evolve theta over time and compute log-likelihood.
    #[allow(clippy::too_many_arguments)]
    fn evolve_and_loglik(
        u1: &Array1<f64>,
        u2: &Array1<f64>,
        copula_type: TvCopulaType,
        a: f64,
        b: f64,
        c: f64,
        psi: &Array1<f64>,
        t: usize,
    ) -> (Array1<f64>, f64) {
        let mut theta_path = Array1::zeros(t);
        let mut ll = 0.0_f64;

        // Initialize theta at unconditional level
        let theta_init = match copula_type {
            TvCopulaType::Gaussian => 0.0, // independence
            TvCopulaType::Clayton => 0.5,
            TvCopulaType::Gumbel => 1.5,
        };
        theta_path[0] = theta_init;

        for i in 1..t {
            // Raw dynamics
            let raw = a + b * theta_path[i - 1] + c * psi[i - 1];

            // Transform to valid range
            theta_path[i] = match copula_type {
                TvCopulaType::Gaussian => raw.tanh(),                // (-1, 1)
                TvCopulaType::Clayton => raw.exp().max(0.01),        // (0, inf)
                TvCopulaType::Gumbel => (1.0 + raw.exp()).max(1.01), // [1, inf)
            };

            // Log-likelihood contribution
            let u1v = u1[i].clamp(1e-10, 1.0 - 1e-10);
            let u2v = u2[i].clamp(1e-10, 1.0 - 1e-10);
            let theta = theta_path[i];

            let density = match copula_type {
                TvCopulaType::Gaussian => {
                    // Gaussian copula density with correlation theta
                    let z1 = Self::inv_normal_cdf(u1v);
                    let z2 = Self::inv_normal_cdf(u2v);
                    let det = (1.0 - theta * theta).max(1e-10);
                    let quad = (theta * theta * (z1 * z1 + z2 * z2) - 2.0 * theta * z1 * z2) / det;
                    (-0.5 * (det.ln() + quad)).exp()
                        / (2.0 * std::f64::consts::PI * det).sqrt().max(1e-300)
                }
                TvCopulaType::Clayton => {
                    let sum = u1v.powf(-theta) + u2v.powf(-theta) - 1.0;
                    if sum > 0.0 {
                        (1.0 + theta)
                            * sum.powf(-1.0 / theta - 2.0)
                            * (u1v * u2v).powf(-theta - 1.0)
                    } else {
                        1e-300
                    }
                }
                TvCopulaType::Gumbel => {
                    let l1 = (-u1v.ln()).powf(theta);
                    let l2 = (-u2v.ln()).powf(theta);
                    let s = (l1 + l2).powf(1.0 / theta);
                    if s > 0.0 {
                        s.exp() / (u1v * u2v).max(1e-300)
                    } else {
                        1e-300
                    }
                }
            };

            if density > 0.0 {
                ll += density.ln();
            }
        }

        (theta_path, ll)
    }

    fn empirical_cdf(col: &ndarray::ArrayView1<f64>) -> Array1<f64> {
        let t = col.len();
        let mut u = Array1::zeros(t);

        let mut sorted: Vec<(usize, f64)> = (0..t).map(|i| (i, col[i])).collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (rank, &(orig_idx, _)) in sorted.iter().enumerate() {
            u[orig_idx] = (rank + 1) as f64 / (t + 1) as f64;
        }

        u
    }

    fn inv_normal_cdf(p: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.inverse_cdf(p)
    }
}
