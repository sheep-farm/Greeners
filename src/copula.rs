//! Copula-based dependence modeling.
//!
//! Fits parametric copulas to estimate non-linear dependence
//! structures between variables, beyond what linear correlation
//! captures. Supports Gaussian, Clayton, and Gumbel copulas.
//!
//! The copula approach separates marginal distributions from the
//! dependence structure (Sklar's theorem):
//!
//! F(x1, ..., xk) = C(F1(x1), ..., Fk(xk); theta)
//!
//! where C is the copula function and theta is the dependence
//! parameter.
//!
//! Estimation: two-step (inference functions for margins, IFM):
//!   1. Fit marginal distributions (empirical CDF or parametric)
//!   2. Fit copula parameter via maximum likelihood on uniform[0,1]
//!      transformed data

use crate::linalg::LinalgDeterminant as _;
use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Copula types supported.
#[derive(Debug, Clone, Copy)]
pub enum CopulaType {
    Gaussian,
    Clayton,
    Gumbel,
    Frank,
}

impl fmt::Display for CopulaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CopulaType::Gaussian => write!(f, "Gaussian"),
            CopulaType::Clayton => write!(f, "Clayton"),
            CopulaType::Gumbel => write!(f, "Gumbel"),
            CopulaType::Frank => write!(f, "Frank"),
        }
    }
}

/// Result of copula fitting.
#[derive(Debug)]
pub struct CopulaResult {
    /// Copula type
    pub copula_type: CopulaType,
    /// Dependence parameter(s). For Gaussian: correlation matrix (k x k).
    /// For Clayton/Gumbel/Frank: single theta parameter.
    pub theta: f64,
    /// Correlation matrix (for Gaussian copula)
    pub corr_matrix: Array2<f64>,
    /// Kendall's tau (pairwise)
    pub kendall_tau: Array2<f64>,
    /// Spearman's rho (pairwise)
    pub spearman_rho: Array2<f64>,
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
}

impl fmt::Display for CopulaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Copula Dependence Model ")?;
        writeln!(f, "Type: {}", self.copula_type)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;

        match self.copula_type {
            CopulaType::Gaussian => {
                writeln!(f, "\n  Correlation matrix:")?;
                write!(f, "  {:<10} ", "")?;
                for j in 0..self.n_vars {
                    let name = self.var_names.get(j).map(|s| s.as_str()).unwrap_or("?");
                    write!(f, "{:>10} ", name)?;
                }
                writeln!(f)?;
                for i in 0..self.n_vars {
                    let name = self.var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
                    write!(f, "  {:<10} ", name)?;
                    for j in 0..self.n_vars {
                        write!(f, "{:>10.4} ", self.corr_matrix[(i, j)])?;
                    }
                    writeln!(f)?;
                }
            }
            CopulaType::Clayton => {
                writeln!(f, "{:<20} {:>12.6}", "theta:", self.theta)?;
                writeln!(f, "  (theta > 0: lower tail dependence)")?;
            }
            CopulaType::Gumbel => {
                writeln!(f, "{:<20} {:>12.6}", "theta:", self.theta)?;
                writeln!(f, "  (theta >= 1: upper tail dependence)")?;
            }
            CopulaType::Frank => {
                writeln!(f, "{:<20} {:>12.6}", "theta:", self.theta)?;
                writeln!(f, "  (theta != 0: symmetric dependence)")?;
            }
        }

        writeln!(f, "\n  Kendall's tau (pairwise):")?;
        for i in 0..self.n_vars {
            for j in (i + 1)..self.n_vars {
                let n1 = self.var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
                let n2 = self.var_names.get(j).map(|s| s.as_str()).unwrap_or("?");
                writeln!(f, "  {} -- {}: {:.6}", n1, n2, self.kendall_tau[(i, j)])?;
            }
        }

        writeln!(f, "\n  Spearman's rho (pairwise):")?;
        for i in 0..self.n_vars {
            for j in (i + 1)..self.n_vars {
                let n1 = self.var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
                let n2 = self.var_names.get(j).map(|s| s.as_str()).unwrap_or("?");
                writeln!(f, "  {} -- {}: {:.6}", n1, n2, self.spearman_rho[(i, j)])?;
            }
        }

        writeln!(
            f,
            "\n{:<20} {:>12.4}",
            "Log-likelihood:", self.log_likelihood
        )?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        write!(f, "{:=^78}", "")
    }
}

pub struct Copula;

impl Copula {
    /// Fit a copula model to multivariate data.
    ///
    /// # Arguments
    /// * `x` - Data matrix (T x k)
    /// * `copula_type` - Type of copula to fit
    /// * `var_names` - Optional variable names
    pub fn fit(
        x: &Array2<f64>,
        copula_type: CopulaType,
        var_names: Option<Vec<String>>,
    ) -> Result<CopulaResult, GreenersError> {
        let t = x.nrows();
        let k = x.ncols();
        if t < 5 || k < 2 {
            return Err(GreenersError::InvalidOperation(
                "Copula: need at least 5 obs and 2 variables".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());

        // Step 1: Transform to uniform[0,1] via empirical CDF (rank-based)
        let u = Self::empirical_cdf_transform(x);

        // Step 2: Compute rank correlation matrices
        let kendall_tau = Self::kendall_tau_matrix(x);
        let spearman_rho = Self::spearman_rho_matrix(x);

        // Step 3: Fit copula parameter
        let (theta, corr_matrix, log_likelihood) = match copula_type {
            CopulaType::Gaussian => Self::fit_gaussian(&u, &kendall_tau, k)?,
            CopulaType::Clayton => Self::fit_clayton(&u)?,
            CopulaType::Gumbel => Self::fit_gumbel(&u, &kendall_tau, k)?,
            CopulaType::Frank => Self::fit_frank(&u)?,
        };

        let n_params = match copula_type {
            CopulaType::Gaussian => k * (k - 1) / 2,
            _ => 1,
        };
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + (t as f64) * n_params as f64;

        Ok(CopulaResult {
            copula_type,
            theta,
            corr_matrix,
            kendall_tau,
            spearman_rho,
            log_likelihood,
            aic,
            bic,
            n_obs: t,
            n_vars: k,
            var_names: names,
        })
    }

    /// Transform data to uniform[0,1] via empirical CDF (rank/(n+1)).
    fn empirical_cdf_transform(x: &Array2<f64>) -> Array2<f64> {
        let t = x.nrows();
        let k = x.ncols();
        let mut u = Array2::zeros((t, k));

        for j in 0..k {
            // Get ranks
            let col: Vec<(usize, f64)> = (0..t).map(|i| (i, x[(i, j)])).collect();
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut ranks = vec![0usize; t];
            for (rank, &(orig_idx, _)) in sorted.iter().enumerate() {
                ranks[orig_idx] = rank + 1;
            }

            for i in 0..t {
                u[(i, j)] = ranks[i] as f64 / (t + 1) as f64;
            }
        }

        u
    }

    /// Kendall's tau matrix.
    fn kendall_tau_matrix(x: &Array2<f64>) -> Array2<f64> {
        let t = x.nrows();
        let k = x.ncols();
        let mut tau = Array2::<f64>::eye(k);

        for i in 0..k {
            for j in (i + 1)..k {
                let mut concordant = 0_i64;
                let mut discordant = 0_i64;
                for a in 0..t {
                    for b in (a + 1)..t {
                        let dx_i = x[(a, i)] - x[(b, i)];
                        let dx_j = x[(a, j)] - x[(b, j)];
                        if dx_i * dx_j > 0.0 {
                            concordant += 1;
                        } else if dx_i * dx_j < 0.0 {
                            discordant += 1;
                        }
                    }
                }
                let n_pairs = t * (t - 1) / 2;
                let val = (concordant - discordant) as f64 / n_pairs as f64;
                tau[(i, j)] = val;
                tau[(j, i)] = val;
            }
        }

        tau
    }

    /// Spearman's rho matrix.
    fn spearman_rho_matrix(x: &Array2<f64>) -> Array2<f64> {
        let t = x.nrows();
        let k = x.ncols();
        let mut rho = Array2::<f64>::eye(k);

        // Compute ranks for each column
        let mut ranks = Array2::zeros((t, k));
        for j in 0..k {
            let col: Vec<(usize, f64)> = (0..t).map(|i| (i, x[(i, j)])).collect();
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for (rank, &(orig_idx, _)) in sorted.iter().enumerate() {
                ranks[(orig_idx, j)] = (rank + 1) as f64;
            }
        }

        for i in 0..k {
            for j in (i + 1)..k {
                let mean_i = ranks.column(i).mean().unwrap_or(0.0);
                let mean_j = ranks.column(j).mean().unwrap_or(0.0);
                let mut num = 0.0;
                let mut den_i = 0.0;
                let mut den_j = 0.0;
                for a in 0..t {
                    let di = ranks[(a, i)] - mean_i;
                    let dj = ranks[(a, j)] - mean_j;
                    num += di * dj;
                    den_i += di * di;
                    den_j += dj * dj;
                }
                let val = num / (den_i * den_j).sqrt().max(1e-10);
                rho[(i, j)] = val;
                rho[(j, i)] = val;
            }
        }

        rho
    }

    /// Fit Gaussian copula: estimate correlation matrix from Kendall's tau.
    /// For Gaussian copula: tau = (2/pi) * arcsin(rho)
    /// So rho = sin(pi * tau / 2)
    fn fit_gaussian(
        _u: &Array2<f64>,
        kendall_tau: &Array2<f64>,
        k: usize,
    ) -> Result<(f64, Array2<f64>, f64), GreenersError> {
        let mut corr = Array2::<f64>::eye(k);
        for i in 0..k {
            for j in (i + 1)..k {
                let rho = (std::f64::consts::PI * kendall_tau[(i, j)] / 2.0).sin();
                corr[(i, j)] = rho;
                corr[(j, i)] = rho;
            }
        }

        // Log-likelihood (Gaussian copula)
        let n = _u.nrows();
        let corr_inv = (corr.clone() + Array2::<f64>::eye(k) * 1e-8)
            .inv()
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let det = corr.det().unwrap_or(1e-300).max(1e-300);

        // Transform u to z (inverse normal CDF)
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let mut ll = 0.0;
        for i in 0..n {
            let z: Array1<f64> = (0..k)
                .map(|j| {
                    let p = _u[(i, j)].clamp(1e-10, 1.0 - 1e-10);
                    normal.inverse_cdf(p)
                })
                .collect();
            let quad = z.dot(&corr_inv.dot(&z)) - z.mapv(|v| v * v).sum();
            ll += -0.5 * (det.ln() + quad);
        }

        Ok((0.0, corr, ll))
    }

    /// Fit Clayton copula: theta via Kendall's tau.
    /// For bivariate Clayton: tau = theta / (theta + 2)
    /// So theta = 2 * tau / (1 - tau)
    fn fit_clayton(u: &Array2<f64>) -> Result<(f64, Array2<f64>, f64), GreenersError> {
        let t = u.nrows();
        let k = u.ncols();

        // Use average pairwise Kendall's tau
        let x = Self::u_to_x(u);
        let tau_mat = Self::kendall_tau_matrix(&x);
        let mut avg_tau = 0.0;
        let mut count = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                avg_tau += tau_mat[(i, j)];
                count += 1;
            }
        }
        avg_tau /= count.max(1) as f64;
        let theta = (2.0 * avg_tau / (1.0 - avg_tau)).max(0.1);

        // Log-likelihood (bivariate Clayton copula, averaged)
        let mut ll = 0.0;
        for i in 0..t {
            for j in 0..k {
                for l in (j + 1)..k {
                    let u1 = u[(i, j)].clamp(1e-10, 1.0 - 1e-10);
                    let u2 = u[(i, l)].clamp(1e-10, 1.0 - 1e-10);
                    // Clayton copula density: (1+theta) * (u1^-theta + u2^-theta - 1)^(-1/theta-2) * (u1*u2)^(-theta-1)
                    let sum = u1.powf(-theta) + u2.powf(-theta) - 1.0;
                    if sum > 0.0 {
                        let density = (1.0 + theta)
                            * sum.powf(-1.0 / theta - 2.0)
                            * (u1 * u2).powf(-theta - 1.0);
                        if density > 0.0 {
                            ll += density.ln();
                        }
                    }
                }
            }
        }
        ll /= count.max(1) as f64;

        Ok((theta, Array2::<f64>::eye(k), ll * t as f64))
    }

    /// Fit Gumbel copula: theta via Kendall's tau.
    /// For bivariate Gumbel: tau = 1 - 1/theta
    /// So theta = 1 / (1 - tau)
    fn fit_gumbel(
        u: &Array2<f64>,
        kendall_tau: &Array2<f64>,
        k: usize,
    ) -> Result<(f64, Array2<f64>, f64), GreenersError> {
        let t = u.nrows();

        let mut avg_tau = 0.0;
        let mut count = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                avg_tau += kendall_tau[(i, j)];
                count += 1;
            }
        }
        avg_tau /= count.max(1) as f64;
        let theta = (1.0 / (1.0 - avg_tau)).max(1.0);

        // Log-likelihood (simplified)
        let mut ll = 0.0;
        for i in 0..t {
            for j in 0..k {
                for l in (j + 1)..k {
                    let u1 = u[(i, j)].clamp(1e-10, 1.0 - 1e-10);
                    let u2 = u[(i, l)].clamp(1e-10, 1.0 - 1e-10);
                    // Gumbel copula: C = exp(-((-ln u1)^theta + (-ln u2)^theta)^(1/theta))
                    let l1 = (-u1.ln()).powf(theta);
                    let l2 = (-u2.ln()).powf(theta);
                    let s = (l1 + l2).powf(1.0 / theta);
                    if s > 0.0 {
                        ll += -s;
                    }
                }
            }
        }
        ll /= count.max(1) as f64;

        Ok((theta, Array2::<f64>::eye(k), ll))
    }

    /// Fit Frank copula: theta via maximum likelihood (grid search).
    fn fit_frank(u: &Array2<f64>) -> Result<(f64, Array2<f64>, f64), GreenersError> {
        let t = u.nrows();
        let k = u.ncols();

        // Grid search over theta
        let mut best_theta = 0.0_f64;
        let mut best_ll = f64::NEG_INFINITY;

        let n_grid = 41;
        for i in 0..n_grid {
            let theta = -20.0 + 40.0 * i as f64 / (n_grid - 1) as f64;
            if theta.abs() < 0.01 {
                continue;
            }

            let mut ll = 0.0;
            for row in 0..t {
                for j in 0..k {
                    for l in (j + 1)..k {
                        let u1 = u[(row, j)].clamp(1e-10, 1.0 - 1e-10);
                        let u2 = u[(row, l)].clamp(1e-10, 1.0 - 1e-10);
                        // Frank copula density
                        let denom = (theta / (1.0 - (-theta).exp())).abs().max(1e-300);
                        let num = (-theta).exp() - 1.0;
                        let c_val = denom
                            * ((-theta * u1).exp() + (-theta * u2).exp() - num).powi(-2)
                            * (-theta * (u1 + u2)).exp().abs();
                        if c_val > 0.0 {
                            ll += c_val.ln();
                        }
                    }
                }
            }
            if ll > best_ll {
                best_ll = ll;
                best_theta = theta;
            }
        }

        Ok((best_theta, Array2::<f64>::eye(k), best_ll))
    }

    /// Convert uniform back to x (for Kendall's tau computation).
    fn u_to_x(u: &Array2<f64>) -> Array2<f64> {
        // Use quantile function of standard normal
        let normal = Normal::new(0.0, 1.0).unwrap();
        u.mapv(|p| normal.inverse_cdf(p.clamp(1e-10, 1.0 - 1e-10)))
    }
}
