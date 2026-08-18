use crate::glm::{Family, Link};
use greeners_core::error::GreenersError;
use greeners_core::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

use greeners_core::bspline::BSplineBasis;

// ─── GLMGam ────────────────────────────────────────────────────────────────────

/// Result of GLM-GAM estimation.
#[derive(Debug)]
pub struct GamResult {
    pub params: Array1<f64>,
    /// Number of parametric (linear) terms
    pub n_linear: usize,
    /// Number of smooth terms
    pub n_smooth: usize,
    /// Effective degrees of freedom for smooth terms
    pub edf: f64,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    /// GCV score
    pub gcv_score: f64,
    pub scale: f64,
    pub n_obs: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for GamResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " GLM-GAM (Penalized Splines) ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Linear terms:", self.n_linear)?;
        writeln!(f, "{:<20} {:>10}", "Smooth terms:", self.n_smooth)?;
        writeln!(f, "{:<20} {:>10.2}", "EDF:", self.edf)?;
        writeln!(f, "{:<20} {:>10.4}", "GCV:", self.gcv_score)?;
        writeln!(f, "{:<20} {:>10.4}", "Scale:", self.scale)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        let show = self.n_linear.min(self.params.len());
        for i in 0..show {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                name, self.params[i], self.std_errors[i], self.z_values[i], self.p_values[i]
            )?;
        }
        if self.params.len() > show {
            writeln!(
                f,
                "... ({} smooth basis coefficients not shown)",
                self.params.len() - show
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// GLM with Generalized Additive Model via penalized splines.
pub struct GLMGam;

impl GLMGam {
    /// Fit GLM-GAM.
    ///
    /// - x_linear: parametric design matrix (n x p), including intercept if desired
    /// - x_smooth: smooth basis matrix (n x q), from BSplineBasis::new()
    /// - family: distribution family
    /// - link: link function
    /// - alpha: smoothing parameter(s)
    pub fn fit(
        y: &Array1<f64>,
        x_linear: &Array2<f64>,
        x_smooth: &Array2<f64>,
        family: &Family,
        link: &Link,
        alpha: f64,
    ) -> Result<GamResult, GreenersError> {
        Self::fit_with_names(y, x_linear, x_smooth, family, link, alpha, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x_linear: &Array2<f64>,
        x_smooth: &Array2<f64>,
        family: &Family,
        link: &Link,
        alpha: f64,
        variable_names: Option<Vec<String>>,
    ) -> Result<GamResult, GreenersError> {
        let n = y.len();
        let p = x_linear.ncols();
        let q = x_smooth.ncols();
        let total_k = p + q;

        if n != x_linear.nrows() || n != x_smooth.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "Dimension mismatch in GLMGam inputs".into(),
            ));
        }

        // Combine X = [X_linear | X_smooth]
        let mut x_full = Array2::<f64>::zeros((n, total_k));
        for i in 0..n {
            for j in 0..p {
                x_full[[i, j]] = x_linear[[i, j]];
            }
            for j in 0..q {
                x_full[[i, p + j]] = x_smooth[[i, j]];
            }
        }

        // Penalty matrix: penalize only smooth terms
        let s_penalty = BSplineBasis::penalty_matrix(q);
        let mut penalty = Array2::<f64>::zeros((total_k, total_k));
        for i in 0..q {
            for j in 0..q {
                penalty[[p + i, p + j]] = alpha * s_penalty[[i, j]];
            }
        }

        // Penalized IRLS
        let mut beta = Array1::<f64>::zeros(total_k);
        let max_iter = 100;
        let tol = 1e-6;
        let mut converged = false;
        let mut n_iter = 0;
        #[allow(unused_assignments)]
        let mut scale = 1.0;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            let eta = x_full.dot(&beta);
            let mu: Array1<f64> = eta.mapv(|e| apply_inv_link(link, e));

            // Working weights and working response
            let mut w = Array1::<f64>::zeros(n);
            let mut z = Array1::<f64>::zeros(n);

            for i in 0..n {
                let d = apply_dinv_link(link, eta[i]);
                let v = gam_variance(family, mu[i]);
                w[i] = (d * d / v).max(1e-10);
                z[i] = eta[i] + (y[i] - mu[i]) / d.max(1e-15);
            }

            // Penalized WLS: (X'WX + P) beta = X'Wz
            let mut xtwx = Array2::<f64>::zeros((total_k, total_k));
            let mut xtwz = Array1::<f64>::zeros(total_k);

            for i in 0..n {
                let xi = x_full.row(i);
                let wi = w[i];
                let zi = z[i];
                for a in 0..total_k {
                    xtwz[a] += wi * xi[a] * zi;
                    for b in 0..total_k {
                        xtwx[[a, b]] += wi * xi[a] * xi[b];
                    }
                }
            }

            let lhs = &xtwx + &penalty;
            let new_beta = match lhs.inv() {
                Ok(inv) => inv.dot(&xtwz),
                Err(_) => break,
            };

            let diff = (&new_beta - &beta)
                .iter()
                .map(|d| d.abs())
                .fold(0.0_f64, f64::max);

            beta = new_beta;

            if diff < tol {
                converged = true;
                break;
            }
        }

        // Compute scale
        let eta = x_full.dot(&beta);
        let mu: Array1<f64> = eta.mapv(|e| apply_inv_link(link, e));
        let resid_dev: f64 = (0..n)
            .map(|i| (y[i] - mu[i]).powi(2) / gam_variance(family, mu[i]).max(1e-10))
            .sum();

        // EDF: trace of hat matrix H = X (X'WX + P)^{-1} X'W
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let d = apply_dinv_link(link, eta[i]);
            let v = gam_variance(family, mu[i]);
            w[i] = (d * d / v).max(1e-10);
        }

        let mut xtwx = Array2::<f64>::zeros((total_k, total_k));
        for i in 0..n {
            let xi = x_full.row(i);
            let wi = w[i];
            for a in 0..total_k {
                for b in 0..total_k {
                    xtwx[[a, b]] += wi * xi[a] * xi[b];
                }
            }
        }

        let lhs = &xtwx + &penalty;
        let lhs_inv = lhs.inv()?;
        let hat_diag: f64 = (0..total_k)
            .map(|j| {
                let mut s = 0.0;
                for a in 0..total_k {
                    s += lhs_inv[[j, a]] * xtwx[[a, j]];
                }
                s
            })
            .sum();

        let edf = hat_diag;
        scale = resid_dev / (n as f64 - edf).max(1.0);

        // GCV = n * deviance / (n - edf)^2
        let gcv = n as f64 * resid_dev / (n as f64 - edf).powi(2).max(1.0);

        // Standard errors
        let cov = lhs_inv.dot(&xtwx).dot(&lhs_inv) * scale;
        let std_errors: Array1<f64> = (0..total_k)
            .map(|j| cov[[j, j]].abs().sqrt())
            .collect::<Vec<_>>()
            .into();

        let z_values = &beta / &std_errors;
        let normal = Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal.cdf(z.abs())));

        Ok(GamResult {
            params: beta,
            n_linear: p,
            n_smooth: q,
            edf,
            std_errors,
            z_values,
            p_values,
            gcv_score: gcv,
            scale,
            n_obs: n,
            n_iter,
            converged,
            variable_names,
        })
    }
}

fn apply_inv_link(link: &Link, eta: f64) -> f64 {
    match link {
        Link::Identity => eta,
        Link::Log => eta.exp(),
        Link::Logit => 1.0 / (1.0 + (-eta).exp()),
        _ => eta, // fallback to identity
    }
}

fn apply_dinv_link(link: &Link, eta: f64) -> f64 {
    match link {
        Link::Identity => 1.0,
        Link::Log => eta.exp(),
        Link::Logit => {
            let p = 1.0 / (1.0 + (-eta).exp());
            p * (1.0 - p)
        }
        _ => 1.0,
    }
}

fn gam_variance(family: &Family, mu: f64) -> f64 {
    match family {
        Family::Gaussian => 1.0,
        Family::Binomial => (mu * (1.0 - mu)).max(1e-10),
        Family::Poisson => mu.max(1e-10),
        Family::Gamma => (mu * mu).max(1e-10),
        _ => 1.0,
    }
}
