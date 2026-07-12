use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::gamma::{digamma, ln_gamma};
use std::fmt;

/// Link function for Beta regression.
#[derive(Debug, Clone)]
pub enum BetaLink {
    Logit,
    Probit,
    CLogLog,
}

impl BetaLink {
    fn link(&self, mu: f64) -> f64 {
        match self {
            BetaLink::Logit => (mu / (1.0 - mu)).ln(),
            BetaLink::Probit => {
                let n = Normal::new(0.0, 1.0).unwrap();
                n.inverse_cdf(mu)
            }
            BetaLink::CLogLog => (-(-mu).ln_1p()).ln(),
        }
    }

    fn inv_link(&self, eta: f64) -> f64 {
        match self {
            BetaLink::Logit => 1.0 / (1.0 + (-eta).exp()),
            BetaLink::Probit => {
                let n = Normal::new(0.0, 1.0).unwrap();
                n.cdf(eta)
            }
            BetaLink::CLogLog => 1.0 - (-eta.exp()).exp(),
        }
    }

    fn dinv_link(&self, eta: f64) -> f64 {
        match self {
            BetaLink::Logit => {
                let p = 1.0 / (1.0 + (-eta).exp());
                p * (1.0 - p)
            }
            BetaLink::Probit => {
                let n = Normal::new(0.0, 1.0).unwrap();
                use statrs::distribution::Continuous;
                n.pdf(eta)
            }
            BetaLink::CLogLog => {
                let e = eta.exp();
                e * (-e).exp()
            }
        }
    }
}

/// Result of Beta regression.
#[derive(Debug)]
pub struct BetaResult {
    pub params: Array1<f64>,
    pub precision_param: f64,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub pseudo_r2: f64,
    pub n_obs: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub variable_names: Option<Vec<String>>,
}

impl BetaResult {
    pub fn predict(&self, x_new: &Array2<f64>, link: &BetaLink) -> Array1<f64> {
        let eta = x_new.dot(&self.params);
        eta.mapv(|e| link.inv_link(e))
    }
}

impl fmt::Display for BetaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Beta Regression ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.4}", "Log-Likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>10.4}", "BIC:", self.bic)?;
        writeln!(f, "{:<20} {:>10.4}", "Pseudo R²:", self.pseudo_r2)?;
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "Precision (phi):", self.precision_param
        )?;
        writeln!(
            f,
            "{:<20} {:>10}",
            "Converged:",
            if self.converged { "Yes" } else { "No" }
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
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
        writeln!(f, "{:=^78}", "")
    }
}

/// Beta regression model for response in (0, 1).
pub struct BetaModel;

impl BetaModel {
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        link: &BetaLink,
    ) -> Result<BetaResult, GreenersError> {
        Self::fit_with_names(y, x, link, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        link: &BetaLink,
        variable_names: Option<Vec<String>>,
    ) -> Result<BetaResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != x.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "y and x row count mismatch".into(),
            ));
        }

        for &yi in y.iter() {
            if yi <= 0.0 || yi >= 1.0 {
                return Err(GreenersError::InvalidOperation(
                    "Response must be strictly in (0, 1) for Beta regression".into(),
                ));
            }
        }

        // Initial beta from OLS on link(y), fall back to zeros if singular.
        let y_star: Array1<f64> = y.mapv(|yi| link.link(yi));
        let xtx = x.t().dot(x);
        let xty = x.t().dot(&y_star);
        let beta_init = match xtx.inv() {
            Ok(inv) => inv.dot(&xty),
            Err(_) => Array1::zeros(k),
        };

        // Optimise [beta; log_phi] jointly by BFGS.
        let mut theta = beta_init.to_vec();
        theta.push(0.0); // log_phi = 0 => phi = 1

        let (opt_theta, n_iter, converged) = bfgs_optimize(
            &theta,
            |t| neg_log_likelihood(t, y, x, link),
            |t| neg_gradient(t, y, x, link),
            2000,
            1e-5,
        );

        let beta = Array1::from_vec(opt_theta[..k].to_vec());
        let log_phi = opt_theta[k];
        let phi = log_phi.exp();

        let ll = log_likelihood(&opt_theta, y, x, link);

        // Null log-likelihood (intercept-only with same phi).
        let y_mean = y.mean().unwrap_or(0.5);
        let ll_null = (0..n)
            .map(|i| {
                let yi = y[i];
                ln_gamma(phi) - ln_gamma(y_mean * phi) - ln_gamma((1.0 - y_mean) * phi)
                    + (y_mean * phi - 1.0) * yi.ln()
                    + ((1.0 - y_mean) * phi - 1.0) * (1.0 - yi).ln()
            })
            .sum::<f64>();

        let n_params = k + 1;
        let aic = -2.0 * ll + 2.0 * n_params as f64;
        let bic = -2.0 * ll + (n_params as f64) * (n as f64).ln();
        let pseudo_r2 = 1.0 - ll / ll_null;

        // Standard errors for beta from the observed inverse Hessian of the
        // log-likelihood for theta = [beta; log_phi].  This matches the default
        // output of R's betareg package.
        let mut hess = Array2::<f64>::zeros((k + 1, k + 1));
        let eps = 1e-5;
        let g0 = neg_gradient(&opt_theta, y, x, link);
        for j in 0..=k {
            let mut theta_plus = opt_theta.clone();
            theta_plus[j] += eps;
            let g_plus = neg_gradient(&theta_plus, y, x, link);
            for l in 0..=k {
                hess[[j, l]] = (g_plus[l] - g0[l]) / eps;
            }
        }

        // Make Hessian symmetric and positive-definite for inversion.
        for j in 0..=k {
            for l in (j + 1)..=k {
                let avg = (hess[[j, l]] + hess[[l, j]]) / 2.0;
                hess[[j, l]] = avg;
                hess[[l, j]] = avg;
            }
        }

        let cov = hess.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let std_errors: Array1<f64> = (0..k)
            .map(|j| cov[[j, j]].abs().sqrt())
            .collect::<Vec<_>>()
            .into();

        let z_values = &beta / &std_errors;
        let normal = Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal.cdf(z.abs())));

        Ok(BetaResult {
            params: beta,
            precision_param: phi,
            std_errors,
            z_values,
            p_values,
            log_likelihood: ll,
            aic,
            bic,
            pseudo_r2,
            n_obs: n,
            n_iter,
            converged,
            variable_names,
        })
    }
}

// ============================================================================
// Log-likelihood, gradient and information for [beta; log_phi] parametrisation.
// ============================================================================

fn theta_to_model<'a>(
    theta: &[f64],
    y: &'a Array1<f64>,
    x: &'a Array2<f64>,
    link: &'a BetaLink,
) -> (usize, f64, Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = y.len();
    let k = theta.len() - 1;
    let beta = Array1::from_vec(theta[..k].to_vec());
    let log_phi = theta[k];
    let phi = log_phi.exp();
    let eta = x.dot(&beta);
    let mu = eta.mapv(|e| link.inv_link(e).clamp(1e-10, 1.0 - 1e-10));
    (n, phi, beta, eta, mu)
}

fn log_likelihood(theta: &[f64], y: &Array1<f64>, x: &Array2<f64>, link: &BetaLink) -> f64 {
    let (n, phi, _, _, mu) = theta_to_model(theta, y, x, link);
    let mut ll = 0.0;
    for i in 0..n {
        let mi = mu[i];
        let yi = y[i];
        ll += ln_gamma(phi) - ln_gamma(mi * phi) - ln_gamma((1.0 - mi) * phi)
            + (mi * phi - 1.0) * yi.ln()
            + ((1.0 - mi) * phi - 1.0) * (1.0 - yi).ln();
    }
    ll
}

fn neg_log_likelihood(theta: &[f64], y: &Array1<f64>, x: &Array2<f64>, link: &BetaLink) -> f64 {
    -log_likelihood(theta, y, x, link)
}

fn neg_gradient(theta: &[f64], y: &Array1<f64>, x: &Array2<f64>, link: &BetaLink) -> Vec<f64> {
    let (n, phi, _, eta, mu) = theta_to_model(theta, y, x, link);
    let k = theta.len() - 1;
    let mut grad = vec![0.0; theta.len()];

    for i in 0..n {
        let yi = y[i];
        let mi = mu[i];
        let di = link.dinv_link(eta[i]);

        let y_star = yi.ln() - (1.0 - yi).ln();
        let mu_star = digamma(mi * phi) - digamma((1.0 - mi) * phi);

        // d ll / d beta_j
        let s_beta = phi * di * (y_star - mu_star);
        for j in 0..k {
            grad[j] -= x[[i, j]] * s_beta;
        }

        // d ll / d log_phi = d ll / d phi * phi
        let s_phi = digamma(phi) - mi * digamma(mi * phi) - (1.0 - mi) * digamma((1.0 - mi) * phi)
            + mi * yi.ln()
            + (1.0 - mi) * (1.0 - yi).ln();
        grad[k] -= s_phi * phi;
    }

    grad
}

/// Small BFGS optimiser with Armijo backtracking.
fn bfgs_optimize(
    init: &[f64],
    cost: impl Fn(&[f64]) -> f64,
    gradient: impl Fn(&[f64]) -> Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize, bool) {
    let n = init.len();
    let mut params = init.to_vec();
    let mut inv_hess = vec![vec![0.0; n]; n];
    for (i, row) in inv_hess.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    let mut best_val = cost(&params);
    let mut grad = gradient(&params);
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
            converged = true;
            break;
        }

        let direction: Vec<f64> = (0..n)
            .map(|i| -(0..n).map(|j| inv_hess[i][j] * grad[j]).sum::<f64>())
            .collect();
        let slope: f64 = direction.iter().zip(grad.iter()).map(|(d, g)| d * g).sum();

        let mut step = 1.0;
        let mut new_params = params.clone();
        let mut new_val = best_val;
        let mut found = false;
        for _ in 0..30 {
            let candidate: Vec<f64> = params
                .iter()
                .zip(direction.iter())
                .map(|(p, d)| p + step * d)
                .collect();
            let val = cost(&candidate);
            if val.is_finite() && val < best_val + 1e-4 * step * slope {
                new_params = candidate;
                new_val = val;
                found = true;
                break;
            }
            step *= 0.5;
        }

        if !found {
            // Reset to steepest descent if line search fails.
            let grad_norm_sq = grad.iter().map(|g| g * g).sum::<f64>();
            if grad_norm_sq > 0.0 {
                step = 1.0;
                for _ in 0..30 {
                    let candidate: Vec<f64> = params
                        .iter()
                        .zip(grad.iter())
                        .map(|(p, g)| p - step * g / grad_norm_sq.sqrt())
                        .collect();
                    let val = cost(&candidate);
                    if val.is_finite() && val < best_val {
                        new_params = candidate;
                        new_val = val;
                        found = true;
                        break;
                    }
                    step *= 0.5;
                }
            }
            if !found {
                break;
            }
        }

        let new_grad = gradient(&new_params);
        let s: Vec<f64> = new_params
            .iter()
            .zip(params.iter())
            .map(|(a, b)| a - b)
            .collect();
        let y_vec: Vec<f64> = new_grad
            .iter()
            .zip(grad.iter())
            .map(|(a, b)| a - b)
            .collect();
        let sy: f64 = s.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();

        if sy > 1e-10 {
            let hy: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| inv_hess[i][j] * y_vec[j]).sum::<f64>())
                .collect();
            let yhy: f64 = y_vec.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();
            for i in 0..n {
                for j in 0..n {
                    inv_hess[i][j] +=
                        (sy + yhy) * s[i] * s[j] / (sy * sy) - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                }
            }
        }

        params = new_params;
        best_val = new_val;
        grad = new_grad;
    }

    (params, n_iter, converged)
}
