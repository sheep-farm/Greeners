use crate::error::GreenersError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
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

        // Check y in (0, 1)
        for &yi in y.iter() {
            if yi <= 0.0 || yi >= 1.0 {
                return Err(GreenersError::InvalidOperation(
                    "Response must be strictly in (0, 1) for Beta regression".into(),
                ));
            }
        }

        // Initialize beta from OLS on link(y)
        let y_star: Array1<f64> = y.mapv(|yi| link.link(yi));
        let xtx = x.t().dot(x);
        let xty = x.t().dot(&y_star);
        let mut beta = match xtx.inv() {
            Ok(inv) => inv.dot(&xty),
            Err(_) => Array1::zeros(k),
        };

        // Initialize phi
        let mut phi = 1.0;

        let max_iter = 200;
        let tol = 1e-8;
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            let eta = x.dot(&beta);
            let mu: Array1<f64> = eta.mapv(|e| link.inv_link(e).clamp(1e-10, 1.0 - 1e-10));

            // Score and Hessian for beta
            let mut score = Array1::<f64>::zeros(k);
            let mut hessian = Array2::<f64>::zeros((k, k));

            for i in 0..n {
                let yi = y[i];
                let mi = mu[i];
                let di = link.dinv_link(eta[i]);

                let y_star_i = yi.ln() - (1.0 - yi).ln();
                let mu_star_i = digamma(mi * phi) - digamma((1.0 - mi) * phi);

                let w_i = phi * di * di * (trigamma(mi * phi) + trigamma((1.0 - mi) * phi));

                let s_i = phi * di * (y_star_i - mu_star_i);

                for j in 0..k {
                    score[j] += x[[i, j]] * s_i;
                    for l in 0..k {
                        hessian[[j, l]] -= x[[i, j]] * w_i * x[[i, l]];
                    }
                }
            }

            // Update beta
            let neg_hessian = hessian.mapv(|h| -h);
            let delta = match neg_hessian.inv() {
                Ok(inv) => inv.dot(&score),
                Err(_) => break,
            };

            let new_beta = &beta + &delta;

            // Update phi via MLE (scalar optimization)
            let eta_new = x.dot(&new_beta);
            let mu_new: Array1<f64> = eta_new.mapv(|e| link.inv_link(e).clamp(1e-10, 1.0 - 1e-10));

            // Score for phi
            let mut phi_score = 0.0;
            let mut phi_hessian = 0.0;
            for i in 0..n {
                let mi = mu_new[i];
                let yi = y[i];
                phi_score +=
                    digamma(phi) - mi * digamma(mi * phi) - (1.0 - mi) * digamma((1.0 - mi) * phi)
                        + mi * yi.ln()
                        + (1.0 - mi) * (1.0 - yi).ln();
                phi_hessian += trigamma(phi)
                    - mi * mi * trigamma(mi * phi)
                    - (1.0 - mi) * (1.0 - mi) * trigamma((1.0 - mi) * phi);
            }

            if phi_hessian.abs() > 1e-15 {
                let phi_update = phi - phi_score / phi_hessian;
                if phi_update > 0.0 {
                    phi = phi_update;
                }
            }

            let diff = delta.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
            beta = new_beta;

            if diff < tol {
                converged = true;
                break;
            }
        }

        // Final log-likelihood
        let eta = x.dot(&beta);
        let mu: Array1<f64> = eta.mapv(|e| link.inv_link(e).clamp(1e-10, 1.0 - 1e-10));
        let mut ll = 0.0;
        for i in 0..n {
            let mi = mu[i];
            let yi = y[i];
            ll += ln_gamma(phi) - ln_gamma(mi * phi) - ln_gamma((1.0 - mi) * phi)
                + (mi * phi - 1.0) * yi.ln()
                + ((1.0 - mi) * phi - 1.0) * (1.0 - yi).ln();
        }

        // Null log-likelihood
        let y_mean = y.mean().unwrap_or(0.5);
        let mut ll_null = 0.0;
        for i in 0..n {
            let yi = y[i];
            ll_null += ln_gamma(phi) - ln_gamma(y_mean * phi) - ln_gamma((1.0 - y_mean) * phi)
                + (y_mean * phi - 1.0) * yi.ln()
                + ((1.0 - y_mean) * phi - 1.0) * (1.0 - yi).ln();
        }

        let n_params = k + 1; // beta + phi
        let aic = -2.0 * ll + 2.0 * n_params as f64;
        let bic = -2.0 * ll + (n_params as f64) * (n as f64).ln();
        let pseudo_r2 = 1.0 - ll / ll_null;

        // Standard errors from Fisher information
        let mut info = Array2::<f64>::zeros((k, k));
        for i in 0..n {
            let mi = mu[i];
            let di = link.dinv_link(eta[i]);
            let w = phi * di * di * (trigamma(mi * phi) + trigamma((1.0 - mi) * phi));
            for j in 0..k {
                for l in 0..k {
                    info[[j, l]] += x[[i, j]] * w * x[[i, l]];
                }
            }
        }

        let cov = info.inv()?;
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

/// Trigamma function approximation.
fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    // Use series expansion for large x, recursion for small
    let mut val = x;
    let mut result = 0.0;
    // Shift to large x
    while val < 6.0 {
        result += 1.0 / (val * val);
        val += 1.0;
    }
    // Asymptotic series
    let inv = 1.0 / (val * val);
    result += 1.0 / val + inv / 2.0 + inv / val * (1.0 / 6.0 - inv * (1.0 / 30.0 - inv / 42.0));
    result
}
