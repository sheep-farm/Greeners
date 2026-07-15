//! Nonlinear Least Squares (NLS) estimation via Levenberg-Marquardt.
//!
//! Supports arbitrary nonlinear models specified as a prediction
//! function `f(params, x_row) -> y_hat`. Numerical gradients via
//! finite differences. Common functional forms (exponential, power,
//! logistic, Cobb-Douglas) are provided as helpers.

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fmt;

/// Result of NLS estimation.
#[derive(Debug)]
pub struct NlsResult {
    /// Estimated parameters
    pub params: Array1<f64>,
    /// Standard errors
    pub std_errors: Array1<f64>,
    /// t-statistics
    pub t_values: Array1<f64>,
    /// p-values
    pub p_values: Array1<f64>,
    /// Residual sum of squares
    pub rss: f64,
    /// R-squared (1 - RSS/TSS)
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Degrees of freedom (n - k)
    pub df_resid: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Number of iterations
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Residuals
    pub residuals: Array1<f64>,
    /// Fitted values
    pub fitted: Array1<f64>,
    /// Sigma (residual standard error)
    pub sigma: f64,
    /// Parameter names
    pub param_names: Option<Vec<String>>,
}

impl fmt::Display for NlsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Nonlinear Least Squares ")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Parameters:", self.n_params)?;
        writeln!(f, "{:<20} {:>12}", "Iterations:", self.n_iter)?;
        writeln!(
            f,
            "{:<20} {:>12}",
            "Converged:",
            if self.converged { "yes" } else { "no" }
        )?;
        writeln!(f, "{:<20} {:>12.6}", "RSS:", self.rss)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "Adj R-sq:", self.adj_r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "Sigma:", self.sigma)?;

        writeln!(f, "\n{:-^78}", "")?;
        let header = format!(
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Parameter", "Coef.", "Std.Err.", "t", "P>|t|"
        );
        writeln!(f, "{header}")?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.n_params {
            let name = self
                .param_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("p{}", i));
            let t_str = if self.t_values[i].is_nan() || self.t_values[i].is_infinite() {
                format!("{:>10}", "—")
            } else {
                format!("{:>10.3}", self.t_values[i])
            };
            let p_str = if self.p_values[i].is_nan() || self.p_values[i].is_infinite() {
                format!("{:>10}", "—")
            } else {
                format!("{:>10.4}", self.p_values[i])
            };
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {t_str} {p_str}",
                name, self.params[i], self.std_errors[i]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct NLS;

impl NLS {
    /// Fit NLS using Levenberg-Marquardt with numerical gradients.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Independent variables matrix (n × m)
    /// * `predict` - Function f(params, x_row) -> predicted y
    /// * `start` - Starting values for parameters
    /// * `max_iter` - Maximum iterations
    /// * `tol` - Convergence tolerance (relative change in RSS)
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        predict: &dyn Fn(&[f64], &[f64]) -> f64,
        start: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> Result<NlsResult, GreenersError> {
        let n = y.len();
        let k = start.len();
        if n == 0 || k == 0 {
            return Err(GreenersError::InvalidOperation(
                "NLS: empty data or parameters".into(),
            ));
        }
        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "y (n={n}) and x (nrows={}) must have same number of rows",
                x.nrows()
            )));
        }

        let mut params = start.to_vec();
        let mut residuals = vec![0.0_f64; n];
        let mut fitted = vec![0.0_f64; n];

        // Compute residuals for current params
        let compute_rss = |params: &[f64], residuals: &mut [f64], fitted: &mut [f64]| -> f64 {
            let mut rss = 0.0;
            for i in 0..n {
                let x_row: Vec<f64> = x.row(i).to_vec();
                let pred = predict(params, &x_row);
                fitted[i] = pred;
                let r = y[i] - pred;
                residuals[i] = r;
                rss += r * r;
            }
            rss
        };

        let mut rss = compute_rss(&params, &mut residuals, &mut fitted);
        let mut lambda = 1e-3_f64;
        let mut n_iter = 0;
        let mut converged = false;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            // Compute Jacobian (numerical gradients)
            // J[i][j] = d(residual_i)/d(param_j) = -d(f_i)/d(param_j)
            let mut jacobian = Array2::<f64>::zeros((n, k));
            let eps = 1e-8_f64;
            for j in 0..k {
                let mut params_pert = params.clone();
                let h = eps * (1.0 + params[j].abs());
                params_pert[j] += h;
                for i in 0..n {
                    let x_row: Vec<f64> = x.row(i).to_vec();
                    let pred_pert = predict(&params_pert, &x_row);
                    jacobian[(i, j)] = -(pred_pert - fitted[i]) / h;
                }
            }

            // Levenberg-Marquardt: (J'J + lambda*diag(J'J)) * delta = -J'r
            let jt = jacobian.t();
            let jtj = jt.dot(&jacobian);
            let jtr = jt.dot(&Array1::from(residuals.clone()));

            // Augment diagonal
            let mut a = jtj.clone();
            for j in 0..k {
                a[(j, j)] += lambda * jtj[(j, j)].abs().max(1e-10);
            }

            let delta = match a.inv() {
                Ok(a_inv) => a_inv.dot(&jtr),
                Err(_) => {
                    // Increase lambda and retry
                    lambda *= 10.0;
                    if lambda > 1e12 {
                        break;
                    }
                    continue;
                }
            };

            // Trial step
            let mut params_trial = params.clone();
            for j in 0..k {
                params_trial[j] += delta[j];
            }

            // Evaluate trial RSS
            let mut residuals_trial = vec![0.0_f64; n];
            let mut fitted_trial = vec![0.0_f64; n];
            let rss_trial = compute_rss(&params_trial, &mut residuals_trial, &mut fitted_trial);

            if rss_trial < rss {
                // Accept step
                params = params_trial;
                residuals = residuals_trial;
                fitted = fitted_trial;
                let improvement = (rss - rss_trial) / rss.max(1e-15);
                rss = rss_trial;
                lambda *= 0.5;
                if improvement < tol {
                    converged = true;
                    break;
                }
            } else {
                // Reject step, increase lambda
                lambda *= 10.0;
                if lambda > 1e12 {
                    break;
                }
            }
        }

        // Compute standard errors from final Jacobian
        let mut jacobian = Array2::<f64>::zeros((n, k));
        let eps = 1e-8_f64;
        for j in 0..k {
            let mut params_pert = params.clone();
            let h = eps * (1.0 + params[j].abs());
            params_pert[j] += h;
            for i in 0..n {
                let x_row: Vec<f64> = x.row(i).to_vec();
                let pred_pert = predict(&params_pert, &x_row);
                jacobian[(i, j)] = -(pred_pert - fitted[i]) / h;
            }
        }
        let jt = jacobian.t();
        let jtj = jt.dot(&jacobian);
        let df_resid = n - k;
        let sigma2 = rss / df_resid.max(1) as f64;
        let cov = match jtj.inv() {
            Ok(inv) => inv * sigma2,
            Err(_) => Array2::zeros((k, k)),
        };
        let std_errors = cov.diag().mapv(|v| v.sqrt());
        let t_values = Array1::from(params.clone()) / &std_errors;
        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values = t_values.mapv(|t| {
            if t.is_nan() || t.is_infinite() {
                f64::NAN
            } else {
                2.0 * (1.0 - t_dist.cdf(t.abs()))
            }
        });

        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let r_squared = if tss > 1e-15 { 1.0 - rss / tss } else { 0.0 };
        let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_resid.max(1) as f64;

        Ok(NlsResult {
            params: Array1::from(params),
            std_errors,
            t_values,
            p_values,
            rss,
            r_squared,
            adj_r_squared,
            n_obs: n,
            df_resid,
            n_params: k,
            n_iter,
            converged,
            residuals: Array1::from(residuals),
            fitted: Array1::from(fitted),
            sigma: sigma2.sqrt(),
            param_names: None,
        })
    }

    /// Fit with parameter names.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        predict: &dyn Fn(&[f64], &[f64]) -> f64,
        start: &[f64],
        param_names: Vec<String>,
        max_iter: usize,
        tol: f64,
    ) -> Result<NlsResult, GreenersError> {
        let mut result = Self::fit(y, x, predict, start, max_iter, tol)?;
        result.param_names = Some(param_names);
        Ok(result)
    }
}

// ── Common functional forms ──────────────────────────────────────────────

/// Exponential model: y = a * exp(b * x)
pub fn predict_exp(params: &[f64], x: &[f64]) -> f64 {
    params[0] * (params[1] * x[0]).exp()
}

/// Power model: y = a * x^b
pub fn predict_power(params: &[f64], x: &[f64]) -> f64 {
    if x[0] > 0.0 {
        params[0] * x[0].powf(params[1])
    } else {
        0.0
    }
}

/// Logistic model: y = a / (1 + exp(-b * (x - c)))
pub fn predict_logistic(params: &[f64], x: &[f64]) -> f64 {
    params[0] / (1.0 + (-(params[1] * (x[0] - params[2]))).exp())
}

/// Cobb-Douglas: y = a * x1^b1 * x2^b2 * ... * xk^bk
pub fn predict_cobb_douglas(params: &[f64], x: &[f64]) -> f64 {
    let a = params[0];
    let mut result = a;
    for (i, xi) in x.iter().enumerate() {
        if *xi > 0.0 {
            result *= xi.powf(params[1 + i]);
        }
    }
    result
}

/// CES (constant elasticity of substitution): y = a * (b1*x1^rho + b2*x2^rho)^(1/rho)
pub fn predict_ces(params: &[f64], x: &[f64]) -> f64 {
    let a = params[0];
    let b1 = params[1];
    let b2 = params[2];
    let rho = params[3];
    let inner = b1 * x[0].max(1e-10).powf(rho) + b2 * x[1].max(1e-10).powf(rho);
    a * inner.max(1e-10).powf(1.0 / rho)
}
