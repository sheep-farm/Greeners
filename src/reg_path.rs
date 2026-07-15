//! Regularization Path (Efron, Hastie, Johnstone & Tibshirani 2004;
//! Zou & Hastie 2005).
//!
//! Computes the full path of Ridge, Lasso, and ElasticNet
//! coefficients as the regularization parameter lambda varies.
//!
//! Ridge: beta(lambda) = (X'X + lambda*I)^{-1} X'y
//! Lasso: coordinate descent (cyclic)
//! ElasticNet: beta(lambda) = argmin ||y - X*beta||^2 +
//!   lambda * [alpha * ||beta||_1 + (1-alpha)/2 * ||beta||^2]
//!
//! Reports coefficient paths, optimal lambda (via CV or BIC),
//! and selected variables.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of regularization path.
#[derive(Debug)]
pub struct RegPathResult {
    /// Lambda values (n_lambdas)
    pub lambdas: Array1<f64>,
    /// Coefficient paths (n_lambdas x p), WITHOUT intercept
    pub coef_path: Array2<f64>,
    /// Intercept path (n_lambdas)
    pub intercept_path: Array1<f64>,
    /// Optimal lambda (via BIC)
    pub optimal_lambda: f64,
    /// Coefficients at optimal lambda
    pub optimal_coefs: Array1<f64>,
    /// Intercept at optimal lambda
    pub optimal_intercept: f64,
    /// BIC path (n_lambdas)
    pub bic_path: Array1<f64>,
    /// Type: "ridge", "lasso", or "elasticnet"
    pub reg_type: String,
    /// Alpha (elasticnet mixing): 1.0 = lasso, 0.0 = ridge
    pub alpha: f64,
    /// Number of lambda values
    pub n_lambdas: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of predictors (excluding intercept)
    pub n_pred: usize,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Variables selected at optimal lambda (non-zero)
    pub selected_vars: Vec<String>,
}

impl fmt::Display for RegPathResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Regularization Path ")?;
        writeln!(f, "{} (alpha={:.2})", self.reg_type, self.alpha)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Predictors:", self.n_pred)?;
        writeln!(f, "{:<20} {:>12}", "Lambda values:", self.n_lambdas)?;
        writeln!(f, "{:<20} {:>12.6}", "Optimal lambda:", self.optimal_lambda)?;
        writeln!(
            f,
            "{:<20} {:>12}",
            "Selected vars:",
            self.selected_vars.len()
        )?;

        // Coefficients at optimal lambda
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Coefficients at optimal lambda:")?;
        writeln!(f, "  {:<14} {:>12}", "Variable", "Coef")?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(f, "  {:<14} {:>12.6}", "Intercept", self.optimal_intercept)?;
        for (j, name) in self.variable_names.iter().enumerate() {
            let coef = self.optimal_coefs[j];
            let marker = if coef.abs() < 1e-10 { " (0)" } else { "" };
            writeln!(f, "  {:<14} {:>12.6}{}", name, coef, marker)?;
        }

        // BIC path (first 10 and last 10)
        writeln!(f, "\n  BIC path (selected):")?;
        writeln!(f, "  {:<10} {:>12} {:>12}", "Lambda", "BIC", "Non-zero")?;
        writeln!(f, "{:-^78}", "")?;
        let n_show = 10.min(self.n_lambdas);
        let step = (self.n_lambdas / n_show).max(1);
        for idx in (0..self.n_lambdas).step_by(step) {
            let n_nonzero = (0..self.n_pred)
                .filter(|&j| self.coef_path[(idx, j)].abs() > 1e-10)
                .count();
            writeln!(
                f,
                "  {:<10.4} {:>12.2} {:>12}",
                self.lambdas[idx], self.bic_path[idx], n_nonzero
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct RegPath;

impl RegPath {
    /// Compute regularization path.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Predictors (n x p), WITHOUT intercept
    /// * `reg_type` - "ridge", "lasso", or "elasticnet"
    /// * `alpha` - ElasticNet mixing (1.0=lasso, 0.0=ridge), default 1.0
    /// * `n_lambdas` - Number of lambda values (default 50)
    /// * `variable_names` - Optional variable names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        reg_type: &str,
        alpha: Option<f64>,
        n_lambdas: Option<usize>,
        variable_names: Option<Vec<String>>,
    ) -> Result<RegPathResult, GreenersError> {
        let n = y.len();
        let p = x.ncols();
        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "RegPath: y and x must have same n".into(),
            ));
        }
        if n < p + 2 {
            return Err(GreenersError::InvalidOperation(
                "RegPath: need more observations than predictors".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..p).map(|i| format!("x{}", i)).collect());
        let a = alpha.unwrap_or(1.0).clamp(0.0, 1.0);
        let n_lam = n_lambdas.unwrap_or(50);

        // Standardize x and y
        let x_mean: Array1<f64> = (0..p)
            .map(|j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64)
            .collect();
        let x_std: Array1<f64> = (0..p)
            .map(|j| {
                let var = (0..n).map(|i| (x[(i, j)] - x_mean[j]).powi(2)).sum::<f64>() / n as f64;
                var.sqrt().max(1e-10)
            })
            .collect();
        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = y.std(0.0).max(1e-10);

        let x_norm: Array2<f64> = {
            let mut data: Vec<f64> = Vec::with_capacity(n * p);
            for i in 0..n {
                for j in 0..p {
                    data.push((x[(i, j)] - x_mean[j]) / x_std[j]);
                }
            }
            Array2::from_shape_vec((n, p), data).unwrap()
        };
        let y_norm: Array1<f64> = y.mapv(|v| (v - y_mean) / y_std);

        // Compute lambda_max = max |x_j' y| / (n * alpha)
        let xty: Array1<f64> = x_norm.t().dot(&y_norm);
        let lambda_max =
            xty.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max) / (n as f64 * a.max(0.01));
        let lambda_min = lambda_max * 1e-4;

        // Generate lambda path (log-spaced)
        let lambdas: Array1<f64> = (0..n_lam)
            .map(|i| {
                let t = i as f64 / (n_lam - 1) as f64;
                lambda_max * (lambda_min / lambda_max).powf(t)
            })
            .collect();

        let mut coef_path = Array2::zeros((n_lam, p));
        let mut intercept_path = Array1::zeros(n_lam);
        let mut bic_path = Array1::zeros(n_lam);

        for (idx, &lam) in lambdas.iter().enumerate() {
            let coefs = if reg_type == "ridge" {
                Self::ridge_fit(&x_norm, &y_norm, lam, n, p)?
            } else if reg_type == "lasso" {
                Self::lasso_fit(&x_norm, &y_norm, lam, 1.0, n, p)?
            } else {
                Self::lasso_fit(&x_norm, &y_norm, lam, a, n, p)?
            };

            // Un-standardize coefficients
            for j in 0..p {
                coef_path[(idx, j)] = coefs[j] * y_std / x_std[j];
            }
            intercept_path[idx] =
                y_mean - (0..p).map(|j| coef_path[(idx, j)] * x_mean[j]).sum::<f64>();

            // BIC: compute residual SS
            let mut sse = 0.0;
            for i in 0..n {
                let pred = intercept_path[idx]
                    + (0..p).map(|j| coef_path[(idx, j)] * x[(i, j)]).sum::<f64>();
                sse += (y[i] - pred).powi(2);
            }
            let n_nonzero = (0..p)
                .filter(|&j| coef_path[(idx, j)].abs() > 1e-10)
                .count();
            let sigma2 = sse / n as f64;
            bic_path[idx] = n as f64 * sigma2.ln() + n_nonzero as f64 * (n as f64).ln();
        }

        // Find optimal lambda (min BIC)
        let mut best_idx = 0;
        let mut best_bic = bic_path[0];
        for i in 1..n_lam {
            if bic_path[i] < best_bic {
                best_bic = bic_path[i];
                best_idx = i;
            }
        }
        let optimal_lambda = lambdas[best_idx];
        let optimal_coefs = coef_path.row(best_idx).to_owned();
        let optimal_intercept = intercept_path[best_idx];

        let selected_vars: Vec<String> = (0..p)
            .filter(|&j| optimal_coefs[j].abs() > 1e-10)
            .map(|j| names[j].clone())
            .collect();

        Ok(RegPathResult {
            lambdas,
            coef_path,
            intercept_path,
            optimal_lambda,
            optimal_coefs,
            optimal_intercept,
            bic_path,
            reg_type: reg_type.to_string(),
            alpha: a,
            n_lambdas: n_lam,
            n_obs: n,
            n_pred: p,
            variable_names: names,
            selected_vars,
        })
    }

    fn ridge_fit(
        x: &Array2<f64>,
        y: &Array1<f64>,
        lambda: f64,
        _n: usize,
        p: usize,
    ) -> Result<Array1<f64>, GreenersError> {
        let xt = x.t();
        let xtx = xt.dot(x);
        let mut xtx_reg = xtx.clone();
        for j in 0..p {
            xtx_reg[(j, j)] += lambda;
        }
        let xtx_inv = xtx_reg.inv()?;
        let xty = xt.dot(y);
        Ok(xtx_inv.dot(&xty))
    }

    fn lasso_fit(
        x: &Array2<f64>,
        y: &Array1<f64>,
        lambda: f64,
        alpha: f64,
        n: usize,
        p: usize,
    ) -> Result<Array1<f64>, GreenersError> {
        // Coordinate descent
        let mut beta: Array1<f64> = Array1::zeros(p);
        let lam = lambda * n as f64; // scale

        // Precompute column-wise products
        let mut col_sq = vec![0.0_f64; p];
        for j in 0..p {
            for i in 0..n {
                col_sq[j] += x[(i, j)] * x[(i, j)];
            }
        }

        let max_iter = 100;
        for _ in 0..max_iter {
            let mut max_change = 0.0;
            for j in 0..p {
                // Compute partial residual: r = y - X * beta + X_j * beta_j
                let mut rho = 0.0;
                for i in 0..n {
                    let mut pred = 0.0;
                    for jj in 0..p {
                        if jj != j {
                            pred += x[(i, jj)] * beta[jj];
                        }
                    }
                    rho += x[(i, j)] * (y[i] - pred);
                }

                // Soft thresholding
                let l1 = lam * alpha;
                let l2 = lam * (1.0 - alpha);
                let denom = col_sq[j] + l2;
                if denom < 1e-15 {
                    continue;
                }
                let new_beta = if rho > l1 {
                    (rho - l1) / denom
                } else if rho < -l1 {
                    (rho + l1) / denom
                } else {
                    0.0
                };

                let change = (new_beta - beta[j]).abs();
                if change > max_change {
                    max_change = change;
                }
                beta[j] = new_beta;
            }
            if max_change < 1e-8 {
                break;
            }
        }

        Ok(beta)
    }
}
