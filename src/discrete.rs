use crate::error::GreenersError;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Inverse;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::fmt;

/// Structure to store results from binary choice models (Logit/Probit).
#[derive(Debug)]
pub struct BinaryModelResult {
    pub model_name: String, // "Logit" or "Probit"
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub iterations: usize,
    pub log_likelihood: f64,
    pub pseudo_r2: f64, // McFadden's R2
}

impl fmt::Display for BinaryModelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" {} Regression Results (MLE) ", self.model_name)
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Dep. Variable:", "y", "Log-Likelihood:", self.log_likelihood
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Model:", self.model_name, "Pseudo R-sq:", self.pseudo_r2
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Method:", "Newton-Raphson", "Iterations:", self.iterations
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            writeln!(
                f,
                "x{:<9} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                i, self.params[i], self.std_errors[i], self.z_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// Logit implementation (Logistic Regression).
pub struct Logit;

impl Logit {
    pub fn fit(y: &Array1<f64>, x: &Array2<f64>) -> Result<BinaryModelResult, GreenersError> {
        let n = x.nrows();
        let k = x.ncols();

        let mut beta = Array1::<f64>::zeros(k);

        let tol = 1e-6;
        let max_iter = 100;
        let mut diff = 1.0;
        let mut iter = 0;
        let mut log_likelihood = 0.0;

        while diff > tol && iter < max_iter {
            // A. Linear Predictor
            let xb = x.dot(&beta);

            // B. Sigmoid (Probabilities)
            let p = xb.mapv(|val| 1.0 / (1.0 + (-val).exp()));

            // C. Gradient
            let error = y - &p;
            let gradient = x.t().dot(&error);

            // D. Hessian (W = p * (1-p))
            let w_diag = &p * &(1.0 - &p);

            let mut x_weighted = x.clone();
            for (i, mut row) in x_weighted.axis_iter_mut(Axis(0)).enumerate() {
                row *= w_diag[i];
            }
            let hessian = -x.t().dot(&x_weighted);

            // E. Update
            let neg_hessian = -hessian;
            let inv_neg_hessian = match neg_hessian.inv() {
                Ok(mat) => mat,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };

            let change = inv_neg_hessian.dot(&gradient);
            beta = &beta + &change;

            diff = change.mapv(|v| v.powi(2)).sum().sqrt();
            iter += 1;

            // Log Likelihood
            log_likelihood = 0.0;
            for i in 0..n {
                let prob = p[i].clamp(1e-10, 1.0 - 1e-10);
                if y[i] > 0.5 {
                    log_likelihood += prob.ln();
                } else {
                    log_likelihood += (1.0 - prob).ln();
                }
            }
        }

        if iter == max_iter {
            return Err(GreenersError::OptimizationFailed);
        }

        // Post-Estimation
        let xb = x.dot(&beta);
        let p = xb.mapv(|val| 1.0 / (1.0 + (-val).exp()));
        let w_diag = &p * &(1.0 - &p);

        let mut x_weighted = x.clone();
        for (i, mut row) in x_weighted.axis_iter_mut(Axis(0)).enumerate() {
            row *= w_diag[i];
        }
        let neg_hessian = x.t().dot(&x_weighted);
        let cov_matrix = neg_hessian.inv()?;

        let std_errors = cov_matrix.diag().mapv(f64::sqrt);
        let z_values = &beta / &std_errors;

        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));

        let y_mean = y.mean().unwrap_or(0.5);
        let ll_null = (n as f64) * (y_mean * y_mean.ln() + (1.0 - y_mean) * (1.0 - y_mean).ln());
        let pseudo_r2 = 1.0 - (log_likelihood / ll_null);

        Ok(BinaryModelResult {
            model_name: "Logit".to_string(),
            params: beta,
            std_errors,
            z_values,
            p_values,
            iterations: iter,
            log_likelihood,
            pseudo_r2,
        })
    }
}

/// Probit implementation (Regression with Normal CDF).
pub struct Probit;

impl Probit {
    pub fn fit(y: &Array1<f64>, x: &Array2<f64>) -> Result<BinaryModelResult, GreenersError> {
        let n = x.nrows();
        let k = x.ncols();

        let mut beta = Array1::<f64>::zeros(k);
        let normal_dist = Normal::new(0.0, 1.0).unwrap();

        let tol = 1e-6;
        let max_iter = 100;
        let mut diff = 1.0;
        let mut iter = 0;
        let mut log_likelihood = 0.0;

        while diff > tol && iter < max_iter {
            let xb = x.dot(&beta);

            // 1. Probabilities (p) & Densities (f)
            let mut p = Array1::<f64>::zeros(n);
            let mut f = Array1::<f64>::zeros(n);

            for i in 0..n {
                let val = xb[i];
                p[i] = normal_dist.cdf(val).clamp(1e-10, 1.0 - 1e-10);
                f[i] = normal_dist.pdf(val);
            }

            // 2. Gradient Term: (y - p) * (f / (p*(1-p)))
            let numerator = f.clone();
            let denominator = &p * &(1.0 - &p);
            let weight_factor = &numerator / &denominator;

            let error = y - &p;
            let score_term = &error * &weight_factor;
            let gradient = x.t().dot(&score_term);

            // 3. Hessian Weight: W = f^2 / (p*(1-p))
            let w_diag = (&f * &f) / &denominator;

            let mut x_weighted = x.clone();
            for (i, mut row) in x_weighted.axis_iter_mut(Axis(0)).enumerate() {
                row *= w_diag[i];
            }
            let hessian = -x.t().dot(&x_weighted);

            // 4. Update
            let neg_hessian = -hessian;
            let inv_neg_hessian = match neg_hessian.inv() {
                Ok(mat) => mat,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };

            let change = inv_neg_hessian.dot(&gradient);
            beta = &beta + &change;

            diff = change.mapv(|v| v.powi(2)).sum().sqrt();
            iter += 1;

            // Log Likelihood
            log_likelihood = 0.0;
            for i in 0..n {
                if y[i] > 0.5 {
                    log_likelihood += p[i].ln();
                } else {
                    log_likelihood += (1.0 - p[i]).ln();
                }
            }
        }

        if iter == max_iter {
            return Err(GreenersError::OptimizationFailed);
        }

        // Post-Estimation Stats
        let xb = x.dot(&beta);
        let mut w_diag = Array1::<f64>::zeros(n);

        for i in 0..n {
            let val = xb[i];
            let p_val = normal_dist.cdf(val).clamp(1e-10, 1.0 - 1e-10);
            let f_val = normal_dist.pdf(val);
            w_diag[i] = (f_val * f_val) / (p_val * (1.0 - p_val));
        }

        let mut x_weighted = x.clone();
        for (i, mut row) in x_weighted.axis_iter_mut(Axis(0)).enumerate() {
            row *= w_diag[i];
        }
        let neg_hessian = x.t().dot(&x_weighted);
        let cov_matrix = neg_hessian.inv()?;

        let std_errors = cov_matrix.diag().mapv(f64::sqrt);
        let z_values = &beta / &std_errors;
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));

        let y_mean = y.mean().unwrap_or(0.5);
        let ll_null = (n as f64) * (y_mean * y_mean.ln() + (1.0 - y_mean) * (1.0 - y_mean).ln());
        let pseudo_r2 = 1.0 - (log_likelihood / ll_null);

        Ok(BinaryModelResult {
            model_name: "Probit".to_string(),
            params: beta,
            std_errors,
            z_values,
            p_values,
            iterations: iter,
            log_likelihood,
            pseudo_r2,
        })
    }
}
