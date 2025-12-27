use crate::error::GreenersError;
use crate::{DataFrame, Formula};
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
    // Store X for marginal effects calculations
    x_data: Option<Array2<f64>>,
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
    /// Estimates Logit model using a formula and DataFrame.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
    ) -> Result<BinaryModelResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        Self::fit(&y, &x)
    }

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
            x_data: Some(x.clone()),
        })
    }
}

impl BinaryModelResult {
    /// Calculate Average Marginal Effects (AME)
    ///
    /// AME = (1/n) Σ_i ∂P(y_i=1|x_i)/∂x_j
    ///
    /// For Logit: ∂P/∂x_j = β_j × φ(x'β) where φ(z) = exp(z)/(1+exp(z))²
    /// For Probit: ∂P/∂x_j = β_j × φ(x'β) where φ(z) = (1/√2π)exp(-z²/2)
    ///
    /// # Arguments
    /// * `x` - Design matrix (same as used in estimation)
    ///
    /// # Returns
    /// Array of average marginal effects (one per coefficient)
    ///
    /// # Interpretation
    /// AME_j = average effect of 1-unit increase in x_j on Pr(y=1)
    pub fn average_marginal_effects(&self, x: &Array2<f64>) -> Result<Array1<f64>, GreenersError> {
        let n = x.nrows();
        let k = x.ncols();

        let mut ame = Array1::<f64>::zeros(k);

        // For each observation, calculate marginal effect and average
        for i in 0..n {
            let x_i = x.row(i);
            let xb = x_i.dot(&self.params);

            // Calculate density at x'β
            let density = if self.model_name == "Logit" {
                // Logistic density: exp(z)/(1+exp(z))²
                let exp_xb = xb.exp();
                exp_xb / (1.0 + exp_xb).powi(2)
            } else {
                // Normal density: (1/√2π)exp(-z²/2)
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.pdf(xb)
            };

            // Marginal effect for observation i: β_j × density
            for j in 0..k {
                ame[j] += self.params[j] * density;
            }
        }

        // Average across observations
        ame /= n as f64;

        Ok(ame)
    }

    /// Calculate Marginal Effects at Means (MEM)
    ///
    /// MEM = ∂P(y=1|x̄)/∂x_j evaluated at sample means x̄
    ///
    /// # Arguments
    /// * `x` - Design matrix (same as used in estimation)
    ///
    /// # Returns
    /// Array of marginal effects at means (one per coefficient)
    ///
    /// # Interpretation
    /// MEM_j = effect of 1-unit increase in x_j on Pr(y=1) for average individual
    ///
    /// # Note
    /// AME is generally preferred over MEM because:
    /// - AME accounts for heterogeneity across observations
    /// - MEM may evaluate at impossible combinations (average of dummies)
    /// - AME is more robust to non-linearities
    pub fn marginal_effects_at_means(&self, x: &Array2<f64>) -> Result<Array1<f64>, GreenersError> {
        let k = x.ncols();

        // Calculate means of X (excluding intercept if present)
        let x_means = x.mean_axis(Axis(0)).unwrap();

        // Linear prediction at means
        let xb_mean = x_means.dot(&self.params);

        // Calculate density at x̄'β
        let density = if self.model_name == "Logit" {
            let exp_xb = xb_mean.exp();
            exp_xb / (1.0 + exp_xb).powi(2)
        } else {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.pdf(xb_mean)
        };

        // Marginal effects: β_j × density
        let mut mem = Array1::<f64>::zeros(k);
        for j in 0..k {
            mem[j] = self.params[j] * density;
        }

        Ok(mem)
    }

    /// Calculate predicted probabilities
    ///
    /// # Arguments
    /// * `x` - Design matrix
    ///
    /// # Returns
    /// Array of predicted probabilities Pr(y=1|x)
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array1<f64> {
        let xb = x.dot(&self.params);

        if self.model_name == "Logit" {
            // Logistic CDF: 1/(1+exp(-x'β))
            xb.mapv(|val| 1.0 / (1.0 + (-val).exp()))
        } else {
            // Normal CDF: Φ(x'β)
            let normal = Normal::new(0.0, 1.0).unwrap();
            xb.mapv(|val| normal.cdf(val))
        }
    }

    /// Calculate confidence intervals for Average Marginal Effects (AME)
    /// using the delta method with numerical derivatives
    ///
    /// # Arguments
    /// * `x` - Design matrix
    /// * `alpha` - Significance level (default 0.05 for 95% CI)
    ///
    /// # Returns
    /// Tuple of (lower_bounds, upper_bounds) for each marginal effect
    ///
    /// # Note
    /// Uses conservative numerical approximation of standard errors
    /// For publication-quality inference, consider bootstrap methods
    pub fn ame_confidence_intervals(&self, x: &Array2<f64>, alpha: f64) -> Result<(Array1<f64>, Array1<f64>), GreenersError> {
        let ame = self.average_marginal_effects(x)?;
        let k = ame.len();

        // Critical value for normal distribution
        let z_crit = 1.96; // 95% CI (alpha=0.05)

        // Conservative SE approximation: use coefficient SE scaled by average density
        let n = x.nrows();
        let mut avg_density = 0.0;

        for i in 0..n {
            let x_i = x.row(i);
            let xb = x_i.dot(&self.params);

            let density = if self.model_name == "Logit" {
                let exp_xb = xb.exp();
                exp_xb / (1.0 + exp_xb).powi(2)
            } else {
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.pdf(xb)
            };
            avg_density += density;
        }
        avg_density /= n as f64;

        // Approximate SE for marginal effects
        let mut lower = Array1::<f64>::zeros(k);
        let mut upper = Array1::<f64>::zeros(k);

        for j in 0..k {
            let me_se = self.std_errors[j] * avg_density;
            lower[j] = ame[j] - z_crit * me_se;
            upper[j] = ame[j] + z_crit * me_se;
        }

        Ok((lower, upper))
    }

    /// Model comparison statistics
    ///
    /// # Returns
    /// Tuple of (AIC, BIC, Log-Likelihood, Pseudo R²)
    pub fn model_stats(&self) -> (f64, f64, f64, f64) {
        let k = self.params.len() as f64;
        let aic = -2.0 * self.log_likelihood + 2.0 * k;
        let bic = -2.0 * self.log_likelihood + k * (self.iterations as f64).ln();

        (aic, bic, self.log_likelihood, self.pseudo_r2)
    }
}

/// Probit implementation (Regression with Normal CDF).
pub struct Probit;

impl Probit {
    /// Estimates Probit model using a formula and DataFrame.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
    ) -> Result<BinaryModelResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        Self::fit(&y, &x)
    }

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
            x_data: Some(x.clone()),
        })
    }
}
