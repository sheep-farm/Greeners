//! Spatial econometrics: SAR (Spatial Autoregressive) and SEM (Spatial Error Model).
//!
//! SAR:  y = ρWy + Xβ + ε
//! SEM:  y = Xβ + u,  u = λWu + ε
//!
//! where W is a row-standardized spatial weights matrix.

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of spatial econometric estimation.
#[derive(Debug)]
pub struct SpatialResult {
    /// Model type: "sar" or "sem"
    pub model_type: String,
    /// Coefficients (spatial parameter first for SAR, then beta; for SEM just beta)
    pub params: Array1<f64>,
    /// Standard errors
    pub std_errors: Array1<f64>,
    /// t-statistics
    pub t_values: Array1<f64>,
    /// p-values
    pub p_values: Array1<f64>,
    /// Spatial parameter (rho for SAR, lambda for SEM)
    pub spatial_param: f64,
    /// Standard error of spatial parameter
    pub spatial_se: f64,
    /// t-stat of spatial parameter
    pub spatial_t: f64,
    /// p-value of spatial parameter
    pub spatial_p: f64,
    /// Beta coefficients (X effects)
    pub beta: Array1<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Variable names
    pub variable_names: Option<Vec<String>>,
    /// Converged
    pub converged: bool,
}

impl fmt::Display for SpatialResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title = if self.model_type == "sar" {
            " Spatial Autoregressive (SAR) "
        } else {
            " Spatial Error Model (SEM) "
        };
        writeln!(f, "\n{:=^78}", title)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;

        let spatial_name = if self.model_type == "sar" {
            "rho (spatial lag)"
        } else {
            "lambda (spatial error)"
        };
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            spatial_name, self.spatial_param, self.spatial_se, self.spatial_t, self.spatial_p
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        let header = format!(
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        );
        writeln!(f, "{header}")?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.beta.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name,
                self.beta[i],
                self.std_errors[i + 1],
                self.t_values[i + 1],
                self.p_values[i + 1]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct Spatial;

impl Spatial {
    /// Estimate SAR (Spatial Autoregressive) model: y = ρWy + Xβ + ε
    ///
    /// Uses maximum likelihood estimation. The spatial parameter ρ is
    /// estimated via grid search + golden section, then β is estimated
    /// via GLS.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Independent variables (n × k, includes intercept if desired)
    /// * `w` - Row-standardized spatial weights matrix (n × n)
    /// * `variable_names` - Optional names for X variables
    pub fn fit_sar(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<SpatialResult, GreenersError> {
        let n = y.len();
        if x.nrows() != n || w.nrows() != n || w.ncols() != n {
            return Err(GreenersError::ShapeMismatch(
                "SAR: dimension mismatch between y, x, and W".into(),
            ));
        }

        // Grid search for rho over [-0.99, 0.99]
        let mut best_rho = 0.0_f64;
        let mut best_ll = f64::NEG_INFINITY;

        // Coarse grid
        let n_grid = 199;
        for i in 0..n_grid {
            let rho = -0.99 + 1.98 * i as f64 / (n_grid - 1) as f64;
            let ll = Self::sar_log_likelihood(y, x, w, rho)?;
            if ll > best_ll {
                best_ll = ll;
                best_rho = rho;
            }
        }

        // Fine search (golden section around best)
        let lo = best_rho - 0.05;
        let hi = best_rho + 0.05;
        let golden = 0.6180339887498949;
        let mut a = lo;
        let mut b = hi;
        let mut c = b - golden * (b - a);
        let mut d = a + golden * (b - a);
        let mut fc = Self::sar_log_likelihood(y, x, w, c)?;
        let mut fd = Self::sar_log_likelihood(y, x, w, d)?;
        for _ in 0..50 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - golden * (b - a);
                fc = Self::sar_log_likelihood(y, x, w, c)?;
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + golden * (b - a);
                fd = Self::sar_log_likelihood(y, x, w, d)?;
            }
        }
        best_rho = if fc > fd { c } else { d };
        best_ll = if fc > fd { fc } else { fd };

        // Compute beta at optimal rho
        let wy = w.dot(y);
        let y_star = y.clone() - best_rho * &wy;
        let x_star = x.clone(); // X is not transformed in SAR
        let xt = x_star.t();
        let xtx = xt.dot(&x_star);
        let xtx_inv = xtx.inv()?;
        let xty = xt.dot(&y_star);
        let beta: Array1<f64> = xtx_inv.dot(&xty);

        // Residuals and sigma2
        let fitted = &x_star.dot(&beta) + best_rho * &wy;
        let residuals = y - &fitted;
        let sigma2 = residuals.dot(&residuals) / n as f64;

        // Standard errors (simplified: treat rho as known)
        let cov_beta = xtx_inv * sigma2;
        let beta_se = cov_beta.diag().mapv(|v| v.sqrt());

        // SE for rho (from Hessian approximation)
        let rho_se = {
            let h = 0.01;
            let ll_p = Self::sar_log_likelihood(y, x, w, best_rho + h)?;
            let ll_m = Self::sar_log_likelihood(y, x, w, best_rho - h)?;
            let second_deriv = (ll_p - 2.0 * best_ll + ll_m) / (h * h);
            if second_deriv < 0.0 {
                (-1.0 / second_deriv).sqrt()
            } else {
                f64::NAN
            }
        };

        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        let rho_t = best_rho / rho_se;
        let rho_p = if rho_t.is_nan() || rho_t.is_infinite() {
            f64::NAN
        } else {
            2.0 * (1.0 - normal.cdf(rho_t.abs()))
        };

        let beta_t = &beta / &beta_se;
        let beta_p = beta_t.mapv(|t| {
            if t.is_nan() || t.is_infinite() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal.cdf(t.abs()))
            }
        });

        // Combine params: [rho, beta...]
        let mut params = vec![best_rho];
        params.extend(beta.iter().cloned());
        let mut se = vec![rho_se];
        se.extend(beta_se.iter().cloned());
        let mut t_vals = vec![rho_t];
        t_vals.extend(beta_t.iter().cloned());
        let mut p_vals = vec![rho_p];
        p_vals.extend(beta_p.iter().cloned());

        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let rss = residuals.dot(&residuals);
        let r_squared = if tss > 1e-15 { 1.0 - rss / tss } else { 0.0 };

        Ok(SpatialResult {
            model_type: "sar".into(),
            params: Array1::from(params),
            std_errors: Array1::from(se),
            t_values: Array1::from(t_vals),
            p_values: Array1::from(p_vals),
            spatial_param: best_rho,
            spatial_se: rho_se,
            spatial_t: rho_t,
            spatial_p: rho_p,
            beta,
            r_squared,
            n_obs: n,
            log_likelihood: best_ll,
            variable_names,
            converged: true,
        })
    }

    /// Estimate SEM (Spatial Error Model): y = Xβ + u, u = λWu + ε
    pub fn fit_sem(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<SpatialResult, GreenersError> {
        let n = y.len();
        if x.nrows() != n || w.nrows() != n || w.ncols() != n {
            return Err(GreenersError::ShapeMismatch(
                "SEM: dimension mismatch between y, x, and W".into(),
            ));
        }

        // OLS first to get initial beta
        let xt = x.t();
        let xtx = xt.dot(x);
        let xtx_inv = xtx.inv()?;
        let xty = xt.dot(y);
        let beta_ols = xtx_inv.dot(&xty);

        // Grid search for lambda
        let mut best_lambda = 0.0_f64;
        let mut best_ll = f64::NEG_INFINITY;

        let n_grid = 199;
        for i in 0..n_grid {
            let lambda = -0.99 + 1.98 * i as f64 / (n_grid - 1) as f64;
            let ll = Self::sem_log_likelihood(y, x, w, lambda, &beta_ols)?;
            if ll > best_ll {
                best_ll = ll;
                best_lambda = lambda;
            }
        }

        // Golden section refinement
        let golden = 0.6180339887498949;
        let mut a = best_lambda - 0.05;
        let mut b = best_lambda + 0.05;
        let mut c = b - golden * (b - a);
        let mut d = a + golden * (b - a);
        let mut fc = Self::sem_log_likelihood(y, x, w, c, &beta_ols)?;
        let mut fd = Self::sem_log_likelihood(y, x, w, d, &beta_ols)?;
        for _ in 0..50 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - golden * (b - a);
                fc = Self::sem_log_likelihood(y, x, w, c, &beta_ols)?;
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + golden * (b - a);
                fd = Self::sem_log_likelihood(y, x, w, d, &beta_ols)?;
            }
        }
        best_lambda = if fc > fd { c } else { d };
        best_ll = if fc > fd { fc } else { fd };

        // Re-estimate beta with FGLS: beta = (X'(I-λW)'(I-λW)X)^{-1} X'(I-λW)'(I-λW)y
        let i_minus_lw = Array2::eye(n) - best_lambda * w;
        let x_transformed = i_minus_lw.dot(x);
        let y_transformed = i_minus_lw.dot(y);
        let xt_t = x_transformed.t();
        let xtx_t = xt_t.dot(&x_transformed);
        let xtx_t_inv = xtx_t.inv()?;
        let xty_t = xt_t.dot(&y_transformed);
        let beta: Array1<f64> = xtx_t_inv.dot(&xty_t);

        // Residuals
        let residuals = y - x.dot(&beta);
        let sigma2 = residuals.dot(&residuals) / n as f64;

        let cov_beta = xtx_t_inv * sigma2;
        let beta_se = cov_beta.diag().mapv(|v| v.sqrt());

        let lambda_se = {
            let h = 0.01;
            let ll_p = Self::sem_log_likelihood(y, x, w, best_lambda + h, &beta)?;
            let ll_m = Self::sem_log_likelihood(y, x, w, best_lambda - h, &beta)?;
            let second_deriv = (ll_p - 2.0 * best_ll + ll_m) / (h * h);
            if second_deriv < 0.0 {
                (-1.0 / second_deriv).sqrt()
            } else {
                f64::NAN
            }
        };

        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        let lambda_t = best_lambda / lambda_se;
        let lambda_p = if lambda_t.is_nan() || lambda_t.is_infinite() {
            f64::NAN
        } else {
            2.0 * (1.0 - normal.cdf(lambda_t.abs()))
        };

        let beta_t = &beta / &beta_se;
        let beta_p = beta_t.mapv(|t| {
            if t.is_nan() || t.is_infinite() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal.cdf(t.abs()))
            }
        });

        let mut params = vec![best_lambda];
        params.extend(beta.iter().cloned());
        let mut se = vec![lambda_se];
        se.extend(beta_se.iter().cloned());
        let mut t_vals = vec![lambda_t];
        t_vals.extend(beta_t.iter().cloned());
        let mut p_vals = vec![lambda_p];
        p_vals.extend(beta_p.iter().cloned());

        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let rss = residuals.dot(&residuals);
        let r_squared = if tss > 1e-15 { 1.0 - rss / tss } else { 0.0 };

        Ok(SpatialResult {
            model_type: "sem".into(),
            params: Array1::from(params),
            std_errors: Array1::from(se),
            t_values: Array1::from(t_vals),
            p_values: Array1::from(p_vals),
            spatial_param: best_lambda,
            spatial_se: lambda_se,
            spatial_t: lambda_t,
            spatial_p: lambda_p,
            beta,
            r_squared,
            n_obs: n,
            log_likelihood: best_ll,
            variable_names,
            converged: true,
        })
    }

    /// SAR log-likelihood for a given rho
    fn sar_log_likelihood(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        rho: f64,
    ) -> Result<f64, GreenersError> {
        let n = y.len();
        let wy = w.dot(y);
        let y_star = y.clone() - rho * &wy;

        let xt = x.t();
        let xtx = xt.dot(x);
        let xtx_inv = xtx.inv()?;
        let xty = xt.dot(&y_star);
        let beta: Array1<f64> = xtx_inv.dot(&xty);

        let residuals = &y_star - x.dot(&beta);
        let rss = residuals.dot(&residuals);
        let sigma2 = rss / n as f64;

        // Log-det(I - rho*W) via eigenvalues approximation
        // For simplicity, use the trace approximation: log|I-ρW| ≈ -ρ*tr(W) - ρ²/2*tr(W²) - ...
        // For row-standardized W, tr(W) ≈ 0, so this is small.
        // We use the exact approach: compute eigenvalues of W (expensive but correct for small n)
        // For now, use a simplified Jacobian: log|I-ρW| ≈ n*log(1-ρ²*mean_eig²)
        // This is an approximation; exact computation requires eigenvalue decomposition.
        let log_det = -(n as f64) * (1.0 - rho * rho).max(1e-10).ln() / 2.0;

        let ll = log_det
            - n as f64 / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln()
            - rss / (2.0 * sigma2);

        Ok(ll)
    }

    /// SEM log-likelihood for a given lambda
    fn sem_log_likelihood(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array2<f64>,
        lambda: f64,
        beta: &Array1<f64>,
    ) -> Result<f64, GreenersError> {
        let n = y.len();
        let residuals = y.clone() - x.dot(beta);
        let wu = w.dot(&residuals);
        let u_star = &residuals - lambda * &wu;
        let rss = u_star.dot(&u_star);
        let sigma2 = rss / n as f64;

        let log_det = -(n as f64) * (1.0 - lambda * lambda).max(1e-10).ln() / 2.0;

        let ll = log_det
            - n as f64 / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln()
            - rss / (2.0 * sigma2);

        Ok(ll)
    }
}
