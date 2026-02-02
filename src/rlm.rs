use crate::error::GreenersError;
use crate::{CovarianceType, OLS};
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Robust norm functions for M-estimation.
#[derive(Debug, Clone)]
pub enum RobustNorm {
    /// Huber's T norm with tuning constant c (default 1.345)
    Huber(f64),
    /// Tukey's biweight with tuning constant c (default 4.685)
    Tukey(f64),
    /// Ordinary least squares (no robustification)
    LeastSquares,
    /// Andrew's wave with tuning constant c (default π)
    AndrewWave(f64),
    /// Hampel's three-part redescending with constants (a, b, c)
    /// Default: (2.0, 4.0, 8.0)
    Hampel(f64, f64, f64),
}

impl RobustNorm {
    /// Weight function w(z) = psi(z)/z for IRLS
    pub fn weights(&self, z: f64) -> f64 {
        let az = z.abs();
        match self {
            RobustNorm::Huber(c) => {
                if az <= *c {
                    1.0
                } else {
                    c / az
                }
            }
            RobustNorm::Tukey(c) => {
                if az <= *c {
                    (1.0 - (z / c).powi(2)).powi(2)
                } else {
                    0.0
                }
            }
            RobustNorm::LeastSquares => 1.0,
            RobustNorm::AndrewWave(c) => {
                if az < 1e-15 {
                    1.0
                } else if az <= *c {
                    (std::f64::consts::PI * z / c).sin() / (std::f64::consts::PI * z / c)
                } else {
                    0.0
                }
            }
            RobustNorm::Hampel(a, b, c) => {
                if az <= *a {
                    1.0
                } else if az <= *b {
                    a / az
                } else if az <= *c {
                    a * (c - az) / (az * (c - b))
                } else {
                    0.0
                }
            }
        }
    }

    /// Influence function psi(z)
    pub fn psi(&self, z: f64) -> f64 {
        let az = z.abs();
        match self {
            RobustNorm::Huber(c) => z.clamp(-*c, *c),
            RobustNorm::Tukey(c) => {
                if az <= *c {
                    z * (1.0 - (z / c).powi(2)).powi(2)
                } else {
                    0.0
                }
            }
            RobustNorm::LeastSquares => z,
            RobustNorm::AndrewWave(c) => {
                if az <= *c {
                    (std::f64::consts::PI * z / c).sin() * c / std::f64::consts::PI
                } else {
                    0.0
                }
            }
            RobustNorm::Hampel(a, b, c) => {
                if az <= *a {
                    z
                } else if az <= *b {
                    a * z.signum()
                } else if az <= *c {
                    a * z.signum() * (c - az) / (c - b)
                } else {
                    0.0
                }
            }
        }
    }
}

impl Default for RobustNorm {
    fn default() -> Self {
        RobustNorm::Huber(1.345)
    }
}

/// Result of Robust Linear Model estimation.
#[derive(Debug)]
pub struct RlmResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub scale: f64,
    pub n_iter: usize,
    pub converged: bool,
    pub n_obs: usize,
    pub variable_names: Option<Vec<String>>,
}

impl RlmResult {
    pub fn predict(&self, x_new: &Array2<f64>) -> Array1<f64> {
        x_new.dot(&self.params)
    }

    pub fn residuals(&self, y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
        y - &x.dot(&self.params)
    }

    pub fn fitted_values(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.params)
    }
}

impl fmt::Display for RlmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Robust Linear Model (M-estimation) ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.4}", "Scale (MAD):", self.scale)?;
        writeln!(
            f,
            "{:<20} {:>10}",
            "Converged:",
            if self.converged { "Yes" } else { "No" }
        )?;
        writeln!(f, "{:<20} {:>10}", "Iterations:", self.n_iter)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "t", "P>|t|", "[0.025", "0.975]"
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
                "{:<12} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3}",
                name,
                self.params[i],
                self.std_errors[i],
                self.t_values[i],
                self.p_values[i],
                self.conf_lower[i],
                self.conf_upper[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// Robust Linear Model estimator using M-estimation (IRLS).
pub struct RLM;

impl RLM {
    /// Fit RLM using IRLS with the given robust norm.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        norm: &RobustNorm,
        cov_type: CovarianceType,
    ) -> Result<RlmResult, GreenersError> {
        Self::fit_with_names(y, x, norm, cov_type, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        norm: &RobustNorm,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
    ) -> Result<RlmResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != x.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "y and x row count mismatch".into(),
            ));
        }

        // Initial OLS estimate
        let ols = OLS::fit(y, x, CovarianceType::NonRobust)?;
        let mut params = ols.params.clone();

        let max_iter = 250;
        let tol = 1e-8;
        let mut converged = false;
        let mut n_iter = 0;
        let mut scale;

        for iter in 0..max_iter {
            n_iter = iter + 1;
            let resid = y - &x.dot(&params);

            // MAD scale estimate
            let mut abs_resid: Vec<f64> = resid.iter().map(|r| r.abs()).collect();
            abs_resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_abs = if abs_resid.len().is_multiple_of(2) {
                (abs_resid[abs_resid.len() / 2 - 1] + abs_resid[abs_resid.len() / 2]) / 2.0
            } else {
                abs_resid[abs_resid.len() / 2]
            };
            scale = median_abs / 0.6745;
            if scale < 1e-15 {
                scale = 1e-15;
            }

            // Standardized residuals and weights
            let w: Array1<f64> = resid.mapv(|r| norm.weights(r / scale));

            // Weighted least squares: (X'WX)^-1 X'Wy
            let mut xtwx = Array2::<f64>::zeros((k, k));
            let mut xtwy = Array1::<f64>::zeros(k);
            for i in 0..n {
                let xi = x.row(i);
                let wi = w[i];
                for j in 0..k {
                    xtwy[j] += wi * xi[j] * y[i];
                    for l in 0..k {
                        xtwx[[j, l]] += wi * xi[j] * xi[l];
                    }
                }
            }

            let new_params = xtwx.inv()?.dot(&xtwy);

            let diff = (&new_params - &params)
                .iter()
                .map(|d| d.abs())
                .fold(0.0_f64, f64::max);
            let param_scale = params.iter().map(|p| p.abs()).fold(1.0_f64, f64::max);

            params = new_params;

            if diff / param_scale < tol {
                converged = true;
                break;
            }
        }

        // Final scale
        let resid = y - &x.dot(&params);
        let mut abs_resid: Vec<f64> = resid.iter().map(|r| r.abs()).collect();
        abs_resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_abs = if abs_resid.len().is_multiple_of(2) {
            (abs_resid[abs_resid.len() / 2 - 1] + abs_resid[abs_resid.len() / 2]) / 2.0
        } else {
            abs_resid[abs_resid.len() / 2]
        };
        scale = median_abs / 0.6745;
        if scale < 1e-15 {
            scale = 1e-15;
        }

        // Covariance matrix - use H-sandwich or standard
        let w: Array1<f64> = resid.mapv(|r| norm.weights(r / scale));
        let std_errors = Self::compute_se(x, &w, scale, &cov_type)?;

        let t_values = &params / &std_errors;
        let normal = Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));
        let z_crit = 1.96;
        let conf_lower = &params - &(&std_errors * z_crit);
        let conf_upper = &params + &(&std_errors * z_crit);

        Ok(RlmResult {
            params,
            std_errors,
            t_values,
            p_values,
            conf_lower,
            conf_upper,
            scale,
            n_iter,
            converged,
            n_obs: n,
            variable_names,
        })
    }

    fn compute_se(
        x: &Array2<f64>,
        w: &Array1<f64>,
        scale: f64,
        _cov_type: &CovarianceType,
    ) -> Result<Array1<f64>, GreenersError> {
        let k = x.ncols();
        let n = x.nrows();

        // (X'WX)^-1 scaled by s^2
        let mut xtwx = Array2::<f64>::zeros((k, k));
        for i in 0..n {
            let xi = x.row(i);
            let wi = w[i];
            for j in 0..k {
                for l in 0..k {
                    xtwx[[j, l]] += wi * xi[j] * xi[l];
                }
            }
        }
        let xtwx_inv = xtwx.inv()?;
        let se = (0..k)
            .map(|j| (xtwx_inv[[j, j]] * scale * scale).abs().sqrt())
            .collect::<Vec<_>>();
        Ok(Array1::from(se))
    }
}
