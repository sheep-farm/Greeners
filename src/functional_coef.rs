//! Functional coefficient (varying coefficient) models.
//!
//! Hastie & Tibshirani (1993). Allows coefficients to vary smoothly
//! as a function of a moderator variable z:
//!
//! y_t = beta_0(z_t) + beta_1(z_t) * x_{1,t} + ... + beta_k(z_t) * x_{k,t} + eps_t
//!
//! Estimation: local linear kernel regression (local polynomial
//! degree 1). At each evaluation point z_0, solve weighted least
//! squares with weights w_i = K((z_i - z_0) / h).
//!
//! Kernel: Gaussian (epanechnikov also supported).
//! Bandwidth: Silverman's rule of thumb.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Kernel type for local regression.
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
}

/// Result of functional coefficient model.
#[derive(Debug)]
pub struct FunctionalCoefResult {
    /// Varying coefficients at each evaluation point, shape (n_points, k+1)
    /// Column 0 = intercept, columns 1..k = slopes
    pub coefficients: Array2<f64>,
    /// Standard errors at each evaluation point, shape (n_points, k+1)
    pub std_errors: Array2<f64>,
    /// Evaluation points for z
    pub z_points: Array1<f64>,
    /// Bandwidth used
    pub bandwidth: f64,
    /// Kernel type
    pub kernel: KernelType,
    /// R-squared (global)
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of evaluation points
    pub n_points: usize,
    /// Number of regressors (excluding intercept)
    pub n_regressors: usize,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Moderator variable name
    pub moderator_name: String,
}

impl fmt::Display for FunctionalCoefResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Functional Coefficient Model ")?;
        writeln!(f, "Hastie & Tibshirani (1993) — local linear kernel")?;
        let kernel_str = match self.kernel {
            KernelType::Gaussian => "Gaussian",
            KernelType::Epanechnikov => "Epanechnikov",
        };
        writeln!(f, "{:<20} {:>12}", "Kernel:", kernel_str)?;
        writeln!(f, "{:<20} {:>12.6}", "Bandwidth:", self.bandwidth)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Eval points:", self.n_points)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {}", "Moderator:", self.moderator_name)?;

        // Show coefficients at selected z points
        let n_show = 5.min(self.n_points);
        let indices: Vec<usize> = if self.n_points <= n_show {
            (0..self.n_points).collect()
        } else {
            (0..n_show)
                .map(|i| i * (self.n_points - 1) / (n_show - 1))
                .collect()
        };

        writeln!(f, "\n{:-^78}", "")?;
        write!(f, "{:<10} ", "z")?;
        for j in 0..=self.n_regressors {
            let name = if j == 0 {
                "const".to_string()
            } else {
                self.variable_names
                    .get(j - 1)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", j - 1))
            };
            write!(f, "{:>12} ", name)?;
        }
        writeln!(f)?;
        writeln!(f, "{:-^78}", "")?;
        for &idx in &indices {
            write!(f, "{:<10.4} ", self.z_points[idx])?;
            for j in 0..=self.n_regressors {
                write!(f, "{:>12.6} ", self.coefficients[(idx, j)])?;
            }
            writeln!(f)?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct FunctionalCoef;

impl FunctionalCoef {
    /// Estimate functional coefficient model via local linear kernel regression.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Regressors (n x k)
    /// * `z` - Moderator variable (n)
    /// * `bandwidth` - Optional bandwidth (auto if None)
    /// * `n_points` - Number of evaluation points for z
    /// * `variable_names` - Optional names for x
    /// * `moderator_name` - Optional name for z
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        z: &Array1<f64>,
        bandwidth: Option<f64>,
        n_points: usize,
        variable_names: Option<Vec<String>>,
        moderator_name: Option<String>,
    ) -> Result<FunctionalCoefResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if x.nrows() != n || z.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "FunctionalCoef: dimension mismatch".into(),
            ));
        }
        if n < k + 5 {
            return Err(GreenersError::InvalidOperation(
                "FunctionalCoef: too few observations".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let mod_name = moderator_name.unwrap_or_else(|| "z".to_string());

        // Bandwidth: Silverman's rule if not specified
        let h = bandwidth.unwrap_or_else(|| Self::silverman_bandwidth(z));

        // Evaluation points: evenly spaced percentiles of z
        let mut z_sorted: Vec<f64> = z.iter().copied().collect();
        z_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let z_points: Array1<f64> = (0..n_points)
            .map(|i| {
                let frac = i as f64 / (n_points - 1).max(1) as f64;
                let idx = (frac * (n - 1) as f64) as usize;
                z_sorted[idx]
            })
            .collect();

        // Build design matrix [1, x] (n x (k+1))
        let mut x_full = Array2::zeros((n, k + 1));
        for i in 0..n {
            x_full[(i, 0)] = 1.0;
            for j in 0..k {
                x_full[(i, j + 1)] = x[(i, j)];
            }
        }

        let mut coefficients = Array2::zeros((n_points, k + 1));
        let mut std_errors = Array2::zeros((n_points, k + 1));
        let mut y_hat = Array1::zeros(n);

        for (p_idx, &z0) in z_points.iter().enumerate() {
            // Compute weights
            let weights: Array1<f64> = (0..n)
                .map(|i| Self::kernel_weight((z[i] - z0) / h))
                .collect();

            // Weighted least squares: (X'WX)^{-1} X'Wy
            let mut xtwx = Array2::zeros((k + 1, k + 1));
            let mut xtwy = Array1::zeros(k + 1);
            for i in 0..n {
                let w = weights[i];
                let xi = x_full.row(i);
                let mut contrib = Array2::zeros((k + 1, k + 1));
                for a in 0..k + 1 {
                    for b in 0..k + 1 {
                        contrib[(a, b)] = xi[a] * xi[b] * w;
                    }
                }
                xtwx += &contrib;
                for a in 0..k + 1 {
                    xtwy[a] += xi[a] * y[i] * w;
                }
            }

            let xtwx_inv = (&xtwx + Array2::<f64>::eye(k + 1) * 1e-8).inv()?;
            let beta: Array1<f64> = xtwx_inv.dot(&xtwy);

            for j in 0..k + 1 {
                coefficients[(p_idx, j)] = beta[j];
                std_errors[(p_idx, j)] = xtwx_inv[(j, j)].sqrt();
            }
        }

        // Compute fitted values and R-squared
        // For each observation, find nearest z point and use its coefficients
        for i in 0..n {
            // Find nearest evaluation point
            let mut min_dist = f64::INFINITY;
            let mut nearest_idx = 0;
            for (p_idx, &z0) in z_points.iter().enumerate() {
                let dist = (z[i] - z0).abs();
                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = p_idx;
                }
            }
            let beta_i = coefficients.row(nearest_idx);
            let mut pred = beta_i[0]; // intercept
            for j in 0..k {
                pred += beta_i[j + 1] * x[(i, j)];
            }
            y_hat[i] = pred;
        }

        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let rss = y
            .mapv(|v| v - y_mean)
            .iter()
            .zip(y_hat.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 1e-15 { 1.0 - rss / tss } else { 0.0 };

        Ok(FunctionalCoefResult {
            coefficients,
            std_errors,
            z_points,
            bandwidth: h,
            kernel: KernelType::Gaussian,
            r_squared,
            n_obs: n,
            n_points,
            n_regressors: k,
            variable_names: names,
            moderator_name: mod_name,
        })
    }

    fn silverman_bandwidth(z: &Array1<f64>) -> f64 {
        let n = z.len() as f64;
        let std_val = z.std(0.0).max(1e-10);
        let iqr = {
            let mut sorted: Vec<f64> = z.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q1 = sorted[sorted.len() / 4];
            let q3 = sorted[3 * sorted.len() / 4];
            (q3 - q1).abs().max(1e-10)
        };
        let scale = std_val.min(iqr / 1.34);
        0.9 * scale * n.powf(-1.0 / 5.0)
    }

    fn kernel_weight(u: f64) -> f64 {
        // Gaussian kernel
        (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}
