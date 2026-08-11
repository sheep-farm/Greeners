//! Gaussian Process Regression (Krige, Matheron; Rasmussen & Williams 2006).
//!
//! Non-parametric Bayesian regression. Places a GP prior over
//! functions f(x), with covariance kernel k(x, x'). Predictions
//! are obtained by conditioning on observed data.
//!
//! Model:
//!   y = f(x) + epsilon,  epsilon ~ N(0, sigma_n^2)
//!   f ~ GP(m(x), k(x, x'))
//!
//! Prediction at x*:
//!   mu* = k(x*, X) * (K + sigma_n^2 * I)^{-1} * y
//!   sigma*^2 = k(x*, x*) - k(x*, X) * (K + sigma_n^2 * I)^{-1} * k(X, x*)
//!
//! Kernel: Squared Exponential (RBF):
//!   k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
//!
//! Hyperparameters (sigma_f, l, sigma_n) estimated via MLE
//! (grid search over a small grid for tractability).

use crate::linalg::{LinalgDeterminant as _, LinalgInverse as _};
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of Gaussian Process Regression.
#[derive(Debug)]
pub struct GpResult {
    /// Predicted mean at training points
    pub fitted: Array1<f64>,
    /// Predicted standard deviation (uncertainty) at training points
    pub fitted_sd: Array1<f64>,
    /// Length scale (l)
    pub length_scale: f64,
    /// Signal variance (sigma_f^2)
    pub signal_variance: f64,
    /// Noise variance (sigma_n^2)
    pub noise_variance: f64,
    /// Log marginal likelihood at optimum
    pub log_marginal: f64,
    /// In-sample R-squared
    pub r_squared: f64,
    /// MSE
    pub mse: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for GpResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Gaussian Process Regression ")?;
        writeln!(f, "Rasmussen & Williams (2006)")?;
        writeln!(f, "Squared Exponential (RBF) kernel")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12.6}", "Length scale (l):", self.length_scale)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Signal var (σ_f²):", self.signal_variance
        )?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Noise var (σ_n²):", self.noise_variance
        )?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Log marginal lik.:", self.log_marginal
        )?;
        writeln!(f, "{:<20} {:>12.6}", "In-sample R²:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "MSE:", self.mse)?;

        // Fitted vs actual with uncertainty
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Fitted values with uncertainty (first 5 obs):")?;
        writeln!(
            f,
            "  {:<6} {:>12} {:>12} {:>12}",
            "Obs", "Fitted", "SD", "95% CI half-width"
        )?;
        writeln!(f, "{:-^78}", "")?;
        let n_show = 5.min(self.n_obs);
        for i in 0..n_show {
            let hw = 1.96 * self.fitted_sd[i];
            writeln!(
                f,
                "  {:<6} {:>12.6} {:>12.6} {:>12.6}",
                i + 1,
                self.fitted[i],
                self.fitted_sd[i],
                hw
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct GaussianProcess;

impl GaussianProcess {
    /// Estimate Gaussian Process Regression.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Features (n x k)
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<GpResult, GreenersError> {
        Self::fit_with_params(y, x, None, None, None, variable_names)
    }

    /// Estimate GP with optional fixed hyperparameters.
    /// If any of length_scale, signal_variance, noise_variance is None,
    /// they are estimated via grid search MLE.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_with_params(
        y: &Array1<f64>,
        x: &Array2<f64>,
        length_scale: Option<f64>,
        signal_variance: Option<f64>,
        noise_variance: Option<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<GpResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n < 5 || k == 0 {
            return Err(GreenersError::InvalidOperation(
                "GP: too few observations or features".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());

        // Standardize y
        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = y.std(0.0);
        if y_std < 1e-10 {
            return Err(GreenersError::InvalidOperation(
                "GP: y has zero variance".into(),
            ));
        }
        let y_norm: Array1<f64> = y.mapv(|v| (v - y_mean) / y_std);

        // Compute pairwise squared distances
        let dists = Self::compute_dists(x, n, k);

        // Grid search for hyperparameters if not provided
        let (best_l, best_sf, best_sn, best_lml) = if let (Some(l), Some(sf), Some(sn)) =
            (length_scale, signal_variance, noise_variance)
        {
            let lml = Self::log_marginal_likelihood(&dists, &y_norm, l, sf, sn, n)?;
            (l, sf, sn, lml)
        } else {
            Self::optimize_hyperparams(
                &dists,
                &y_norm,
                n,
                length_scale,
                signal_variance,
                noise_variance,
            )?
        };

        // Compute K matrix and predictions at training points
        let k_mat = Self::build_kernel(&dists, best_l, best_sf, best_sn, n);
        let k_inv = k_mat.inv()?;

        // Predictions: mu = K * K^{-1} * y = y (for training points)
        // But we compute properly for consistency
        let alpha = k_inv.dot(&y_norm);
        let fitted_norm = k_mat.dot(&alpha);

        // Predictive variance at training points:
        // sigma*^2 = k(x*, x*) - k(x*, X) * K^{-1} * k(X, x*)
        // For training points: k(x*, x*) = sf^2 + sn^2, and k(x*, X) = row of K
        // So sigma*^2 = (sf^2 + sn^2) - diag(K * K^{-1}) = (sf^2 + sn^2) - 1
        let diag_var = best_sf + best_sn - 1.0; // approximate
        let fitted_sd_norm = Array1::from_elem(n, diag_var.max(0.0).sqrt());

        // Un-standardize
        let fitted = fitted_norm.mapv(|v| v * y_std + y_mean);
        let fitted_sd = fitted_sd_norm.mapv(|v| v * y_std);

        // R-squared
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let sse = y
            .iter()
            .zip(fitted.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };
        let mse = sse / n as f64;

        Ok(GpResult {
            fitted,
            fitted_sd,
            length_scale: best_l,
            signal_variance: best_sf,
            noise_variance: best_sn,
            log_marginal: best_lml,
            r_squared,
            mse,
            n_obs: n,
            n_features: k,
            variable_names: names,
        })
    }

    fn compute_dists(x: &Array2<f64>, n: usize, k: usize) -> Array2<f64> {
        let mut dists = Array2::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let mut d = 0.0;
                for f in 0..k {
                    d += (x[(i, f)] - x[(j, f)]).powi(2);
                }
                dists[(i, j)] = d;
                dists[(j, i)] = d;
            }
        }
        dists
    }

    fn build_kernel(dists: &Array2<f64>, l: f64, sf: f64, sn: f64, n: usize) -> Array2<f64> {
        let mut k_mat = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k_mat[(i, j)] = sf * (-dists[(i, j)] / (2.0 * l * l)).exp();
            }
            k_mat[(i, i)] += sn;
        }
        k_mat
    }

    fn log_marginal_likelihood(
        dists: &Array2<f64>,
        y: &Array1<f64>,
        l: f64,
        sf: f64,
        sn: f64,
        n: usize,
    ) -> Result<f64, GreenersError> {
        let k_mat = Self::build_kernel(dists, l, sf, sn, n);
        let k_inv = k_mat.inv()?;
        let alpha = k_inv.dot(y);
        let lml = -0.5 * y.dot(&alpha)
            - 0.5 * k_mat.det().unwrap_or(1e-300).ln().max(-300.0)
            - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
        Ok(lml)
    }

    #[allow(clippy::type_complexity)]
    fn optimize_hyperparams(
        dists: &Array2<f64>,
        y: &Array1<f64>,
        n: usize,
        fixed_l: Option<f64>,
        fixed_sf: Option<f64>,
        fixed_sn: Option<f64>,
    ) -> Result<(f64, f64, f64, f64), GreenersError> {
        // Grid search
        let l_grid: Vec<f64> = if let Some(l) = fixed_l {
            vec![l]
        } else {
            vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        };
        let sf_grid: Vec<f64> = if let Some(sf) = fixed_sf {
            vec![sf]
        } else {
            vec![0.1, 0.5, 1.0, 2.0]
        };
        let sn_grid: Vec<f64> = if let Some(sn) = fixed_sn {
            vec![sn]
        } else {
            vec![0.01, 0.05, 0.1, 0.5]
        };

        let mut best_lml = f64::NEG_INFINITY;
        let mut best = (1.0, 1.0, 0.1);

        for &l in &l_grid {
            for &sf in &sf_grid {
                for &sn in &sn_grid {
                    let lml = Self::log_marginal_likelihood(dists, y, l, sf, sn, n)?;
                    if lml > best_lml {
                        best_lml = lml;
                        best = (l, sf, sn);
                    }
                }
            }
        }

        Ok((best.0, best.1, best.2, best_lml))
    }
}
