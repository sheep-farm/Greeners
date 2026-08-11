//! Fully Modified OLS (FMOLS) for cointegrating regressions.
//!
//! Phillips & Hansen (1990) nonparametric correction for endogeneity
//! and serial correlation in cointegrated systems.
//!
//! y_t = alpha + beta' * x_t + u_t
//!
//! where y_t and x_t are I(1) and cointegrated. The OLS estimator is
//! super-consistent but has second-order bias from endogeneity and
//! serial correlation. FMOLS corrects this via:
//!
//! 1. Long-run covariance of residuals (Omega)
//! 2. Lead/lag correction (Delta) for x
//! 3. Semiparametric correction term
//!
//! y_t^+ = y_t - Omega_ux * Omega_xx^{-1} * Delta_x

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of FMOLS estimation.
#[derive(Debug)]
pub struct FmolsResult {
    /// Intercept
    pub alpha: f64,
    /// Slope coefficients
    pub beta: Array1<f64>,
    /// SE of alpha
    pub alpha_se: f64,
    /// SE of beta
    pub beta_se: Array1<f64>,
    /// t-values of beta
    pub beta_t: Array1<f64>,
    /// p-values of beta
    pub beta_p: Array1<f64>,
    /// Long-run covariance matrix Omega
    pub omega: Array2<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of regressors
    pub n_regressors: usize,
    /// Bandwidth used for kernel
    pub bandwidth: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for FmolsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Fully Modified OLS (FMOLS) ")?;
        writeln!(f, "Phillips & Hansen (1990) nonparametric correction")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Regressors:", self.n_regressors)?;
        writeln!(f, "{:<20} {:>12}", "Bandwidth:", self.bandwidth)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            "alpha",
            self.alpha,
            self.alpha_se,
            self.alpha / self.alpha_se.max(1e-10),
            0.0
        )?;
        for i in 0..self.beta.len() {
            let name = self
                .variable_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.beta[i], self.beta_se[i], self.beta_t[i], self.beta_p[i]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct FMOLS;

impl FMOLS {
    /// Estimate FMOLS for a cointegrating regression.
    ///
    /// # Arguments
    /// * `y` - Dependent variable I(1) (T)
    /// * `x` - Regressors I(1) (T x k)
    /// * `variable_names` - Optional names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<FmolsResult, GreenersError> {
        let t = y.len();
        let k = x.ncols();
        if x.nrows() != t {
            return Err(GreenersError::ShapeMismatch(
                "FMOLS: y and x must have same length".into(),
            ));
        }
        if t < k + 5 {
            return Err(GreenersError::InvalidOperation(
                "FMOLS: too few observations".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());

        // Step 1: OLS regression to get residuals
        let mut z = Array2::zeros((t, k + 1));
        for i in 0..t {
            z[(i, 0)] = 1.0;
            for j in 0..k {
                z[(i, j + 1)] = x[(i, j)];
            }
        }
        let zt = z.t();
        let ztz = zt.dot(&z);
        let ztz_reg = &ztz + Array2::<f64>::eye(k + 1) * 1e-8;
        let ztz_inv = ztz_reg.inv()?;
        let zty = zt.dot(y);
        let ols_beta: Array1<f64> = ztz_inv.dot(&zty);

        let residuals = y - z.dot(&ols_beta);

        // Step 2: Compute long-run covariance (Omega) via Bartlett kernel
        // Bandwidth: Newey-West automatic = floor(4 * (T/100)^(2/9))
        let bw = (4.0 * (t as f64 / 100.0).powf(2.0 / 9.0)) as usize;
        let bandwidth = bw.max(1);

        // Combined residual + delta_x for long-run covariance
        // u_t = [residual_t, delta_x_t']'
        let mut combined = Array2::zeros((t - 1, k + 1));
        for i in 0..t - 1 {
            combined[(i, 0)] = residuals[i + 1]; // u_t (use t=1..T-1)
            for j in 0..k {
                combined[(i, j + 1)] = x[(i + 1, j)] - x[(i, j)]; // delta_x_t
            }
        }

        let omega = Self::long_run_covariance(&combined, bandwidth);

        // Partition Omega:
        // Omega = [[Omega_uu, Omega_ux], [Omega_xu, Omega_xx]]
        let omega_uu = omega[(0, 0)];
        let omega_ux: Array1<f64> = (0..k).map(|j| omega[(0, j + 1)]).collect();
        let _omega_xu: Array1<f64> = (0..k).map(|j| omega[(j + 1, 0)]).collect();
        let omega_xx: Array2<f64> = omega.slice(ndarray::s![1..k + 1, 1..k + 1]).to_owned();

        let omega_xx_inv = (&omega_xx + Array2::<f64>::eye(k) * 1e-10).inv()?;

        // Step 3: Compute correction term
        // Delta_plus = Omega_ux - Omega_uu * 0 (simplified for no trend)
        // y_t^+ = y_t - Omega_ux * Omega_xx^{-1} * delta_x_t (correction)
        let correction_coef = omega_ux.insert_axis(ndarray::Axis(0)).dot(&omega_xx_inv); // (1 x k)

        // Build corrected y
        let mut y_plus = y.clone();
        for i in 1..t {
            let delta_x = x.row(i).to_owned() - x.row(i - 1);
            let corr = correction_coef.dot(&delta_x);
            y_plus[i] -= corr[0];
        }

        // Step 4: FMOLS regression on corrected y
        let zty_plus = zt.dot(&y_plus);
        let fmols_beta: Array1<f64> = ztz_inv.dot(&zty_plus);

        let alpha = fmols_beta[0];
        let beta = fmols_beta.slice(ndarray::s![1..k + 1]).to_owned();

        // Step 5: Standard errors
        // FMOLS variance: Omega_uu * (Z'Z)^{-1} (simplified)
        let sigma_uu = omega_uu;
        let cov = &ztz_inv * sigma_uu;
        let std_errors = cov.diag().mapv(|v| v.sqrt());
        let alpha_se = std_errors[0];
        let beta_se = std_errors.slice(ndarray::s![1..k + 1]).to_owned();

        let t_values = &beta / &beta_se;
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        // R-squared (from original OLS)
        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let rss = residuals.dot(&residuals);
        let r_squared = if tss > 1e-15 { 1.0 - rss / tss } else { 0.0 };

        Ok(FmolsResult {
            alpha,
            beta,
            alpha_se,
            beta_se,
            beta_t: t_values,
            beta_p: p_values,
            omega,
            r_squared,
            n_obs: t,
            n_regressors: k,
            bandwidth,
            variable_names: names,
        })
    }

    /// Long-run covariance via Bartlett (Newey-West) kernel.
    fn long_run_covariance(data: &Array2<f64>, bandwidth: usize) -> Array2<f64> {
        let n = data.nrows();
        let k = data.ncols();

        // Simple covariance (Gamma_0)
        let mut omega = Array2::zeros((k, k));
        for i in 0..n {
            let row = data.row(i);
            for a in 0..k {
                for b in 0..k {
                    omega[(a, b)] += row[a] * row[b];
                }
            }
        }
        omega /= n as f64;

        // Add weighted autocovariances (Bartlett kernel)
        for lag in 1..=bandwidth {
            if lag >= n {
                break;
            }
            let weight = 1.0 - lag as f64 / (bandwidth + 1) as f64;
            let mut gamma = Array2::zeros((k, k));
            let n_lag = n - lag;
            for i in 0..n_lag {
                let row1 = data.row(i);
                let row2 = data.row(i + lag);
                for a in 0..k {
                    for b in 0..k {
                        gamma[(a, b)] += row1[a] * row2[b];
                    }
                }
            }
            gamma /= n as f64;
            omega = &omega + &gamma * weight;
            // Symmetric part
            let gamma_t = gamma.t();
            omega = &omega + &gamma_t * weight;
        }

        omega
    }
}
