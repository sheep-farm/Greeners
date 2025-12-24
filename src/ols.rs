use crate::error::GreenersError;
use crate::CovarianceType; // Import the new Enum
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};
use std::fmt;

#[derive(Debug)]
pub struct OlsResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub f_statistic: f64,
    pub prob_f: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub df_resid: usize,
    pub df_model: usize,
    pub sigma: f64,
    pub cov_type: CovarianceType, // Store which type was used
}

impl fmt::Display for OlsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cov_str = match self.cov_type {
            CovarianceType::NonRobust => "Non-Robust".to_string(),
            CovarianceType::HC1 => "Robust (HC1)".to_string(),
            CovarianceType::NeweyWest(lags) => format!("HAC (Newey-West, L={})", lags),
        };

        writeln!(f, "\n{:=^78}", " OLS Regression Results ")?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Dep. Variable:", "y", "R-squared:", self.r_squared
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Model:", "OLS", "Adj. R-squared:", self.adj_r_squared
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Covariance Type:", cov_str, "F-statistic:", self.f_statistic
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4e}",
            "No. Observations:", self.n_obs, "Prob (F-statistic):", self.prob_f
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Df Residuals:", self.df_resid, "Log-Likelihood:", self.log_likelihood
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "AIC:", self.aic, "BIC:", self.bic
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8} | {:>18}",
            "Variable", "coef", "std err", "t", "P>|t|", "[0.025      0.975]"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            writeln!(
                f,
                "x{:<9} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3} | {:>8.4}  {:>8.4}",
                i,
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

pub struct OLS;

impl OLS {
    /// Fits the model. Now accepts `cov_type`.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
    ) -> Result<OlsResult, GreenersError> {
        let n = x.nrows();
        let k = x.ncols();

        if y.len() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "y: {}, X: {}",
                y.len(),
                n
            )));
        }
        if n <= k {
            return Err(GreenersError::ShapeMismatch(
                "Degrees of freedom <= 0".into(),
            ));
        }

        // 1. Beta Estimation (Same for Robust and Non-Robust)
        let x_t = x.t();
        let xt_x = x_t.dot(x);
        let xt_x_inv = xt_x.inv()?;
        let xt_y = x_t.dot(y);
        let beta = xt_x_inv.dot(&xt_y);

        // 2. Residuals
        let predicted = x.dot(&beta);
        let residuals = y - &predicted;
        let ssr = residuals.dot(&residuals);

        let df_resid = n - k;
        let df_model = k - 1;

        let sigma2 = ssr / (df_resid as f64);
        let sigma = sigma2.sqrt();

        // src/ols.rs (dentro de OLS::fit, substitua o bloco 'match cov_type')

        // 3. Covariance Matrix Selection
        let cov_matrix = match cov_type {
            CovarianceType::NonRobust => &xt_x_inv * sigma2,
            CovarianceType::HC1 => {
                let u_squared = residuals.mapv(|r| r.powi(2));
                let mut x_weighted = x.clone();
                for (i, mut row) in x_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_squared[i];
                }
                let meat = x_t.dot(&x_weighted);
                let bread = &xt_x_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                let correction = (n as f64) / (df_resid as f64);
                sandwich * correction
            }
            CovarianceType::NeweyWest(lags) => {
                // HAC Estimator (Newey-West)
                // Formula: (X'X)^-1 * [ Omega_0 + sum(w_l * (Omega_l + Omega_l')) ] * (X'X)^-1

                // 1. Calculate Omega_0 (Same as White's Matrix "Meat")
                let u_squared = residuals.mapv(|r| r.powi(2));
                let mut x_weighted = x.clone();
                for (i, mut row) in x_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_squared[i];
                }
                let mut meat = x_t.dot(&x_weighted); // This is Omega_0

                // 2. Add Autocovariance terms (Omega_l)
                // Bartlett Kernel weights: w(l) = 1 - l / (L + 1)
                for l in 1..=lags {
                    let weight = 1.0 - (l as f64) / ((lags + 1) as f64);

                    // Calculate Omega_l = sum( u_t * u_{t-l} * x_t * x_{t-l}' )
                    // Since specific lag logic is tricky in pure matrix algebra without huge memory,
                    // we iterate carefully.

                    let mut omega_l = Array2::<f64>::zeros((k, k));

                    // Sum over t where lag exists (from l to n)
                    for t in l..n {
                        let u_t = residuals[t];
                        let u_prev = residuals[t - l];

                        let x_row_t = x.row(t);
                        let x_row_prev = x.row(t - l);

                        // Outer product: (x_t * x_{t-l}') scaled by (u_t * u_{t-l})
                        // Using 'scaled_add' is efficient: matrix += alpha * (vec * vec.t)
                        // But ndarray doesn't have concise outer product add, so we do:
                        // term = (u_t * u_prev) * (x_t outer x_{t-l})

                        let scale = u_t * u_prev;

                        // Manual outer product addition for performance
                        for i in 0..k {
                            for j in 0..k {
                                omega_l[[i, j]] += scale * x_row_t[i] * x_row_prev[j];
                            }
                        }
                    }

                    // Add Weighted (Omega_l + Omega_l') to Meat
                    // meat += weight * (omega_l + omega_l.t())
                    let omega_l_t = omega_l.t();
                    let term = &omega_l + &omega_l_t;
                    meat = meat + (&term * weight);
                }

                let bread = &xt_x_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                // Small sample correction (n / n-k)
                let correction = (n as f64) / (df_resid as f64);
                sandwich * correction
            }
        };

        // 4. Standard Errors & Inference
        let std_errors = cov_matrix.diag().mapv(f64::sqrt);
        let t_values = &beta / &std_errors;

        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|_| GreenersError::OptimizationFailed)?;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - t_dist.cdf(t.abs())));

        let t_crit = t_dist.inverse_cdf(0.975);
        let margin_error = &std_errors * t_crit;
        let conf_lower = &beta - &margin_error;
        let conf_upper = &beta + &margin_error;

        // 5. Statistics
        let y_mean = y.mean().unwrap_or(0.0);
        let sst = y.mapv(|val| (val - y_mean).powi(2)).sum();
        let r_squared = if sst.abs() < 1e-12 {
            0.0
        } else {
            1.0 - (ssr / sst)
        };
        let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n as f64 - 1.0) / (df_resid as f64));

        let msm = (sst - ssr) / (df_model as f64);
        let f_statistic = if sigma2 < 1e-12 { 0.0 } else { msm / sigma2 };

        let prob_f = if df_model > 0 {
            let f_dist = FisherSnedecor::new(df_model as f64, df_resid as f64)
                .map_err(|_| GreenersError::OptimizationFailed)?;
            1.0 - f_dist.cdf(f_statistic)
        } else {
            f64::NAN
        };

        let n_f64 = n as f64;
        let log_likelihood =
            -n_f64 / 2.0 * ((2.0 * std::f64::consts::PI).ln() + (ssr / n_f64).ln() + 1.0);
        let aic = 2.0 * (k as f64) - 2.0 * log_likelihood;
        let bic = (k as f64) * n_f64.ln() - 2.0 * log_likelihood;

        Ok(OlsResult {
            params: beta,
            std_errors,
            t_values,
            p_values,
            conf_lower,
            conf_upper,
            r_squared,
            adj_r_squared,
            f_statistic,
            prob_f,
            log_likelihood,
            aic,
            bic,
            n_obs: n,
            df_resid,
            df_model,
            sigma,
            cov_type,
        })
    }
}

// Helper alias for simpler axis usage inside the function
use ndarray as nd;
