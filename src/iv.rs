use crate::error::GreenersError;
use crate::CovarianceType;
use crate::{DataFrame, Formula};
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fmt;
// Alias to facilitate Axis usage in Newey-West loop
use ndarray as nd;

#[derive(Debug)]
pub struct IvResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub n_obs: usize,
    pub df_resid: usize,
    pub sigma: f64,
    pub cov_type: CovarianceType,
}

impl fmt::Display for IvResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIX 1: Added NeweyWest option in Display
        let cov_str = match self.cov_type {
            CovarianceType::NonRobust => "Non-Robust".to_string(),
            CovarianceType::HC1 => "Robust (HC1)".to_string(),
            CovarianceType::NeweyWest(lags) => format!("HAC (Newey-West, L={})", lags),
        };

        writeln!(f, "\n{:=^78}", " IV (2SLS) Regression Results ")?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Dep. Variable:", "y", "R-squared:", self.r_squared
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Estimator:", "2SLS", "Sigma:", self.sigma
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Covariance Type:", cov_str, "No. Observations:", self.n_obs
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            writeln!(
                f,
                "x{:<9} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                i, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct IV;

impl IV {
    /// Estimates IV/2SLS model using formulas and DataFrame.
    ///
    /// # Arguments
    /// * `endog_formula` - Formula for endogenous equation (e.g., "y ~ x1 + x_endog")
    /// * `instrument_formula` - Formula for instruments (e.g., "~ z1 + z2")
    /// * `data` - DataFrame containing all variables
    /// * `cov_type` - Covariance type
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::{IV, DataFrame, Formula, CovarianceType};
    /// use ndarray::Array1;
    /// use std::collections::HashMap;
    ///
    /// let mut data = HashMap::new();
    /// data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    /// data.insert("x1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    /// data.insert("z1".to_string(), Array1::from(vec![2.0, 3.0, 4.0]));
    ///
    /// let df = DataFrame::new(data).unwrap();
    /// let endog_formula = Formula::parse("y ~ x1").unwrap();
    /// let instrument_formula = Formula::parse("~ z1").unwrap();
    ///
    /// let result = IV::from_formula(&endog_formula, &instrument_formula, &df, CovarianceType::HC1).unwrap();
    /// ```
    pub fn from_formula(
        endog_formula: &Formula,
        instrument_formula: &Formula,
        data: &DataFrame,
        cov_type: CovarianceType,
    ) -> Result<IvResult, GreenersError> {
        // Get y and X from endogenous formula
        let (y, x) = data.to_design_matrix(endog_formula)?;

        // Get Z from instrument formula (just the instruments, with intercept if specified)
        let z = if instrument_formula.intercept {
            let n_rows = data.n_rows();
            let n_cols = instrument_formula.independents.len() + 1;
            let mut z_mat = Array2::<f64>::zeros((n_rows, n_cols));

            // Add intercept
            for i in 0..n_rows {
                z_mat[[i, 0]] = 1.0;
            }

            // Add instruments
            for (j, var_name) in instrument_formula.independents.iter().enumerate() {
                let col_data = data.get(var_name)?;
                for i in 0..n_rows {
                    z_mat[[i, j + 1]] = col_data[i];
                }
            }

            z_mat
        } else {
            let n_rows = data.n_rows();
            let n_cols = instrument_formula.independents.len();
            let mut z_mat = Array2::<f64>::zeros((n_rows, n_cols));

            for (j, var_name) in instrument_formula.independents.iter().enumerate() {
                let col_data = data.get(var_name)?;
                for i in 0..n_rows {
                    z_mat[[i, j]] = col_data[i];
                }
            }

            z_mat
        };

        Self::fit(&y, &x, &z, cov_type)
    }

    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        z: &Array2<f64>,
        cov_type: CovarianceType,
    ) -> Result<IvResult, GreenersError> {
        let n = x.nrows();
        let k = x.ncols();
        let l = z.ncols();

        if y.len() != n || z.nrows() != n {
            return Err(GreenersError::ShapeMismatch("Row count mismatch".into()));
        }
        if l < k {
            return Err(GreenersError::ShapeMismatch(format!(
                "Order Condition Failed: Not enough instruments. Z has {} cols, X has {} cols.",
                l, k
            )));
        }

        // --- STAGE 1: Regress X on Z to get X_hat ---
        let z_t = z.t();
        let zt_z = z_t.dot(z);
        let zt_z_inv = zt_z.inv()?;

        let zt_x = z_t.dot(x);
        let first_stage_coeffs = zt_z_inv.dot(&zt_x);
        let x_hat = z.dot(&first_stage_coeffs);

        // --- STAGE 2: Regress y on X_hat ---
        let x_hat_t = x_hat.t();
        let xht_xh = x_hat_t.dot(&x_hat);
        let xht_xh_inv = xht_xh.inv()?;

        let xht_y = x_hat_t.dot(y);
        let beta = xht_xh_inv.dot(&xht_y);

        // --- Residuals ---
        // Uses ORIGINAL X
        let predicted_original = x.dot(&beta);
        let residuals = y - &predicted_original;
        let ssr = residuals.dot(&residuals);

        let df_resid = n - k;
        let sigma2 = ssr / (df_resid as f64);
        let sigma = sigma2.sqrt();

        // --- Covariance Matrix ---
        // FIX 2: NeweyWest implementation in match
        let cov_matrix = match cov_type {
            CovarianceType::NonRobust => &xht_xh_inv * sigma2,
            CovarianceType::HC1 => {
                let u_squared = residuals.mapv(|r| r.powi(2));
                let mut xhat_weighted = x_hat.clone();

                for (i, mut row) in xhat_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_squared[i];
                }

                let meat = x_hat_t.dot(&xhat_weighted);
                let bread = &xht_xh_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                let correction = (n as f64) / (df_resid as f64);
                sandwich * correction
            }
            CovarianceType::NeweyWest(lags) => {
                // HAC Implementation for IV
                // We use X_hat in the "meat" calculation instead of X.

                // 1. Omega_0 (HC part)
                let u_squared = residuals.mapv(|r| r.powi(2));
                let mut xhat_weighted = x_hat.clone();
                for (i, mut row) in xhat_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_squared[i];
                }
                let mut meat = x_hat_t.dot(&xhat_weighted);

                // 2. Autocovariance terms
                for l in 1..=lags {
                    let weight = 1.0 - (l as f64) / ((lags + 1) as f64);
                    let mut omega_l = Array2::<f64>::zeros((k, k));

                    for t in l..n {
                        let scale = residuals[t] * residuals[t - l];
                        let row_t = x_hat.row(t);
                        let row_prev = x_hat.row(t - l);

                        for i in 0..k {
                            for j in 0..k {
                                omega_l[[i, j]] += scale * row_t[i] * row_prev[j];
                            }
                        }
                    }

                    let omega_l_t = omega_l.t();
                    let term = &omega_l + &omega_l_t;
                    meat = meat + (&term * weight);
                }

                let bread = &xht_xh_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                let correction = (n as f64) / (df_resid as f64);
                sandwich * correction
            }
        };

        let std_errors = cov_matrix.diag().mapv(f64::sqrt);
        let t_values = &beta / &std_errors;

        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|_| GreenersError::OptimizationFailed)?;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - t_dist.cdf(t.abs())));

        let y_mean = y.mean().unwrap_or(0.0);
        let sst = y.mapv(|val| (val - y_mean).powi(2)).sum();
        let r_squared = 1.0 - (ssr / sst);

        Ok(IvResult {
            params: beta,
            std_errors,
            t_values,
            p_values,
            r_squared,
            n_obs: n,
            df_resid,
            sigma,
            cov_type,
        })
    }
}
