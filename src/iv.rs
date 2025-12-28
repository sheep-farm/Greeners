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
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for IvResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIX 1: Added NeweyWest option in Display
        let cov_str = match &self.cov_type {
            CovarianceType::NonRobust => "Non-Robust".to_string(),
            CovarianceType::HC1 => "Robust (HC1)".to_string(),
            CovarianceType::HC2 => "Robust (HC2)".to_string(),
            CovarianceType::HC3 => "Robust (HC3)".to_string(),
            CovarianceType::HC4 => "Robust (HC4)".to_string(),
            CovarianceType::NeweyWest(lags) => format!("HAC (Newey-West, L={})", lags),
            CovarianceType::Clustered(clusters) => {
                let n_clusters = clusters
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                format!("Clustered ({} clusters)", n_clusters)
            }
            CovarianceType::ClusteredTwoWay(clusters1, clusters2) => {
                let n_clusters_1 = clusters1
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                let n_clusters_2 = clusters2
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                format!("Two-Way Clustered ({}×{})", n_clusters_1, n_clusters_2)
            }
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
            let var_name = if let Some(ref names) = self.variable_names {
                if i < names.len() {
                    names[i].clone()
                } else {
                    format!("x{}", i)
                }
            } else {
                format!("x{}", i)
            };

            writeln!(
                f,
                "{:<10} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                var_name, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

impl IvResult {
    /// Predict out-of-sample values using estimated parameters
    ///
    /// # Arguments
    /// * `x_new` - New design matrix (n_new × k)
    ///
    /// # Returns
    /// Predicted values for new observations
    ///
    /// # Examples
    ///
    /// # Examples
    ///
    /// ```rust
    /// use greeners::{IV, CovarianceType}; // Adicionado CovarianceType
    /// use ndarray::{Array1, Array2};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let y = Array1::from(vec![1.0, 2.0, 3.0]);
    /// # let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0])?;
    /// # let z = x.clone();
    /// let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust)?;
    ///
    /// let x_new = Array2::from_shape_vec((3, 2), vec![1.0, 5.0, 1.0, 6.0, 1.0, 7.0])?;
    /// let predictions = result.predict(&x_new);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(&self, x_new: &Array2<f64>) -> Array1<f64> {
        x_new.dot(&self.params)
    }

    /// Calculate fitted values for in-sample observations
    ///
    /// # Arguments
    /// * `x` - Original design matrix (n × k)
    ///
    /// # Returns
    /// Fitted values (predictions for training data)
    pub fn fitted_values(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.params)
    }

    /// Calculate residuals for given observations
    ///
    /// # Arguments
    /// * `y` - Actual values
    /// * `x` - Design matrix
    ///
    /// # Returns
    /// Residuals (y - ŷ)
    pub fn residuals(&self, y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
        let y_hat = x.dot(&self.params);
        y - &y_hat
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

        // Build variable names from endogenous formula
        let mut var_names = Vec::new();
        if endog_formula.intercept {
            var_names.push("const".to_string());
        }
        for var in &endog_formula.independents {
            var_names.push(var.clone());
        }

        Self::fit_with_names(&y, &x, &z, cov_type, Some(var_names))
    }

    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        z: &Array2<f64>,
        cov_type: CovarianceType,
    ) -> Result<IvResult, GreenersError> {
        Self::fit_with_names(y, x, z, cov_type, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        z: &Array2<f64>,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
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
            CovarianceType::HC2 => {
                // HC2 for IV: leverage-adjusted
                let mut leverage = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let xhat_i = x_hat.row(i);
                    let temp = xht_xh_inv.dot(&xhat_i);
                    leverage[i] = xhat_i.dot(&temp);
                }

                let mut u_adjusted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let h_i = leverage[i];
                    if h_i >= 0.9999 {
                        u_adjusted[i] = residuals[i].powi(2);
                    } else {
                        u_adjusted[i] = residuals[i].powi(2) / (1.0 - h_i);
                    }
                }

                let mut xhat_weighted = x_hat.clone();
                for (i, mut row) in xhat_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_adjusted[i];
                }

                let meat = x_hat_t.dot(&xhat_weighted);
                let bread = &xht_xh_inv;
                bread.dot(&meat).dot(bread)
            }
            CovarianceType::HC3 => {
                // HC3 for IV: jackknife (most robust)
                let mut leverage = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let xhat_i = x_hat.row(i);
                    let temp = xht_xh_inv.dot(&xhat_i);
                    leverage[i] = xhat_i.dot(&temp);
                }

                let mut u_adjusted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let h_i = leverage[i];
                    if h_i >= 0.9999 {
                        u_adjusted[i] = residuals[i].powi(2);
                    } else {
                        u_adjusted[i] = residuals[i].powi(2) / (1.0 - h_i).powi(2);
                    }
                }

                let mut xhat_weighted = x_hat.clone();
                for (i, mut row) in xhat_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_adjusted[i];
                }

                let meat = x_hat_t.dot(&xhat_weighted);
                let bread = &xht_xh_inv;
                bread.dot(&meat).dot(bread)
            }
            CovarianceType::HC4 => {
                // HC4 for IV: refined jackknife
                let mut leverage = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let xhat_i = x_hat.row(i);
                    let temp = xht_xh_inv.dot(&xhat_i);
                    leverage[i] = xhat_i.dot(&temp);
                }

                let mut u_adjusted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let h_i = leverage[i];
                    if h_i >= 0.9999 {
                        u_adjusted[i] = residuals[i].powi(2);
                    } else {
                        let delta_i = 4.0_f64.min((n as f64) * h_i / (k as f64));
                        u_adjusted[i] = residuals[i].powi(2) / (1.0 - h_i).powf(delta_i);
                    }
                }

                let mut xhat_weighted = x_hat.clone();
                for (i, mut row) in xhat_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_adjusted[i];
                }

                let meat = x_hat_t.dot(&xhat_weighted);
                let bread = &xht_xh_inv;
                bread.dot(&meat).dot(bread)
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
            CovarianceType::Clustered(ref cluster_ids) => {
                // Clustered Standard Errors for IV
                // Same logic as OLS but using X_hat instead of X

                if cluster_ids.len() != n {
                    return Err(GreenersError::ShapeMismatch(format!(
                        "Cluster IDs length ({}) must match number of observations ({})",
                        cluster_ids.len(),
                        n
                    )));
                }

                use std::collections::HashMap;
                let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
                for (obs_idx, &cluster_id) in cluster_ids.iter().enumerate() {
                    clusters.entry(cluster_id).or_default().push(obs_idx);
                }

                let n_clusters = clusters.len();
                let mut meat = Array2::<f64>::zeros((k, k));

                for (_cluster_id, obs_indices) in clusters.iter() {
                    let cluster_size = obs_indices.len();
                    let mut xhat_g = Array2::<f64>::zeros((cluster_size, k));
                    let mut u_g = Array1::<f64>::zeros(cluster_size);

                    for (i, &obs_idx) in obs_indices.iter().enumerate() {
                        xhat_g.row_mut(i).assign(&x_hat.row(obs_idx));
                        u_g[i] = residuals[obs_idx];
                    }

                    for i in 0..cluster_size {
                        for j in 0..cluster_size {
                            let scale = u_g[i] * u_g[j];
                            let x_i = xhat_g.row(i);
                            let x_j = xhat_g.row(j);

                            for p in 0..k {
                                for q in 0..k {
                                    meat[[p, q]] += scale * x_i[p] * x_j[q];
                                }
                            }
                        }
                    }
                }

                let bread = &xht_xh_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                let g_correction = (n_clusters as f64) / ((n_clusters - 1) as f64);
                let df_correction = ((n - 1) as f64) / (df_resid as f64);
                sandwich * g_correction * df_correction
            }
            CovarianceType::ClusteredTwoWay(ref cluster_ids_1, ref cluster_ids_2) => {
                // Two-Way Clustered Standard Errors for IV (Cameron-Gelbach-Miller, 2011)
                // Uses X_hat instead of X (same as one-way clustering for IV)

                if cluster_ids_1.len() != n || cluster_ids_2.len() != n {
                    return Err(GreenersError::ShapeMismatch(format!(
                        "Both cluster ID vectors must match number of observations ({})",
                        n
                    )));
                }

                let compute_clustered_meat = |cluster_ids: &[usize]| -> Array2<f64> {
                    use std::collections::HashMap;
                    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
                    for (obs_idx, &cluster_id) in cluster_ids.iter().enumerate() {
                        clusters.entry(cluster_id).or_default().push(obs_idx);
                    }

                    let mut meat = Array2::<f64>::zeros((k, k));

                    for (_cluster_id, obs_indices) in clusters.iter() {
                        let cluster_size = obs_indices.len();
                        let mut xhat_g = Array2::<f64>::zeros((cluster_size, k));
                        let mut u_g = Array1::<f64>::zeros(cluster_size);

                        for (i, &obs_idx) in obs_indices.iter().enumerate() {
                            xhat_g.row_mut(i).assign(&x_hat.row(obs_idx));
                            u_g[i] = residuals[obs_idx];
                        }

                        for i in 0..cluster_size {
                            for j in 0..cluster_size {
                                let scale = u_g[i] * u_g[j];
                                let x_i = xhat_g.row(i);
                                let x_j = xhat_g.row(j);

                                for p in 0..k {
                                    for q in 0..k {
                                        meat[[p, q]] += scale * x_i[p] * x_j[q];
                                    }
                                }
                            }
                        }
                    }

                    meat
                };

                let meat_1 = compute_clustered_meat(cluster_ids_1);
                let meat_2 = compute_clustered_meat(cluster_ids_2);

                let max_cluster2 = cluster_ids_2.iter().max().unwrap_or(&0) + 1;
                let intersection_ids: Vec<usize> = cluster_ids_1
                    .iter()
                    .zip(cluster_ids_2.iter())
                    .map(|(&c1, &c2)| c1 * max_cluster2 + c2)
                    .collect();

                let meat_12 = compute_clustered_meat(&intersection_ids);

                let meat = &meat_1 + &meat_2 - &meat_12;

                let bread = &xht_xh_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                use std::collections::HashSet;
                let n_clusters_1: HashSet<_> = cluster_ids_1.iter().collect();
                let n_clusters_2: HashSet<_> = cluster_ids_2.iter().collect();
                let g = n_clusters_1.len().min(n_clusters_2.len());

                let g_correction = (g as f64) / ((g - 1) as f64);
                let df_correction = ((n - 1) as f64) / (df_resid as f64);
                sandwich * g_correction * df_correction
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
            variable_names,
        })
    }
}
