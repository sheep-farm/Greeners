use crate::error::GreenersError;
use crate::{CovarianceType, InferenceType};
use crate::{DataFrame, Formula};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, QR};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, Normal, StudentsT};
use std::fmt;

#[derive(Debug, Clone)]
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
    pub cov_type: CovarianceType,            // Store which type was used
    pub inference_type: InferenceType,       // Distribution for hypothesis testing
    pub variable_names: Option<Vec<String>>, // Names of variables (from Formula)
    pub omitted_vars: Vec<String>,           // Variables dropped due to collinearity
}

impl OlsResult {
    /// Generate predictions (fitted values) for new data
    ///
    /// # Arguments
    /// * `x_new` - Design matrix for new observations (must have same number of columns as original X)
    ///
    /// # Returns
    /// Array of predicted values
    ///
    /// # Example
    /// ```no_run
    /// use greeners::{OLS, CovarianceType};
    /// use ndarray::{Array1, Array2};
    ///
    /// let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let x = Array2::from_shape_vec((5, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
    /// let result = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    ///
    /// // Predict for new data
    /// let x_new = Array2::from_shape_vec((2, 2), vec![1.0, 6.0, 1.0, 7.0]).unwrap();
    /// let y_pred = result.predict(&x_new);
    /// ```
    pub fn predict(&self, x_new: &Array2<f64>) -> Array1<f64> {
        x_new.dot(&self.params)
    }

    /// Calculate residuals for given data
    ///
    /// # Arguments
    /// * `y` - Actual values
    /// * `x` - Design matrix
    ///
    /// # Returns
    /// Array of residuals (y - ŷ)
    pub fn residuals(&self, y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
        let y_hat = x.dot(&self.params);
        y - &y_hat
    }

    /// Get fitted values (in-sample predictions)
    ///
    /// # Arguments
    /// * `x` - Original design matrix used in fitting
    ///
    /// # Returns
    /// Array of fitted values
    pub fn fitted_values(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.params)
    }

    /// Model comparison statistics
    ///
    /// # Returns
    /// Tuple of (AIC, BIC, Log-Likelihood, Adjusted R²)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use greeners::{OLS, CovarianceType}; // Importe o enum
    /// use ndarray::{Array1, Array2};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let y = Array1::from(vec![1.0, 2.0, 3.0]);
    /// # let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0])?;
    /// // Adicione o argumento extra aqui:
    /// let result = OLS::fit(&y, &x, CovarianceType::NonRobust)?;
    ///
    /// let (aic, bic, loglik, adj_r2) = result.model_stats();
    /// # Ok(())
    /// # }
    /// ```
    pub fn model_stats(&self) -> (f64, f64, f64, f64) {
        (self.aic, self.bic, self.log_likelihood, self.adj_r_squared)
    }

    /// Calculate partial R² for subset of coefficients
    ///
    /// Measures the contribution of specific variables to model fit
    ///
    /// # Arguments
    /// * `indices` - Indices of coefficients to test (excluding intercept)
    /// * `y` - Dependent variable
    /// * `x` - Full design matrix
    ///
    /// # Returns
    /// Partial R² showing variance explained by specified variables
    ///
    /// # Note
    /// Partial R² = (SSR_restricted - SSR_full) / SSR_restricted
    pub fn partial_r_squared(&self, indices: &[usize], y: &Array1<f64>, x: &Array2<f64>) -> f64 {
        // Full model SSR (already fitted)
        let fitted_full = self.fitted_values(x);
        let resid_full = y - &fitted_full;
        let ssr_full = resid_full.dot(&resid_full);

        // Restricted model: drop specified variables
        let n = x.nrows();
        let k_full = x.ncols();
        let k_restricted = k_full - indices.len();

        if k_restricted == 0 {
            return self.r_squared; // All variables removed = compare to mean
        }

        // Build restricted design matrix (keep columns NOT in indices)
        let mut x_restricted = Array2::<f64>::zeros((n, k_restricted));
        let mut col_idx = 0;
        for j in 0..k_full {
            if !indices.contains(&j) {
                x_restricted.column_mut(col_idx).assign(&x.column(j));
                col_idx += 1;
            }
        }

        // Fit restricted model (simple OLS)
        use ndarray_linalg::Inverse;
        let xt_x = x_restricted.t().dot(&x_restricted);
        let xt_y = x_restricted.t().dot(y);

        if let Ok(xt_x_inv) = xt_x.inv() {
            let beta_restricted = xt_x_inv.dot(&xt_y);
            let fitted_restricted = x_restricted.dot(&beta_restricted);
            let resid_restricted = y - &fitted_restricted;
            let ssr_restricted = resid_restricted.dot(&resid_restricted);

            // Partial R²
            (ssr_restricted - ssr_full) / ssr_restricted
        } else {
            0.0 // Singular matrix
        }
    }

    /// Helper function to compute p-values and confidence intervals
    ///
    /// This function computes statistical inference quantities using either
    /// Student's t-distribution or standard Normal distribution.
    ///
    /// # Arguments
    /// * `t_values` - Test statistics (coefficients / standard errors)
    /// * `std_errors` - Standard errors of coefficient estimates
    /// * `params` - Coefficient estimates
    /// * `df_resid` - Residual degrees of freedom (only used for StudentT)
    /// * `inference_type` - Distribution type to use
    ///
    /// # Returns
    /// Tuple of (p_values, conf_lower, conf_upper)
    pub(crate) fn compute_inference(
        t_values: &Array1<f64>,
        std_errors: &Array1<f64>,
        params: &Array1<f64>,
        df_resid: usize,
        inference_type: &InferenceType,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), GreenersError> {
        let (p_values, critical_value) = match inference_type {
            InferenceType::StudentT => {
                let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
                    .map_err(|_| GreenersError::OptimizationFailed)?;
                let p_vals = t_values.mapv(|t| 2.0 * (1.0 - t_dist.cdf(t.abs())));
                (p_vals, t_dist.inverse_cdf(0.975))
            }
            InferenceType::Normal => {
                let normal_dist =
                    Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
                let p_vals = t_values.mapv(|t| 2.0 * (1.0 - normal_dist.cdf(t.abs())));
                (p_vals, normal_dist.inverse_cdf(0.975))
            }
        };

        let margin_error = std_errors * critical_value;
        let conf_lower = params - &margin_error;
        let conf_upper = params + &margin_error;

        Ok((p_values, conf_lower, conf_upper))
    }

    /// Change inference type and recompute p-values and confidence intervals
    ///
    /// This method allows you to switch between Student's t-distribution and
    /// Normal distribution for hypothesis testing after the model has been fitted.
    /// The coefficient estimates and standard errors remain unchanged.
    ///
    /// # Arguments
    /// * `inference_type` - New distribution type to use
    ///
    /// # Returns
    /// Modified OlsResult with updated p-values and confidence intervals
    ///
    /// # Example
    /// ```
    /// use greeners::{OLS, CovarianceType, InferenceType};
    /// use ndarray::{Array1, Array2};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let x = Array2::from_shape_vec((5, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0])?;
    ///
    /// // Fit with default (Student's t)
    /// let result = OLS::fit(&y, &x, CovarianceType::NonRobust)?;
    ///
    /// // Switch to Normal distribution for large sample asymptotics
    /// let result_z = result.with_inference(InferenceType::Normal)?;
    ///
    /// // Coefficients are identical, but p-values differ
    /// assert_eq!(result.params, result_z.params);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_inference(mut self, inference_type: InferenceType) -> Result<Self, GreenersError> {
        let (p_values, conf_lower, conf_upper) = Self::compute_inference(
            &self.t_values,
            &self.std_errors,
            &self.params,
            self.df_resid,
            &inference_type,
        )?;

        self.p_values = p_values;
        self.conf_lower = conf_lower;
        self.conf_upper = conf_upper;
        self.inference_type = inference_type;

        Ok(self)
    }
}

impl fmt::Display for OlsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stat_label = match self.inference_type {
            InferenceType::StudentT => "t",
            InferenceType::Normal => "z",
        };

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
            "Variable",
            "coef",
            "std err",
            stat_label,
            format!("P>|{}|", stat_label),
            "[0.025      0.975]"
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
                "{:<10} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3} | {:>8.4}  {:>8.4}",
                var_name,
                self.params[i],
                self.std_errors[i],
                self.t_values[i],
                self.p_values[i],
                self.conf_lower[i],
                self.conf_upper[i]
            )?;
        }

        // Show omitted variables if any
        if !self.omitted_vars.is_empty() {
            writeln!(f, "{:-^78}", "")?;
            writeln!(f, "Omitted due to collinearity:")?;
            for var in &self.omitted_vars {
                writeln!(f, "  o.{}", var)?;
            }
        }

        writeln!(f, "{:=^78}", "")
    }
}

pub struct OLS;

impl OLS {
    /// Fits an OLS model using a formula and DataFrame.
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::{OLS, DataFrame, Formula, CovarianceType};
    /// use ndarray::Array1;
    /// use std::collections::HashMap;
    ///
    /// let mut data = HashMap::new();
    /// data.insert("y".to_string(), Array1::from(vec![1.0, 2.1, 3.2, 3.9, 5.1]));
    /// data.insert("x1".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    /// data.insert("x2".to_string(), Array1::from(vec![2.0, 2.5, 3.0, 3.5, 4.0]));
    ///
    /// let df = DataFrame::new(data).unwrap();
    /// let formula = Formula::parse("y ~ x1 + x2").unwrap();
    ///
    /// let result = OLS::from_formula(&formula, &df, CovarianceType::HC1).unwrap();
    /// println!("R-squared: {}", result.r_squared);
    /// ```
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        cov_type: CovarianceType,
    ) -> Result<OlsResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;

        // Build variable names from formula
        let mut var_names = Vec::new();
        if formula.intercept {
            var_names.push("const".to_string());
        }
        for var in &formula.independents {
            var_names.push(var.clone());
        }

        Self::fit_with_names(&y, &x, cov_type, Some(var_names))
    }

    /// Detect and remove perfectly collinear columns using QR decomposition.
    ///
    /// Returns: (clean_x, keep_indices, omitted_indices)
    pub fn detect_collinearity(
        x: &Array2<f64>,
        tolerance: f64,
    ) -> (Array2<f64>, Vec<usize>, Vec<usize>) {
        let n = x.nrows();
        let k = x.ncols();

        // Use QR decomposition to detect rank deficiency
        // Columns with small R diagonal values are linearly dependent
        match x.qr() {
            Ok((_, r)) => {
                let mut keep_indices = Vec::new();
                let mut omit_indices = Vec::new();

                // Check diagonal of R matrix
                for i in 0..k.min(n) {
                    let r_ii = r[[i, i]].abs();
                    if r_ii > tolerance {
                        keep_indices.push(i);
                    } else {
                        omit_indices.push(i);
                    }
                }

                // If all columns kept, return original matrix
                if omit_indices.is_empty() {
                    return (x.clone(), keep_indices, omit_indices);
                }

                // Build reduced matrix with only independent columns
                let x_clean = x.select(ndarray::Axis(1), &keep_indices);
                (x_clean, keep_indices, omit_indices)
            }
            Err(_) => {
                // QR failed, return original (will likely fail in OLS too)
                let keep: Vec<usize> = (0..k).collect();
                (x.clone(), keep, vec![])
            }
        }
    }

    /// Fits the model. Now accepts `cov_type` and optional variable names.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
    ) -> Result<OlsResult, GreenersError> {
        Self::fit_with_names(y, x, cov_type, None)
    }

    /// Fits the model with custom variable names.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
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

        // Detect and remove collinear columns ONLY when we have variable names
        // (i.e., formula-based usage). For internal algorithmic use (variable_names = None),
        // skip collinearity detection to preserve matrix dimensions.
        let (x_to_use, k_clean, omitted_var_names, clean_var_names) = if variable_names.is_some() {
            let tolerance = 1e-10;
            let (x_clean, keep_indices, omit_indices) = Self::detect_collinearity(x, tolerance);

            let mut omitted_names = Vec::new();
            let mut clean_names = Vec::new();

            if let Some(ref names) = variable_names {
                for &idx in &omit_indices {
                    if idx < names.len() {
                        omitted_names.push(names[idx].clone());
                    }
                }
                for &idx in &keep_indices {
                    if idx < names.len() {
                        clean_names.push(names[idx].clone());
                    }
                }
            }

            let k_clean = x_clean.ncols();
            if n <= k_clean {
                return Err(GreenersError::ShapeMismatch(
                    "Degrees of freedom <= 0 after removing collinear variables".into(),
                ));
            }

            (x_clean, k_clean, omitted_names, clean_names)
        } else {
            // No collinearity detection for internal use
            (x.clone(), k, Vec::new(), Vec::new())
        };

        let x_to_use = &x_to_use;

        // 1. Beta Estimation (Same for Robust and Non-Robust)
        let x_t = x_to_use.t();
        let xt_x = x_t.dot(x_to_use);
        let xt_x_inv = xt_x.inv()?;
        let xt_y = x_t.dot(y);
        let beta = xt_x_inv.dot(&xt_y);

        // 2. Residuals
        let predicted = x_to_use.dot(&beta);
        let residuals = y - &predicted;
        let ssr = residuals.dot(&residuals);

        let df_resid = n - k_clean;
        let df_model = k_clean - 1;

        let sigma2 = ssr / (df_resid as f64);
        let sigma = sigma2.sqrt();

        // src/ols.rs (inside OLS::fit, replace the 'match cov_type' block)

        // 3. Covariance Matrix Selection
        let cov_matrix = match &cov_type {
            CovarianceType::NonRobust => &xt_x_inv * sigma2,
            CovarianceType::HC1 => {
                // HC1: White's heteroscedasticity-robust SE with small-sample correction
                // V = (X'X)^-1 * X' diag(u²) X * (X'X)^-1 * (n / (n-k))
                let u_squared = residuals.mapv(|r| r.powi(2));
                let mut x_weighted = x_to_use.clone();
                for (i, mut row) in x_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_squared[i];
                }
                let meat = x_t.dot(&x_weighted);
                let bread = &xt_x_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                let correction = (n as f64) / (df_resid as f64);
                sandwich * correction
            }
            CovarianceType::HC2 => {
                // HC2: Leverage-adjusted heteroscedasticity-robust SE
                // V = (X'X)^-1 * X' diag(u² / (1 - h_i)) X * (X'X)^-1
                // More efficient than HC1 with small samples

                // Calculate leverage values: h_i = x_i' (X'X)^-1 x_i
                let mut leverage = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let x_i = x_to_use.row(i);
                    let temp = xt_x_inv.dot(&x_i);
                    leverage[i] = x_i.dot(&temp);
                }

                // Adjust residuals: u²_i / (1 - h_i)
                let mut u_adjusted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let h_i = leverage[i];
                    if h_i >= 0.9999 {
                        u_adjusted[i] = residuals[i].powi(2); // Avoid division by zero
                    } else {
                        u_adjusted[i] = residuals[i].powi(2) / (1.0 - h_i);
                    }
                }

                // Build sandwich estimator with adjusted weights
                let mut x_weighted = x_to_use.clone();
                for (i, mut row) in x_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_adjusted[i];
                }

                let meat = x_t.dot(&x_weighted);
                let bread = &xt_x_inv;
                bread.dot(&meat).dot(bread)
            }
            CovarianceType::HC3 => {
                // HC3: Jackknife heteroscedasticity-robust SE
                // V = (X'X)^-1 * X' diag(u² / (1 - h_i)²) X * (X'X)^-1
                // Most robust for small samples - recommended default

                // Calculate leverage values
                let mut leverage = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let x_i = x_to_use.row(i);
                    let temp = xt_x_inv.dot(&x_i);
                    leverage[i] = x_i.dot(&temp);
                }

                // Adjust residuals: u²_i / (1 - h_i)²
                let mut u_adjusted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let h_i = leverage[i];
                    if h_i >= 0.9999 {
                        u_adjusted[i] = residuals[i].powi(2); // Avoid division by zero
                    } else {
                        u_adjusted[i] = residuals[i].powi(2) / (1.0 - h_i).powi(2);
                    }
                }

                // Build sandwich estimator with adjusted weights
                let mut x_weighted = x_to_use.clone();
                for (i, mut row) in x_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_adjusted[i];
                }

                let meat = x_t.dot(&x_weighted);
                let bread = &xt_x_inv;
                bread.dot(&meat).dot(bread)
            }
            CovarianceType::HC4 => {
                // HC4: Refined jackknife (Cribari-Neto, 2004)
                // V = (X'X)^-1 * X' diag(u² / (1 - h_i)^δᵢ) X * (X'X)^-1
                // where δᵢ = min(4, n * h_i / k)
                // Best performance with influential observations

                // Calculate leverage values
                let mut leverage = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let x_i = x_to_use.row(i);
                    let temp = xt_x_inv.dot(&x_i);
                    leverage[i] = x_i.dot(&temp);
                }

                // Adjust residuals with power δᵢ
                let mut u_adjusted = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let h_i = leverage[i];
                    if h_i >= 0.9999 {
                        u_adjusted[i] = residuals[i].powi(2);
                    } else {
                        // δᵢ = min(4, n * h_i / k)
                        let delta_i = 4.0_f64.min((n as f64) * h_i / (k_clean as f64));
                        u_adjusted[i] = residuals[i].powi(2) / (1.0 - h_i).powf(delta_i);
                    }
                }

                // Build sandwich estimator
                let mut x_weighted = x_to_use.clone();
                for (i, mut row) in x_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_adjusted[i];
                }

                let meat = x_t.dot(&x_weighted);
                let bread = &xt_x_inv;
                bread.dot(&meat).dot(bread)
            }
            CovarianceType::NeweyWest(lags) => {
                // HAC Estimator (Newey-West)
                // Formula: (X'X)^-1 * [ Omega_0 + sum(w_l * (Omega_l + Omega_l')) ] * (X'X)^-1

                // 1. Calculate Omega_0 (Same as White's Matrix "Meat")
                let u_squared = residuals.mapv(|r| r.powi(2));
                let mut x_weighted = x_to_use.clone();
                for (i, mut row) in x_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
                    row *= u_squared[i];
                }
                let mut meat = x_t.dot(&x_weighted); // This is Omega_0

                // 2. Add Autocovariance terms (Omega_l)
                // Bartlett Kernel weights: w(l) = 1 - l / (L + 1)
                for l in 1..=*lags {
                    let weight = 1.0 - (l as f64) / ((*lags + 1) as f64);

                    // Calculate Omega_l = sum( u_t * u_{t-l} * x_t * x_{t-l}' )
                    // Since specific lag logic is tricky in pure matrix algebra without huge memory,
                    // we iterate carefully.

                    let mut omega_l = Array2::<f64>::zeros((k_clean, k_clean));

                    // Sum over t where lag exists (from l to n)
                    for t in l..n {
                        let u_t = residuals[t];
                        let u_prev = residuals[t - l];

                        let x_row_t = x_to_use.row(t);
                        let x_row_prev = x_to_use.row(t - l);

                        // Outer product: (x_t * x_{t-l}') scaled by (u_t * u_{t-l})
                        // Using 'scaled_add' is efficient: matrix += alpha * (vec * vec.t)
                        // But ndarray doesn't have concise outer product add, so we do:
                        // term = (u_t * u_prev) * (x_t outer x_{t-l})

                        let scale = u_t * u_prev;

                        // Manual outer product addition for performance
                        for i in 0..k_clean {
                            for j in 0..k_clean {
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
            CovarianceType::Clustered(ref cluster_ids) => {
                // Clustered Standard Errors
                // Formula: V_cluster = (X'X)^-1 * [Σ_g (X_g' u_g u_g' X_g)] * (X'X)^-1
                // Critical for panel data, experiments, and grouped observations

                // Validate cluster IDs length
                if cluster_ids.len() != n {
                    return Err(GreenersError::ShapeMismatch(format!(
                        "Cluster IDs length ({}) must match number of observations ({})",
                        cluster_ids.len(),
                        n
                    )));
                }

                // Group observations by cluster
                use std::collections::HashMap;
                let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
                for (obs_idx, &cluster_id) in cluster_ids.iter().enumerate() {
                    clusters.entry(cluster_id).or_default().push(obs_idx);
                }

                let n_clusters = clusters.len();

                // Initialize meat matrix (middle part of sandwich)
                let mut meat = Array2::<f64>::zeros((k_clean, k_clean));

                // For each cluster g: calculate X_g' u_g u_g' X_g
                for (_cluster_id, obs_indices) in clusters.iter() {
                    let cluster_size = obs_indices.len();

                    // Extract X_g and u_g for this cluster
                    let mut x_g = Array2::<f64>::zeros((cluster_size, k_clean));
                    let mut u_g = Array1::<f64>::zeros(cluster_size);

                    for (i, &obs_idx) in obs_indices.iter().enumerate() {
                        x_g.row_mut(i).assign(&x_to_use.row(obs_idx));
                        u_g[i] = residuals[obs_idx];
                    }

                    // Calculate u_g * u_g' (outer product of residuals within cluster)
                    // Then X_g' * (u_g * u_g') * X_g
                    // More explicitly: Σ_i Σ_j (u_gi * u_gj * x_gi * x_gj')

                    for i in 0..cluster_size {
                        for j in 0..cluster_size {
                            let scale = u_g[i] * u_g[j];
                            let x_i = x_g.row(i);
                            let x_j = x_g.row(j);

                            // Add outer product: scale * (x_i ⊗ x_j)
                            for p in 0..k_clean {
                                for q in 0..k_clean {
                                    meat[[p, q]] += scale * x_i[p] * x_j[q];
                                }
                            }
                        }
                    }
                }

                // Apply sandwich formula
                let bread = &xt_x_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                // Small sample correction: (G / (G-1)) * ((N-1) / (N-K))
                // where G = number of clusters, N = observations, K = parameters
                let g_correction = (n_clusters as f64) / ((n_clusters - 1) as f64);
                let df_correction = ((n - 1) as f64) / (df_resid as f64);
                sandwich * g_correction * df_correction
            }
            CovarianceType::ClusteredTwoWay(ref cluster_ids_1, ref cluster_ids_2) => {
                // Two-Way Clustered Standard Errors (Cameron-Gelbach-Miller, 2011)
                // Formula: V = V₁ + V₂ - V₁₂
                // Where:
                //   V₁ = one-way clustering by dimension 1 (e.g., firm)
                //   V₂ = one-way clustering by dimension 2 (e.g., time)
                //   V₁₂ = clustering by intersection (firm × time pairs)
                //
                // This accounts for correlation both within dimension 1,
                // within dimension 2, and avoids double-counting the intersection.

                // Validate inputs
                if cluster_ids_1.len() != n || cluster_ids_2.len() != n {
                    return Err(GreenersError::ShapeMismatch(format!(
                        "Both cluster ID vectors must match number of observations ({})",
                        n
                    )));
                }

                // Helper function to compute clustered meat matrix
                let compute_clustered_meat = |cluster_ids: &[usize]| -> Array2<f64> {
                    use std::collections::HashMap;
                    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
                    for (obs_idx, &cluster_id) in cluster_ids.iter().enumerate() {
                        clusters.entry(cluster_id).or_default().push(obs_idx);
                    }

                    let mut meat = Array2::<f64>::zeros((k_clean, k_clean));

                    for (_cluster_id, obs_indices) in clusters.iter() {
                        let cluster_size = obs_indices.len();
                        let mut x_g = Array2::<f64>::zeros((cluster_size, k_clean));
                        let mut u_g = Array1::<f64>::zeros(cluster_size);

                        for (i, &obs_idx) in obs_indices.iter().enumerate() {
                            x_g.row_mut(i).assign(&x_to_use.row(obs_idx));
                            u_g[i] = residuals[obs_idx];
                        }

                        for i in 0..cluster_size {
                            for j in 0..cluster_size {
                                let scale = u_g[i] * u_g[j];
                                let x_i = x_g.row(i);
                                let x_j = x_g.row(j);

                                for p in 0..k_clean {
                                    for q in 0..k_clean {
                                        meat[[p, q]] += scale * x_i[p] * x_j[q];
                                    }
                                }
                            }
                        }
                    }

                    meat
                };

                // 1. Compute V₁ (cluster by dimension 1)
                let meat_1 = compute_clustered_meat(cluster_ids_1);

                // 2. Compute V₂ (cluster by dimension 2)
                let meat_2 = compute_clustered_meat(cluster_ids_2);

                // 3. Compute V₁₂ (cluster by intersection)
                // Create unique pair IDs: pair_id = cluster1_id * max_cluster2 + cluster2_id
                let max_cluster2 = cluster_ids_2.iter().max().unwrap_or(&0) + 1;
                let intersection_ids: Vec<usize> = cluster_ids_1
                    .iter()
                    .zip(cluster_ids_2.iter())
                    .map(|(&c1, &c2)| c1 * max_cluster2 + c2)
                    .collect();

                let meat_12 = compute_clustered_meat(&intersection_ids);

                // 4. Apply Cameron-Gelbach-Miller formula: V = V₁ + V₂ - V₁₂
                let meat = &meat_1 + &meat_2 - &meat_12;

                // Apply sandwich formula
                let bread = &xt_x_inv;
                let sandwich = bread.dot(&meat).dot(bread);

                // Small sample correction
                // Use minimum number of clusters for conservative inference
                use std::collections::HashSet;
                let n_clusters_1: HashSet<_> = cluster_ids_1.iter().collect();
                let n_clusters_2: HashSet<_> = cluster_ids_2.iter().collect();
                let g = n_clusters_1.len().min(n_clusters_2.len());

                let g_correction = (g as f64) / ((g - 1) as f64);
                let df_correction = ((n - 1) as f64) / (df_resid as f64);
                sandwich * g_correction * df_correction
            }
        };

        // 4. Standard Errors & Inference
        let std_errors = cov_matrix.diag().mapv(f64::sqrt);
        let t_values = &beta / &std_errors;

        // Use default inference type (StudentT)
        let default_inference = InferenceType::default();
        let (p_values, conf_lower, conf_upper) = OlsResult::compute_inference(
            &t_values,
            &std_errors,
            &beta,
            df_resid,
            &default_inference,
        )?;

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
        let aic = 2.0 * (k_clean as f64) - 2.0 * log_likelihood;
        let bic = (k_clean as f64) * n_f64.ln() - 2.0 * log_likelihood;

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
            inference_type: InferenceType::default(),
            variable_names: if !clean_var_names.is_empty() {
                Some(clean_var_names)
            } else {
                variable_names
            },
            omitted_vars: omitted_var_names,
        })
    }
}

// Helper alias for simpler axis usage inside the function
use ndarray as nd;
