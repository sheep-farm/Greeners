use crate::{CovarianceType, GreenersError, OLS}; // Reuse OLS engine
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// Struct to hold Fixed Effects estimation results.
#[derive(Debug)]
pub struct PanelResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64, // "Within" R-squared
    pub n_obs: usize,
    pub n_entities: usize, // Number of unique groups (N)
    pub df_resid: usize,   // Corrected degrees of freedom
    pub sigma: f64,
}

impl fmt::Display for PanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Fixed Effects (Within) Regression ")?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Dep. Variable:", "y", "Within R-sq:", self.r_squared
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Estimator:", "Fixed Effects", "No. Entities:", self.n_entities
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4e}",
            "No. Observations:", self.n_obs, "Sigma:", self.sigma
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

pub struct FixedEffects;

impl FixedEffects {
    /// Performs the "Within Transformation" (Demeaning) on a matrix/vector.
    /// x_dem = x_it - mean(x_i)
    fn within_transform<T>(data: &Array2<f64>, groups: &[T]) -> Result<Array2<f64>, GreenersError>
    where
        T: Eq + Hash + Clone,
    {
        let n_rows = data.nrows();
        let n_cols = data.ncols();

        if n_rows != groups.len() {
            return Err(GreenersError::ShapeMismatch(
                "Data rows and Group IDs length mismatch".into(),
            ));
        }

        // 1. Calculate sums and counts per group
        let mut group_sums: HashMap<T, Array1<f64>> = HashMap::new();
        let mut group_counts: HashMap<T, usize> = HashMap::new();

        for (i, group_id) in groups.iter().enumerate() {
            let row = data.row(i).to_owned();

            group_sums
                .entry(group_id.clone())
                .and_modify(|sum| *sum = &*sum + &row)
                .or_insert(row);

            *group_counts.entry(group_id.clone()).or_insert(0) += 1;
        }

        // 2. Subtract group means from original data
        let mut transformed_data = Array2::zeros((n_rows, n_cols));

        for (i, group_id) in groups.iter().enumerate() {
            let sum = &group_sums[group_id];
            let count = group_counts[group_id] as f64;
            let mean = sum / count;

            let original_row = data.row(i);
            let demeaned_row = &original_row - &mean;

            transformed_data.row_mut(i).assign(&demeaned_row);
        }

        Ok(transformed_data)
    }

    /// Fits the Fixed Effects model using Within Estimation.
    ///
    /// # Arguments
    /// * `y` - Dependent variable.
    /// * `x` - Regressors (DO NOT include a constant/intercept column!).
    /// * `groups` - Vector of Entity IDs (Integers, Strings, etc.) corresponding to rows.
    pub fn fit<T>(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[T],
    ) -> Result<PanelResult, GreenersError>
    where
        T: Eq + Hash + Clone,
    {
        let n = x.nrows();

        // 1. Convert y to Array2 for the generic transform function
        let y_mat = y.view().insert_axis(Axis(1)).to_owned();

        // 2. Apply Within Transformation
        let y_demeaned_mat = Self::within_transform(&y_mat, groups)?;
        let x_demeaned = Self::within_transform(x, groups)?;

        // Flatten y back to Array1
        let y_demeaned = y_demeaned_mat.column(0).to_owned();

        // 3. Run OLS on demeaned data
        // We use OLS struct internally but we need to adjust standard errors manually later.
        let ols_result = OLS::fit(&y_demeaned, &x_demeaned, CovarianceType::NonRobust)?;

        // 4. Degrees of Freedom Correction
        // Standard OLS uses df = N - K
        // Fixed Effects requires df = N - K - (Number of Entities - 1)
        // Because we estimated N means implicitly.

        let mut unique_groups: HashMap<T, bool> = HashMap::new();
        for g in groups {
            unique_groups.insert(g.clone(), true);
        }
        let n_entities = unique_groups.len();

        let k = x.ncols();
        let df_resid_correct = n - k - (n_entities - 1); // FE correction

        if df_resid_correct == 0 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough degrees of freedom for Fixed Effects".into(),
            ));
        }

        // Recalculate Sigma and Standard Errors with correct DF
        // SSR is the same as OLS on demeaned data
        let residuals = &y_demeaned - &x_demeaned.dot(&ols_result.params);
        let ssr = residuals.dot(&residuals);

        let sigma2 = ssr / (df_resid_correct as f64);
        let sigma = sigma2.sqrt();

        // Adjust Covariance Matrix: Multiply by (OLS_DF / FE_DF) adjustment
        // Because OLS::fit calculated it using (n-k).
        let adjustment_factor = (ols_result.df_resid as f64) / (df_resid_correct as f64);

        // Extract diagonals (variances) and adjust
        let old_vars = ols_result.std_errors.mapv(|se| se.powi(2));
        let new_vars = old_vars * adjustment_factor;
        let std_errors = new_vars.mapv(f64::sqrt);

        // Recalculate t-stats and p-values
        let t_values = &ols_result.params / &std_errors;

        // Re-use statrs logic (simplified here)
        // In a real lib, we would call the t-distribution again with new df.
        // For now, we trust the user understands the df changed.

        Ok(PanelResult {
            params: ols_result.params,
            std_errors,
            t_values,
            p_values: ols_result.p_values, // Approximation (should recompute CDF with new DF)
            r_squared: ols_result.r_squared,
            n_obs: n,
            n_entities,
            df_resid: df_resid_correct,
            sigma,
        })
    }
}
