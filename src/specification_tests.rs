use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

/// Specification tests for regression models
pub struct SpecificationTests;

impl SpecificationTests {
    /// White's Test for Heteroskedasticity
    ///
    /// Tests H₀: Homoskedasticity (constant variance) vs H₁: Heteroskedasticity
    ///
    /// # Arguments
    /// * `residuals` - Residuals from OLS regression
    /// * `x` - Design matrix (n × k)
    ///
    /// # Returns
    /// Tuple of (LM_statistic, p_value, degrees_of_freedom)
    ///
    /// # Interpretation
    /// - If p < 0.05: Reject H₀, heteroskedasticity is present (use robust SE)
    /// - If p > 0.05: Fail to reject H₀, homoskedasticity is plausible
    ///
    /// # Example
    /// ```no_run
    /// use greeners::SpecificationTests;
    ///
    /// let (lm_stat, p_value, df) = SpecificationTests::white_test(&residuals, &x)?;
    /// if p_value < 0.05 {
    ///     println!("Heteroskedasticity detected! Use robust standard errors.");
    /// }
    /// ```
    pub fn white_test(
        residuals: &Array1<f64>,
        x: &Array2<f64>,
    ) -> Result<(f64, f64, usize), String> {
        let n = residuals.len();
        let k = x.ncols();

        // Square residuals (dependent variable for auxiliary regression)
        let u_squared = residuals.mapv(|r| r.powi(2));

        // Create auxiliary regressors: x and x² (simplified to avoid singularity)
        // Exclude constant term (first column) to avoid perfect multicollinearity
        let mut aux_regressors = Vec::new();

        // Add constant term
        aux_regressors.push(x.column(0).to_owned());

        // Add non-constant regressors and their squares (skip first column = constant)
        for j in 1..k {
            aux_regressors.push(x.column(j).to_owned());
        }

        // Add squared terms for non-constant regressors
        for j in 1..k {
            let x_j = x.column(j);
            aux_regressors.push(x_j.mapv(|v| v.powi(2)));
        }

        let p = aux_regressors.len(); // Total number of auxiliary regressors

        // Build auxiliary design matrix
        let mut x_aux = Array2::<f64>::zeros((n, p));
        for (j, regressor) in aux_regressors.iter().enumerate() {
            x_aux.column_mut(j).assign(regressor);
        }

        // Auxiliary regression: u² = X_aux * β + error
        // Calculate R² from this auxiliary regression
        let x_t = x_aux.t();
        let xtx = x_t.dot(&x_aux);
        let xtx_inv: Array2<f64> = match xtx.inv() {
            Ok(inv) => inv,
            Err(_) => return Err("Singular matrix in White test auxiliary regression".to_string()),
        };

        let xty = x_t.dot(&u_squared);
        let beta_aux: Array1<f64> = xtx_inv.dot(&xty);
        let fitted: Array1<f64> = x_aux.dot(&beta_aux);

        // Calculate R² for auxiliary regression
        let mean_u_sq = u_squared.mean().unwrap_or(0.0);
        let tss = u_squared
            .iter()
            .map(|&y| (y - mean_u_sq).powi(2))
            .sum::<f64>();
        let rss = fitted
            .iter()
            .zip(u_squared.iter())
            .map(|(&f, &y)| (y - f).powi(2))
            .sum::<f64>();
        let r_squared = 1.0 - rss / tss;

        // White's LM statistic: n * R²
        let lm_stat = (n as f64) * r_squared;

        // Degrees of freedom = number of auxiliary regressors (excluding constant if present)
        let df = p;

        // Under H₀, LM ~ χ²(df)
        let chi2_dist = ChiSquared::new(df as f64).map_err(|e| e.to_string())?;
        let p_value = 1.0 - chi2_dist.cdf(lm_stat);

        Ok((lm_stat, p_value, df))
    }

    /// Ramsey RESET Test for Functional Form Misspecification
    ///
    /// Tests H₀: Model is correctly specified vs H₁: Functional form misspecification
    ///
    /// # Arguments
    /// * `y` - Dependent variable
    /// * `x` - Design matrix (n × k)
    /// * `fitted_values` - Fitted values from original regression
    /// * `power` - Maximum power of fitted values to include (typically 2, 3, or 4)
    ///
    /// # Returns
    /// Tuple of (F_statistic, p_value, df_num, df_denom)
    ///
    /// # Interpretation
    /// - If p < 0.05: Reject H₀, functional form misspecification detected
    /// - If p > 0.05: Fail to reject H₀, functional form appears adequate
    ///
    /// # Example
    /// ```no_run
    /// use greeners::SpecificationTests;
    ///
    /// let fitted = model.fitted_values(&x);
    /// let (f_stat, p_value, _, _) = SpecificationTests::reset_test(&y, &x, &fitted, 3)?;
    /// if p_value < 0.05 {
    ///     println!("Functional form misspecification detected!");
    /// }
    /// ```
    pub fn reset_test(
        y: &Array1<f64>,
        x: &Array2<f64>,
        fitted_values: &Array1<f64>,
        power: usize,
    ) -> Result<(f64, f64, usize, usize), String> {
        if power < 2 {
            return Err("Power must be at least 2 for RESET test".to_string());
        }

        let n = y.len();
        let _k = x.ncols();

        // Original model SSR
        let residuals_orig: Array1<f64> = y - fitted_values;
        let ssr_orig = residuals_orig.dot(&residuals_orig);

        // Augmented model: add ŷ², ŷ³, ..., ŷ^power
        let mut x_augmented = x.to_owned();
        for p in 2..=power {
            let y_hat_p = fitted_values.mapv(|v| v.powi(p as i32));
            let n_rows = x_augmented.nrows();
            let n_cols = x_augmented.ncols();
            let mut new_x = Array2::<f64>::zeros((n_rows, n_cols + 1));
            new_x
                .slice_mut(ndarray::s![.., 0..n_cols])
                .assign(&x_augmented);
            new_x.column_mut(n_cols).assign(&y_hat_p);
            x_augmented = new_x;
        }

        // Estimate augmented model
        let x_aug_t = x_augmented.t();
        let xtx_aug = x_aug_t.dot(&x_augmented);
        let xtx_aug_inv: Array2<f64> = match xtx_aug.inv() {
            Ok(inv) => inv,
            Err(_) => return Err("Singular matrix in RESET test".to_string()),
        };

        let xty_aug = x_aug_t.dot(y);
        let beta_aug: Array1<f64> = xtx_aug_inv.dot(&xty_aug);
        let fitted_aug: Array1<f64> = x_augmented.dot(&beta_aug);
        let residuals_aug: Array1<f64> = y - &fitted_aug;
        let ssr_aug = residuals_aug.dot(&residuals_aug);

        // F-statistic
        let q = power - 1; // Number of restrictions (added powers)
        let df_num = q;
        let df_denom = n - x_augmented.ncols();

        if df_denom <= 0 {
            return Err("Insufficient degrees of freedom for RESET test".to_string());
        }

        let f_stat = ((ssr_orig - ssr_aug) / df_num as f64) / (ssr_aug / df_denom as f64);

        // P-value
        let f_dist =
            FisherSnedecor::new(df_num as f64, df_denom as f64).map_err(|e| e.to_string())?;
        let p_value = 1.0 - f_dist.cdf(f_stat);

        Ok((f_stat, p_value, df_num, df_denom))
    }

    /// Breusch-Godfrey Test for Autocorrelation
    ///
    /// Tests H₀: No autocorrelation up to lag p vs H₁: Autocorrelation present
    ///
    /// # Arguments
    /// * `residuals` - Residuals from OLS regression
    /// * `x` - Design matrix (n × k)
    /// * `lags` - Number of lags to test (typically 1 for AR(1))
    ///
    /// # Returns
    /// Tuple of (LM_statistic, p_value, degrees_of_freedom)
    ///
    /// # Interpretation
    /// - If p < 0.05: Reject H₀, autocorrelation detected
    /// - If p > 0.05: Fail to reject H₀, no evidence of autocorrelation
    ///
    /// # Example
    /// ```no_run
    /// use greeners::SpecificationTests;
    ///
    /// let (lm_stat, p_value, df) = SpecificationTests::breusch_godfrey_test(&residuals, &x, 1)?;
    /// if p_value < 0.05 {
    ///     println!("Autocorrelation detected! Consider using Newey-West SE.");
    /// }
    /// ```
    pub fn breusch_godfrey_test(
        residuals: &Array1<f64>,
        x: &Array2<f64>,
        lags: usize,
    ) -> Result<(f64, f64, usize), String> {
        let n = residuals.len();
        let _k = x.ncols();

        if lags >= n {
            return Err("Number of lags must be less than sample size".to_string());
        }

        // Create lagged residuals matrix
        let mut x_augmented = x.to_owned();
        for lag in 1..=lags {
            let mut lagged = Array1::<f64>::zeros(n);
            for i in lag..n {
                lagged[i] = residuals[i - lag];
            }

            // Append lagged residuals as new column
            let n_rows = x_augmented.nrows();
            let n_cols = x_augmented.ncols();
            let mut new_x = Array2::<f64>::zeros((n_rows, n_cols + 1));
            new_x
                .slice_mut(ndarray::s![.., 0..n_cols])
                .assign(&x_augmented);
            new_x.column_mut(n_cols).assign(&lagged);
            x_augmented = new_x;
        }

        // Drop first 'lags' observations to avoid zeros
        let x_aug_trim = x_augmented.slice(ndarray::s![lags.., ..]).to_owned();
        let u_trim = residuals.slice(ndarray::s![lags..]).to_owned();
        let n_trim = u_trim.len();

        // Auxiliary regression: u_t = X*β + γ₁*u_{t-1} + ... + γₚ*u_{t-p} + error
        let x_aug_t = x_aug_trim.t();
        let xtx_aug = x_aug_t.dot(&x_aug_trim);
        let xtx_aug_inv: Array2<f64> = match xtx_aug.inv() {
            Ok(inv) => inv,
            Err(_) => return Err("Singular matrix in Breusch-Godfrey test".to_string()),
        };

        let xty_aug = x_aug_t.dot(&u_trim);
        let beta_aug: Array1<f64> = xtx_aug_inv.dot(&xty_aug);
        let fitted_aug: Array1<f64> = x_aug_trim.dot(&beta_aug);

        // Calculate R² for auxiliary regression
        let mean_u = u_trim.mean().unwrap_or(0.0);
        let tss = u_trim.iter().map(|&u| (u - mean_u).powi(2)).sum::<f64>();
        let rss = fitted_aug
            .iter()
            .zip(u_trim.iter())
            .map(|(&f, &u)| (u - f).powi(2))
            .sum::<f64>();
        let r_squared = 1.0 - rss / tss;

        // LM statistic: n * R²
        let lm_stat = (n_trim as f64) * r_squared;

        // Degrees of freedom = number of lags
        let df = lags;

        // Under H₀, LM ~ χ²(lags)
        let chi2_dist = ChiSquared::new(df as f64).map_err(|e| e.to_string())?;
        let p_value = 1.0 - chi2_dist.cdf(lm_stat);

        Ok((lm_stat, p_value, df))
    }

    /// Goldfeld-Quandt Test for Heteroskedasticity
    ///
    /// Tests H₀: Homoskedasticity vs H₁: Variance increases with ordering variable
    ///
    /// # Arguments
    /// * `residuals` - Residuals from OLS regression (should be ordered by suspected variable)
    /// * `split_fraction` - Fraction of middle observations to drop (typically 0.2 to 0.33)
    ///
    /// # Returns
    /// Tuple of (F_statistic, p_value, df1, df2)
    ///
    /// # Interpretation
    /// - If p < 0.05: Reject H₀, heteroskedasticity detected
    /// - If p > 0.05: Fail to reject H₀, homoskedasticity plausible
    pub fn goldfeld_quandt_test(
        residuals: &Array1<f64>,
        split_fraction: f64,
    ) -> Result<(f64, f64, usize, usize), String> {
        let n = residuals.len();
        let drop_n = (n as f64 * split_fraction) as usize;
        let group_size = (n - drop_n) / 2;

        if group_size < 2 {
            return Err("Insufficient observations for Goldfeld-Quandt test".to_string());
        }

        // First group: observations 0 to group_size-1
        let group1 = residuals.slice(ndarray::s![0..group_size]);
        let ssr1: f64 = group1.iter().map(|&r| r.powi(2)).sum();

        // Second group: observations n-group_size to n-1
        let group2 = residuals.slice(ndarray::s![(n - group_size)..]);
        let ssr2: f64 = group2.iter().map(|&r| r.powi(2)).sum();

        // F-statistic: ratio of variances (larger / smaller)
        let f_stat = if ssr2 > ssr1 {
            ssr2 / ssr1
        } else {
            ssr1 / ssr2
        };

        let df1 = group_size;
        let df2 = group_size;

        // P-value (two-tailed)
        let f_dist = FisherSnedecor::new(df1 as f64, df2 as f64).map_err(|e| e.to_string())?;
        let p_value = 2.0 * (1.0 - f_dist.cdf(f_stat)).min(f_dist.cdf(f_stat));

        Ok((f_stat, p_value, df1, df2))
    }

    /// Pretty print specification test results
    pub fn print_test_result(
        test_name: &str,
        statistic: f64,
        p_value: f64,
        null_hypothesis: &str,
        alternative_hypothesis: &str,
    ) {
        println!("\n{:=^80}", format!(" {} ", test_name));
        println!("{:-^80}", "");
        println!("H₀: {}", null_hypothesis);
        println!("H₁: {}", alternative_hypothesis);
        println!("\nTest Statistic: {:.4}", statistic);
        println!("P-value: {:.6}", p_value);

        if p_value < 0.01 {
            println!("\n✅ REJECT H₀ at 1% level (p < 0.01)");
            println!("   → Strong evidence for H₁");
        } else if p_value < 0.05 {
            println!("\n✅ REJECT H₀ at 5% level (p < 0.05)");
            println!("   → Evidence for H₁");
        } else if p_value < 0.10 {
            println!("\n⚠️  MARGINALLY REJECT H₀ at 10% level (p < 0.10)");
            println!("   → Weak evidence for H₁");
        } else {
            println!("\n❌ FAIL TO REJECT H₀ (p > 0.10)");
            println!("   → No evidence against H₀");
        }
        println!("{:=^80}", "");
    }
}
