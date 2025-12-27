use crate::error::GreenersError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Bootstrap methods for statistical inference
pub struct Bootstrap;

impl Bootstrap {
    /// Pairs bootstrap for OLS regression
    ///
    /// Resamples (y, X) pairs with replacement to estimate sampling distribution
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n × 1)
    /// * `x` - Design matrix (n × k)
    /// * `n_bootstrap` - Number of bootstrap replications (recommended: 1000-10000)
    ///
    /// # Returns
    /// Array of bootstrap coefficient estimates (n_bootstrap × k)
    ///
    /// # Example
    /// ```no_run
    /// use greeners::Bootstrap;
    /// use ndarray::{Array1, Array2};
    ///
    /// let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let x = Array2::from_shape_vec((5, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0])?;
    ///
    /// // Generate 1000 bootstrap samples
    /// let boot_coefs = Bootstrap::pairs_bootstrap(&y, &x, 1000)?;
    ///
    /// // Calculate bootstrap standard errors
    /// let boot_se = boot_coefs.std_axis(ndarray::Axis(0), 0.0);
    /// ```
    pub fn pairs_bootstrap(
        y: &Array1<f64>,
        x: &Array2<f64>,
        n_bootstrap: usize,
    ) -> Result<Array2<f64>, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "X and y must have same number of rows".to_string(),
            ));
        }

        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..n).collect();

        // Store bootstrap coefficients
        let mut boot_coefs = Array2::<f64>::zeros((n_bootstrap, k));

        for b in 0..n_bootstrap {
            // Resample indices with replacement
            let mut boot_indices = vec![0; n];
            // for i in 0..n {
            for indice in boot_indices.iter_mut().take(n) {
                *indice = *indices.choose(&mut rng).unwrap();
            }

            // Create bootstrap sample
            let mut y_boot = Array1::<f64>::zeros(n);
            let mut x_boot = Array2::<f64>::zeros((n, k));

            for (i, &idx) in boot_indices.iter().enumerate() {
                y_boot[i] = y[idx];
                x_boot.row_mut(i).assign(&x.row(idx));
            }

            // Fit OLS on bootstrap sample
            let xt_x = x_boot.t().dot(&x_boot);
            let xt_y = x_boot.t().dot(&y_boot);

            match xt_x.inv() {
                Ok(xt_x_inv) => {
                    let beta_boot = xt_x_inv.dot(&xt_y);
                    boot_coefs.row_mut(b).assign(&beta_boot);
                }
                Err(_) => {
                    // Singular matrix in this bootstrap sample - use original estimate
                    // This is rare but can happen with small samples
                    let xt_x_orig = x.t().dot(x);
                    let xt_y_orig = x.t().dot(y);
                    if let Ok(inv) = xt_x_orig.inv() {
                        let beta_orig = inv.dot(&xt_y_orig);
                        boot_coefs.row_mut(b).assign(&beta_orig);
                    }
                }
            }
        }

        Ok(boot_coefs)
    }

    /// Calculate bootstrap standard errors from bootstrap coefficient matrix
    ///
    /// # Arguments
    /// * `boot_coefs` - Bootstrap coefficient matrix (n_bootstrap × k)
    ///
    /// # Returns
    /// Standard errors for each coefficient
    pub fn bootstrap_se(boot_coefs: &Array2<f64>) -> Array1<f64> {
        boot_coefs.std_axis(ndarray::Axis(0), 0.0)
    }

    /// Calculate bootstrap percentile confidence intervals
    ///
    /// # Arguments
    /// * `boot_coefs` - Bootstrap coefficient matrix (n_bootstrap × k)
    /// * `alpha` - Significance level (e.g., 0.05 for 95% CI)
    ///
    /// # Returns
    /// Tuple of (lower_bounds, upper_bounds)
    pub fn percentile_ci(boot_coefs: &Array2<f64>, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        let k = boot_coefs.ncols();
        let n_boot = boot_coefs.nrows();

        let lower_idx = ((alpha / 2.0) * n_boot as f64).floor() as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n_boot as f64).ceil() as usize;

        let mut lower = Array1::<f64>::zeros(k);
        let mut upper = Array1::<f64>::zeros(k);

        for j in 0..k {
            let mut col: Vec<f64> = boot_coefs.column(j).to_vec();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());

            lower[j] = col[lower_idx.min(n_boot - 1)];
            upper[j] = col[upper_idx.min(n_boot - 1)];
        }

        (lower, upper)
    }
}

/// Hypothesis testing methods
pub struct HypothesisTest;

impl HypothesisTest {
    /// Wald test for linear restrictions on coefficients
    ///
    /// Tests H₀: R·β = q against H₁: R·β ≠ q
    ///
    /// # Arguments
    /// * `beta` - Coefficient estimates (k × 1)
    /// * `cov_matrix` - Variance-covariance matrix (k × k)
    /// * `r` - Restriction matrix (m × k) where m = number of restrictions
    /// * `q` - Restriction values (m × 1), usually zeros
    ///
    /// # Returns
    /// Tuple of (wald_statistic, p_value, degrees_of_freedom)
    ///
    /// # Example
    /// ```no_run
    /// // Test H₀: β₁ = β₂ = 0 (joint significance test)
    /// let r = Array2::from_shape_vec((2, 3), vec![
    ///     0.0, 1.0, 0.0,  // β₁ = 0
    ///     0.0, 0.0, 1.0,  // β₂ = 0
    /// ])?;
    /// let q = Array1::from(vec![0.0, 0.0]);
    ///
    /// let (wald_stat, p_value, df) = HypothesisTest::wald_test(&beta, &cov_matrix, &r, &q)?;
    /// ```
    pub fn wald_test(
        beta: &Array1<f64>,
        cov_matrix: &Array2<f64>,
        r: &Array2<f64>,
        q: &Array1<f64>,
    ) -> Result<(f64, f64, usize), GreenersError> {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let m = r.nrows(); // Number of restrictions

        // Compute R·β - q
        let r_beta = r.dot(beta);
        let diff = &r_beta - q;

        // Compute R·Cov(β)·R'
        let r_cov = r.dot(cov_matrix);
        let r_cov_rt = r_cov.dot(&r.t());

        // Invert R·Cov(β)·R'
        let r_cov_rt_inv = r_cov_rt.inv()?;

        // Wald statistic: (R·β - q)' · [R·Cov(β)·R']^(-1) · (R·β - q)
        let wald_stat = diff.dot(&r_cov_rt_inv.dot(&diff));

        // Under H₀, Wald ~ χ²(m)
        let chi2_dist = ChiSquared::new(m as f64).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - chi2_dist.cdf(wald_stat);

        Ok((wald_stat, p_value, m))
    }

    /// F-test for nested models (OLS specific)
    ///
    /// Tests whether restricted model is adequate vs full model
    ///
    /// # Arguments
    /// * `ssr_restricted` - Sum of squared residuals from restricted model
    /// * `ssr_full` - Sum of squared residuals from full model
    /// * `n` - Number of observations
    /// * `k_full` - Number of parameters in full model
    /// * `k_restricted` - Number of parameters in restricted model
    ///
    /// # Returns
    /// Tuple of (f_statistic, p_value, df_numerator, df_denominator)
    ///
    /// # Formula
    /// F = [(SSR_r - SSR_f) / (k_f - k_r)] / [SSR_f / (n - k_f)]
    pub fn f_test_nested(
        ssr_restricted: f64,
        ssr_full: f64,
        n: usize,
        k_full: usize,
        k_restricted: usize,
    ) -> Result<(f64, f64, usize, usize), GreenersError> {
        use statrs::distribution::{ContinuousCDF, FisherSnedecor};

        let df_num = k_full - k_restricted;
        let df_denom = n - k_full;

        if df_num == 0 {
            return Err(GreenersError::ShapeMismatch(
                "Models have same number of parameters".to_string(),
            ));
        }

        // F-statistic
        let f_stat = ((ssr_restricted - ssr_full) / df_num as f64) / (ssr_full / df_denom as f64);

        // p-value from F distribution
        let f_dist = FisherSnedecor::new(df_num as f64, df_denom as f64)
            .map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - f_dist.cdf(f_stat);

        Ok((f_stat, p_value, df_num, df_denom))
    }

    /// Joint significance test (all coefficients except intercept = 0)
    ///
    /// Convenience wrapper for Wald test of all slope coefficients
    ///
    /// # Arguments
    /// * `beta` - Coefficient estimates (including intercept)
    /// * `cov_matrix` - Variance-covariance matrix
    /// * `has_intercept` - Whether first coefficient is intercept
    ///
    /// # Returns
    /// Tuple of (test_statistic, p_value, degrees_of_freedom)
    pub fn joint_significance(
        beta: &Array1<f64>,
        cov_matrix: &Array2<f64>,
        has_intercept: bool,
    ) -> Result<(f64, f64, usize), GreenersError> {
        let k = beta.len();
        let start_idx = if has_intercept { 1 } else { 0 };
        let m = k - start_idx;

        if m == 0 {
            return Err(GreenersError::ShapeMismatch(
                "No slope coefficients to test".to_string(),
            ));
        }

        // Build restriction matrix: test all slope coefficients = 0
        let mut r = Array2::<f64>::zeros((m, k));
        for i in 0..m {
            r[[i, start_idx + i]] = 1.0;
        }

        let q = Array1::<f64>::zeros(m);

        Self::wald_test(beta, cov_matrix, &r, &q)
    }
}
