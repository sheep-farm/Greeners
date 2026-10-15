use faer::linalg::solvers::Lblt;
use faer::{Mat, Side};
use greeners_core::error::GreenersError;
use greeners_core::linalg::{
    LinalgCholesky as _, LinalgEigh as _, LinalgInverse as _, LinalgQR as _, LinalgSVD as _, UPLO,
};
use greeners_core::{CovarianceType, InferenceType};
use greeners_core::{DataFrame, Formula};
use ndarray::{Array1, Array2};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal, StudentsT};
use std::fmt;

/// Type alias for inference computation results: (p_values, conf_lower, conf_upper)
type InferenceResult = (Array1<f64>, Array1<f64>, Array1<f64>);

/// Prediction with standard errors and confidence intervals.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub mean: Array1<f64>,
    pub se: Array1<f64>,
    pub ci_lower: Array1<f64>,
    pub ci_upper: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct OlsResult {
    pub params: Array1<f64>,
    /// Full fitted covariance, in retained coefficient order.
    pub covariance: Array2<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    /// Scaled Wald statistic Q/df_model; NaN means the omnibus test is unavailable.
    pub f_statistic: f64,
    /// Omnibus probability under the selected reference distribution; NaN if unavailable.
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
    pub omitted_vars: Vec<(usize, String)>,  // (position, name) of vars dropped for collinearity
    /// Retained intercept column, including fits to weighted designs.
    pub intercept_index: Option<usize>,
    pub x_clean: Option<Array2<f64>>, // Design matrix after collinearity removal
}

impl OlsResult {
    fn validate_fitted_covariance(&self) -> Result<(), GreenersError> {
        let k = self.params.len();
        if self.covariance.dim() != (k, k) || self.params.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::ShapeMismatch(
                "Fitted covariance dimensions and finite parameters must agree".into(),
            ));
        }
        validate_covariance(&self.covariance)
    }

    fn validate_design(&self, x: &Array2<f64>) -> Result<(), GreenersError> {
        if x.nrows() != self.n_obs || x.ncols() != self.params.len() {
            return Err(GreenersError::ShapeMismatch(
                "Use the retained fitted design dimensions".into(),
            ));
        }
        if x.iter().any(|a| !a.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "Design contains non-finite values".into(),
            ));
        }
        self.validate_fitted_covariance()?;
        Ok(())
    }

    fn scalar_inference(&self, estimate: f64, se: f64) -> Result<(f64, f64), GreenersError> {
        if !estimate.is_finite() || !se.is_finite() || se < 0.0 {
            return Err(GreenersError::InvalidOperation(
                "Scalar inference requires a finite estimate and nonnegative standard error".into(),
            ));
        }
        // Explicit degenerate scalar convention: the zero null has statistic zero.
        let stat = if se > 0.0 {
            estimate / se
        } else if estimate.abs() > 0.0 {
            estimate.signum() * f64::INFINITY
        } else {
            0.0
        };
        let p = match self.inference_type {
            InferenceType::StudentT => {
                2.0 * StudentsT::new(0.0, 1.0, self.df_resid as f64)
                    .map_err(|_| GreenersError::OptimizationFailed)?
                    .sf(stat.abs())
            }
            InferenceType::Normal => {
                2.0 * Normal::new(0.0, 1.0)
                    .map_err(|_| GreenersError::OptimizationFailed)?
                    .sf(stat.abs())
            }
        };
        Ok((stat, p))
    }

    pub(crate) fn refresh_omnibus(&mut self) -> Result<(), GreenersError> {
        self.validate_fitted_covariance()?;
        let indices: Vec<_> = (0..self.params.len())
            .filter(|i| Some(*i) != self.intercept_index)
            .collect();
        self.df_model = indices.len();
        if indices.is_empty() {
            self.f_statistic = f64::NAN;
            self.prob_f = f64::NAN;
            return Ok(());
        }
        let mut r = Array2::zeros((indices.len(), self.params.len()));
        for (row, &col) in indices.iter().enumerate() {
            r[[row, col]] = 1.0;
        }
        match self.wald_from_covariance(&r, &Array1::zeros(indices.len())) {
            Ok((stat, p)) => {
                self.f_statistic = stat;
                self.prob_f = p;
            }
            // The point estimate remains usable when a joint inverse is unavailable.
            Err(GreenersError::SingularMatrix) => {
                self.f_statistic = f64::NAN;
                self.prob_f = f64::NAN;
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }

    fn wald_from_covariance(
        &self,
        r: &Array2<f64>,
        q: &Array1<f64>,
    ) -> Result<(f64, f64), GreenersError> {
        let j = r.nrows();
        if j == 0 || r.ncols() != self.params.len() || q.len() != j {
            return Err(GreenersError::ShapeMismatch(
                "Restrictions must be nonempty and match retained coefficients and targets".into(),
            ));
        }
        if r.iter().chain(q.iter()).any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "Restrictions contain non-finite values".into(),
            ));
        }
        let mut scaled_r = r.clone();
        let mut scaled_q = q.clone();
        for (index, mut row) in scaled_r.rows_mut().into_iter().enumerate() {
            let norm = row.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            if norm <= 0.0 {
                return Err(GreenersError::SingularMatrix);
            }
            row /= norm;
            scaled_q[index] /= norm;
        }
        let (_, singular, _) = scaled_r.svd(false, false)?;
        let largest = singular.iter().copied().fold(0.0_f64, f64::max);
        if j > r.ncols()
            || singular.len() < j
            || singular
                .iter()
                .any(|&v| v <= 128.0 * f64::EPSILON * r.ncols().max(j) as f64 * largest)
        {
            return Err(GreenersError::SingularMatrix);
        }
        let diff = scaled_r.dot(&self.params) - &scaled_q;
        if j == 1 {
            let se = contrast_standard_error(&self.covariance, &scaled_r.row(0).to_owned())?;
            let (stat, p) = self.scalar_inference(diff[0], se)?;
            return Ok((stat * stat, p));
        }
        let mut restricted = scaled_r.dot(&self.covariance).dot(&scaled_r.t());
        validate_covariance(&restricted)?;
        restricted = &restricted * 0.5 + &restricted.t() * 0.5;
        // Standardise before factorisation, because restrictions may have different units.
        let scale = restricted.diag().mapv(f64::sqrt);
        if scale.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
            return Err(GreenersError::SingularMatrix);
        }
        let corr = Array2::from_shape_fn((j, j), |(a, b)| restricted[[a, b]] / scale[a] / scale[b]);
        let (eigenvalues, _) = corr.eigh(UPLO::Lower)?;
        let tol = 128.0 * f64::EPSILON * j as f64;
        if eigenvalues.iter().any(|&v| v <= tol) {
            return Err(GreenersError::SingularMatrix);
        }
        corr.cholesky(UPLO::Lower)?;
        let inverse = corr.inv()?;
        let scaled_diff = &diff / &scale;
        let q_se = contrast_standard_error(&inverse, &scaled_diff)?;
        let q_stat = q_se * q_se;
        if !q_stat.is_finite() || (q_stat <= 0.0 && q_se > 0.0) {
            return Err(GreenersError::InvalidOperation(
                "Joint Wald statistic is not representable".into(),
            ));
        }
        let statistic = q_stat / j as f64;
        let p = match self.inference_type {
            InferenceType::StudentT => FisherSnedecor::new(j as f64, self.df_resid as f64)
                .map_err(|_| GreenersError::OptimizationFailed)?
                .sf(statistic),
            InferenceType::Normal => ChiSquared::new(j as f64)
                .map_err(|_| GreenersError::OptimizationFailed)?
                .sf(q_stat),
        };
        Ok((statistic, p))
    }

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
    /// use greeners_ols::ols::{OLS};
    /// use greeners_core::{CovarianceType};
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
    /// use greeners_ols::ols::{OLS};
    /// use greeners_core::{CovarianceType}; // Importe o enum
    /// use ndarray::{Array1, Array2};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let y = Array1::from(vec![1.0, 2.0, 3.0]);
    /// # let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0])?;
    /// // Add the extra argument here:
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
        use greeners_core::linalg::LinalgInverse as _;
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

    /// Compute confidence intervals at a custom significance level.
    ///
    /// Returns a vector of (lower, upper) tuples, one per coefficient.
    pub fn conf_int(&self, alpha: f64) -> Result<Vec<(f64, f64)>, GreenersError> {
        self.validate_fitted_covariance()?;
        if self.std_errors.len() != self.params.len()
            || self.std_errors.iter().any(|&v| !v.is_finite() || v < 0.0)
        {
            return Err(GreenersError::ShapeMismatch(
                "Coefficient interval standard errors must match finite fitted coefficients".into(),
            ));
        }
        validate_alpha(alpha)?;
        let critical_value = match self.inference_type {
            InferenceType::StudentT => {
                let t_dist = StudentsT::new(0.0, 1.0, self.df_resid as f64)
                    .map_err(|_| GreenersError::OptimizationFailed)?;
                -t_dist.inverse_cdf(alpha / 2.0)
            }
            InferenceType::Normal => {
                let normal_dist =
                    Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
                -normal_dist.inverse_cdf(alpha / 2.0)
            }
        };

        let intervals: Vec<_> = (0..self.params.len())
            .map(|i| {
                let margin = self.std_errors[i] * critical_value;
                (self.params[i] - margin, self.params[i] + margin)
            })
            .collect();
        if intervals
            .iter()
            .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite())
        {
            return Err(GreenersError::InvalidOperation(
                "Coefficient interval exceeds numerical precision".into(),
            ));
        }
        Ok(intervals)
    }

    /// Prediction with standard errors and confidence intervals.
    ///
    /// Returns predictions for `x_new` with associated uncertainty.
    /// Uses the full fitted covariance for mean predictions, excluding new observation noise.
    /// `x_orig` must have the retained fitted dimensions; it does not determine covariance.
    pub fn get_prediction(
        &self,
        x_new: &Array2<f64>,
        x_orig: &Array2<f64>,
        alpha: f64,
    ) -> Result<PredictionResult, GreenersError> {
        self.validate_design(x_orig)?;
        validate_alpha(alpha)?;
        if x_new.ncols() != self.params.len() || x_new.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::ShapeMismatch(
                "Prediction design must be finite and match retained coefficients".into(),
            ));
        }
        let mean = x_new.dot(&self.params);
        if mean.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "Mean prediction is non-finite".into(),
            ));
        }
        let propagation = CovariancePropagation::new(&self.covariance);
        let mut se = Array1::zeros(x_new.nrows());
        for (i, row) in x_new.rows().into_iter().enumerate() {
            se[i] = propagation.standard_error(&row.to_owned())?;
        }

        let critical_value = match self.inference_type {
            InferenceType::StudentT => {
                let t_dist = StudentsT::new(0.0, 1.0, self.df_resid as f64)
                    .map_err(|_| GreenersError::OptimizationFailed)?;
                -t_dist.inverse_cdf(alpha / 2.0)
            }
            InferenceType::Normal => {
                let normal_dist =
                    Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
                -normal_dist.inverse_cdf(alpha / 2.0)
            }
        };

        let margin = &se * critical_value;
        let ci_lower = &mean - &margin;
        let ci_upper = &mean + &margin;
        if ci_lower
            .iter()
            .chain(ci_upper.iter())
            .any(|v| !v.is_finite())
        {
            return Err(GreenersError::InvalidOperation(
                "Mean-prediction interval exceeds numerical precision".into(),
            ));
        }

        Ok(PredictionResult {
            mean,
            se,
            ci_lower,
            ci_upper,
        })
    }

    /// Wald test for linear restrictions R*beta = q.
    ///
    /// H0: R*beta = q
    /// F = (R*b - q)' * [R * V * R']^-1 * (R*b - q) / J
    ///
    /// Returns (Q/J, p-value). StudentT uses F(J, n-k); Normal uses chi-square(J) at Q.
    /// Errors on invalid or redundant restrictions, or a singular joint covariance.
    pub fn wald_test(
        &self,
        r_matrix: &Array2<f64>,
        q: &Array1<f64>,
        x: &Array2<f64>,
    ) -> Result<(f64, f64), GreenersError> {
        self.validate_design(x)?;
        self.wald_from_covariance(r_matrix, q)
    }

    /// F-test for joint significance of a subset of coefficients.
    ///
    /// `indices`: indices of coefficients to test (H0: all are zero).
    pub fn f_test(&self, indices: &[usize], x: &Array2<f64>) -> Result<(f64, f64), GreenersError> {
        let j = indices.len();
        let k = self.params.len();

        if indices.iter().any(|&i| i >= k) {
            return Err(GreenersError::ShapeMismatch(
                "Coefficient index exceeds retained model".into(),
            ));
        }
        let mut r_matrix = Array2::<f64>::zeros((j, k));
        for (row, &col) in indices.iter().enumerate() {
            r_matrix[[row, col]] = 1.0;
        }
        let q = Array1::<f64>::zeros(j);

        self.wald_test(&r_matrix, &q, x)
    }

    /// t-test for a single linear restriction r'*beta = q.
    ///
    /// Returns (t-statistic, p-value).
    pub fn t_test(
        &self,
        r_vector: &Array1<f64>,
        q: f64,
        x: &Array2<f64>,
    ) -> Result<(f64, f64), GreenersError> {
        self.validate_design(x)?;
        if r_vector.len() != self.params.len()
            || !q.is_finite()
            || r_vector.iter().any(|v| !v.is_finite())
        {
            return Err(GreenersError::ShapeMismatch(
                "Scalar restriction must be finite and match retained coefficients".into(),
            ));
        }
        let se = contrast_standard_error(&self.covariance, r_vector)?;
        self.scalar_inference(r_vector.dot(&self.params) - q, se)
    }

    /// Nonlinear combination of coefficients via delta method.
    ///
    /// Computes g(β̂), SE via numerical gradient, t-stat and p-value.
    /// Uses the full fitted covariance and the selected scalar reference distribution.
    ///
    /// # Arguments
    /// * `g` - Function that takes coefficient slice and returns scalar
    /// * `x` - Retained fitted design (dimensions and finite values are validated)
    ///
    /// # Returns
    /// (point_estimate, standard_error, t_statistic, p_value)
    ///
    /// # Errors
    /// Returns an error if the contrast, gradient or propagated covariance is
    /// outside its finite domain or unresolved at numerical precision. Flat
    /// samples along an uncertain coordinate cannot distinguish a constant
    /// function from rounded-away variation; this also limits functions that
    /// ignore such a coefficient. Deterministic coordinates are not differentiated.
    pub fn nlcom<F>(&self, g: F, x: &Array2<f64>) -> Result<(f64, f64, f64, f64), GreenersError>
    where
        F: Fn(&[f64]) -> f64,
    {
        self.validate_design(x)?;
        let params = self.params.as_slice().ok_or_else(|| {
            GreenersError::InvalidOperation("Non-contiguous parameter array".to_string())
        })?;
        let k = params.len();
        let g_hat = g(params);

        if !g_hat.is_finite() {
            return Err(GreenersError::InvalidOperation(
                "Nonlinear contrast is outside its finite domain".into(),
            ));
        }
        let mut grad = Array1::zeros(k);
        for j in 0..k {
            let coordinate_se = self.covariance[[j, j]].sqrt();
            // A deterministic coordinate contributes no covariance propagation.
            if coordinate_se > 0.0 {
                grad[j] = nonlinear_derivative(&g, params, j, coordinate_se, g_hat)?;
            }
        }

        let se = contrast_standard_error(&self.covariance, &grad)?;
        let (t, p) = self.scalar_inference(g_hat, se)?;
        Ok((g_hat, se, t, p))
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
    pub fn compute_inference(
        t_values: &Array1<f64>,
        std_errors: &Array1<f64>,
        params: &Array1<f64>,
        df_resid: usize,
        inference_type: &InferenceType,
    ) -> Result<InferenceResult, GreenersError> {
        if params.len() != std_errors.len() || params.len() != t_values.len() {
            return Err(GreenersError::ShapeMismatch(
                "Inference arrays must have equal lengths".into(),
            ));
        }
        if params.iter().any(|v| !v.is_finite())
            || std_errors.iter().any(|&v| !v.is_finite() || v < 0.0)
            || t_values.iter().any(|v| v.is_nan())
        {
            return Err(GreenersError::InvalidOperation(
                "Coefficient inference contains invalid estimates, errors or statistics".into(),
            ));
        }
        let (p_values, critical_value) = match inference_type {
            InferenceType::StudentT => {
                let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
                    .map_err(|_| GreenersError::OptimizationFailed)?;
                let p_vals = t_values.mapv(|t| {
                    if t.is_nan() {
                        f64::NAN
                    } else if !t.is_finite() {
                        0.0
                    } else {
                        2.0 * t_dist.sf(t.abs())
                    }
                });
                (p_vals, t_dist.inverse_cdf(0.975))
            }
            InferenceType::Normal => {
                let normal_dist =
                    Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
                let p_vals = t_values.mapv(|t| {
                    if t.is_nan() {
                        f64::NAN
                    } else if !t.is_finite() {
                        0.0
                    } else {
                        2.0 * normal_dist.sf(t.abs())
                    }
                });
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
    /// use greeners_ols::ols::{OLS};
    /// use greeners_core::{CovarianceType, InferenceType};
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
    /// let result_z = result.clone().with_inference(InferenceType::Normal)?;
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
        self.refresh_omnibus()?;

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
        let f_str = if self.f_statistic.is_nan() {
            "Unavailable".to_string()
        } else if self.f_statistic.is_infinite() {
            "Inf".to_string()
        } else {
            format!("{:.4}", self.f_statistic)
        };
        let prob_f_str = if self.prob_f.is_nan() {
            "Unavailable".to_string()
        } else if self.f_statistic.is_infinite() {
            "0.0".to_string()
        } else {
            format!("{:.4e}", self.prob_f)
        };
        let global_label = match self.inference_type {
            InferenceType::StudentT => "F-statistic:",
            InferenceType::Normal => "Scaled Wald:",
        };
        let probability_label = match self.inference_type {
            InferenceType::StudentT => "Prob (F-statistic):",
            InferenceType::Normal => "Prob (Wald):",
        };
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Covariance Type:", cov_str, global_label, f_str
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "No. Observations:", self.n_obs, probability_label, prob_f_str
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

        let total = self.params.len() + self.omitted_vars.len();
        let mut fit_idx = 0usize;
        for pos in 0..total {
            if let Some((_, name)) = self.omitted_vars.iter().find(|(p, _)| *p == pos) {
                writeln!(f, "{:<10} |  (omitted)", name)?;
            } else {
                let var_name = if let Some(ref names) = self.variable_names {
                    if fit_idx < names.len() {
                        names[fit_idx].clone()
                    } else {
                        format!("x{}", fit_idx)
                    }
                } else {
                    format!("x{}", fit_idx)
                };
                let t_val = self.t_values[fit_idx];
                let t_str = if t_val.abs() > 1e10 {
                    format!("{:.3e}", t_val)
                } else {
                    format!("{:.3}", t_val)
                };
                writeln!(
                    f,
                    "{:<10} | {:>10.4} | {:>10.4} | {:>8} | {:>8.3} | {:>8.4}  {:>8.4}",
                    var_name,
                    self.params[fit_idx],
                    self.std_errors[fit_idx],
                    t_str,
                    self.p_values[fit_idx],
                    self.conf_lower[fit_idx],
                    self.conf_upper[fit_idx]
                )?;
                fit_idx += 1;
            }
        }

        writeln!(f, "{:=^78}", "")?;
        for (_, name) in &self.omitted_vars {
            writeln!(f, "note: {} omitted because of collinearity", name)?;
        }
        Ok(())
    }
}

pub struct OLS;

impl OLS {
    /// Fits an OLS model using a formula and DataFrame.
    ///
    /// # Examples
    /// ```no_run
    /// use greeners_ols::ols::{OLS};
    /// use greeners_core::{DataFrame, Formula, CovarianceType};
    /// use ndarray::Array1;
    /// use indexmap::IndexMap;
    ///
    /// let mut data = IndexMap::new();
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
        let var_names = data.formula_var_names(formula)?;
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
        Self::fit_internal(y, x, cov_type, variable_names, None)
    }

    pub(crate) fn fit_internal(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
        force_intercept: Option<Option<usize>>,
    ) -> Result<OlsResult, GreenersError> {
        let n = x.nrows();
        let k = x.ncols();

        if n == 0
            || k == 0
            || variable_names
                .as_ref()
                .is_some_and(|names| names.len() != k)
        {
            return Err(GreenersError::ShapeMismatch(
                "Design requires at least one column and one name per column".into(),
            ));
        }
        if y.len() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "y: {}, X: {}",
                y.len(),
                n
            )));
        }
        if variable_names.is_none() && n <= k {
            return Err(GreenersError::ShapeMismatch(
                "Degrees of freedom <= 0".into(),
            ));
        }

        // Check for NaN/Inf in input data
        if y.iter().any(|v| !v.is_finite()) || x.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "Input data contains NaN or Inf values".into(),
            ));
        }

        let (x_to_use, k_clean, omitted_positioned, clean_var_names, x_clean_out) =
            if let Some(ref names) = variable_names {
                let cr = greeners_core::linalg::drop_collinear(x, names, 1e-10);
                let k_clean = cr.x_clean.ncols();
                if n <= k_clean {
                    return Err(GreenersError::ShapeMismatch(
                        "Degrees of freedom <= 0 after removing collinear variables".into(),
                    ));
                }
                let has_omitted = !cr.omitted.is_empty();
                let x_c = cr.x_clean;
                (
                    x_c.clone(),
                    k_clean,
                    cr.omitted,
                    cr.clean_names,
                    if has_omitted { Some(x_c) } else { None },
                )
            } else {
                (x.clone(), k, Vec::new(), Vec::new(), None)
            };

        let x_to_use = &x_to_use;

        // Column-scaled QR avoids squaring the design condition number. The
        // covariance bread is a Gram product, so symmetry follows from its construction.
        let scales: Vec<f64> = x_to_use
            .columns()
            .into_iter()
            .map(|col| col.iter().fold(0.0_f64, |norm, &v| norm.hypot(v)))
            .collect();
        if scales.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err(GreenersError::SingularMatrix);
        }
        let normalised = Array2::from_shape_fn((n, k_clean), |(i, j)| x_to_use[[i, j]] / scales[j]);
        let (q_design, r_design) = normalised.qr()?;
        let rank_tolerance = f64::EPSILON * n.max(k_clean) as f64;
        if r_design.diag().iter().any(|&v| v.abs() <= rank_tolerance) {
            return Err(GreenersError::SingularMatrix);
        }
        let r_inverse = r_design.inv()?;
        let inverse_design = Array2::from_shape_fn((k_clean, n), |(j, i)| {
            r_inverse.row(j).dot(&q_design.row(i)) / scales[j]
        });
        let beta = inverse_design.dot(y);
        let normalised_bread = r_inverse.dot(&r_inverse.t());
        let xt_x_inv = Array2::from_shape_fn((k_clean, k_clean), |(i, j)| {
            normalised_bread[[i, j]] / scales[i] / scales[j]
        });
        if xt_x_inv.diag().iter().any(|&v| v <= 0.0 || !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "Design covariance bread is not representable".into(),
            ));
        }
        // 2. Residuals
        let predicted = x_to_use.dot(&beta);
        let residuals = y - &predicted;
        let ssr = residuals.dot(&residuals);
        if !ssr.is_finite() || (ssr <= 0.0 && residuals.iter().any(|v| v.abs() > 0.0)) {
            return Err(GreenersError::InvalidOperation(
                "Residual sum of squares is not representable at the input scale".into(),
            ));
        }

        let detected_intercept = constant_column(x_to_use);
        let intercept_index = match force_intercept {
            Some(Some(original)) => {
                // Weighted constants retain their common row profile even if QR drops
                // the first constant in favour of another proportional column.
                let profile = x.column(original);
                (0..k_clean).find(|&j| {
                    let ratio = x_to_use[[0, j]] / profile[0];
                    ratio.is_finite()
                        && ratio.abs() > 0.0
                        && (0..n).all(|i| {
                            let expected = profile[i] * ratio;
                            (x_to_use[[i, j]] - expected).abs()
                                <= 128.0 * f64::EPSILON * expected.abs()
                        })
                })
            }
            Some(None) => None,
            None => detected_intercept,
        };
        let has_intercept = intercept_index.is_some();

        let df_resid = n - k_clean;
        let df_model = if has_intercept { k_clean - 1 } else { k_clean };

        let sigma2 = ssr / (df_resid as f64);
        if sigma2 <= 0.0 && ssr > 0.0 {
            return Err(GreenersError::InvalidOperation(
                "Residual variance underflows numerical precision".into(),
            ));
        }
        let sigma = sigma2.sqrt();

        // 3. Covariance Matrix Selection
        let cov_matrix = match &cov_type {
            CovarianceType::NonRobust => &xt_x_inv * sigma2,
            CovarianceType::HC1
            | CovarianceType::HC2
            | CovarianceType::HC3
            | CovarianceType::HC4 => {
                let mut scores = inverse_design.clone();
                for i in 0..n {
                    let leverage = q_design.row(i).dot(&q_design.row(i));
                    let adjustment = match cov_type {
                        CovarianceType::HC1 => ((n as f64) / (df_resid as f64)).sqrt(),
                        // Retain the existing high-leverage boundary convention.
                        _ if leverage >= 0.9999 => 1.0,
                        CovarianceType::HC2 => (1.0 - leverage).sqrt().recip(),
                        CovarianceType::HC3 => (1.0 - leverage).recip(),
                        CovarianceType::HC4 => {
                            let delta = 4.0_f64.min(n as f64 * leverage / k_clean as f64);
                            (1.0 - leverage).powf(-0.5 * delta)
                        }
                        _ => unreachable!("This arm contains only HC covariance variants"),
                    };
                    scores
                        .column_mut(i)
                        .mapv_inplace(|v| v * residuals[i] * adjustment);
                }
                let covariance = scores.dot(&scores.t());
                if (0..k_clean).any(|j| {
                    covariance[[j, j]] <= 0.0 && scores.row(j).iter().any(|v| v.abs() > 0.0)
                }) {
                    return Err(GreenersError::InvalidOperation(
                        "Robust coefficient covariance underflows numerical precision".into(),
                    ));
                }
                covariance
            }
            CovarianceType::NeweyWest(lags) => {
                // Bartlett HAC equals the Gram product of zero-padded sums of
                // L+1 consecutive coefficient influences, divided by L+1.
                let scores = Array2::from_shape_fn((k_clean, n), |(j, i)| {
                    inverse_design[[j, i]] * residuals[i]
                });
                let window_scale = (*lags as f64 + 1.0).sqrt();
                let columns = if *lags >= n - 1 {
                    n.checked_add(n - 1)
                } else {
                    n.checked_add(*lags)
                }
                .ok_or_else(|| {
                    GreenersError::InvalidOperation(
                        "HAC window dimensions exceed the supported range".into(),
                    )
                })?;
                let mut windows = Array2::zeros((k_clean, columns));
                let mut running = Array1::zeros(k_clean);
                if *lags >= n - 1 {
                    for i in 0..n - 1 {
                        running += &scores.column(i);
                        windows.column_mut(i).assign(&(&running / window_scale));
                    }
                    running += &scores.column(n - 1);
                    // All full-length windows have the same sum; combine them
                    // into one weighted column instead of allocating O(L) data.
                    let full_weight = (1.0 - (n - 1) as f64 / (*lags as f64 + 1.0)).sqrt();
                    windows
                        .column_mut(columns - 1)
                        .assign(&(&running * full_weight));
                    running.fill(0.0);
                    for i in (1..n).rev() {
                        running += &scores.column(i);
                        windows
                            .column_mut(n - 1 + (n - 1 - i))
                            .assign(&(&running / window_scale));
                    }
                } else {
                    for t in 0..columns {
                        if t < n {
                            running += &scores.column(t);
                        }
                        if t > *lags {
                            running -= &scores.column(t - *lags - 1);
                        }
                        windows.column_mut(t).assign(&(&running / window_scale));
                    }
                }
                let covariance = windows.dot(&windows.t());
                if (0..k_clean).any(|j| {
                    covariance[[j, j]] <= 0.0 && windows.row(j).iter().any(|v| v.abs() > 0.0)
                }) {
                    return Err(GreenersError::InvalidOperation(
                        "HAC covariance underflows numerical precision".into(),
                    ));
                }
                covariance * (n as f64 / df_resid as f64)
            }
            CovarianceType::Clustered(ref cluster_ids) => {
                let (covariance, g) =
                    cluster_score_covariance(&inverse_design, &residuals, cluster_ids)?;
                if g < 2 {
                    return Err(GreenersError::InvalidOperation(
                        "Cluster covariance requires at least two groups".into(),
                    ));
                }
                covariance * (g as f64 / (g - 1) as f64) * ((n - 1) as f64 / df_resid as f64)
            }
            CovarianceType::ClusteredTwoWay(ref first, ref second) => {
                // Cameron-Gelbach-Miller: sum the two clustered covariances and
                // subtract their intersection, retaining the existing common correction.
                let (covariance1, g1) =
                    cluster_score_covariance(&inverse_design, &residuals, first)?;
                let (covariance2, g2) =
                    cluster_score_covariance(&inverse_design, &residuals, second)?;
                let g = g1.min(g2);
                if g < 2 {
                    return Err(GreenersError::InvalidOperation(
                        "Each cluster dimension requires at least two groups".into(),
                    ));
                }
                let mut pairs = indexmap::IndexMap::new();
                let intersection: Vec<_> = first
                    .iter()
                    .zip(second)
                    .map(|(&a, &b)| {
                        let next = pairs.len();
                        *pairs.entry((a, b)).or_insert(next)
                    })
                    .collect();
                let (joint, _) =
                    cluster_score_covariance(&inverse_design, &residuals, &intersection)?;
                (covariance1 + covariance2 - joint)
                    * (g as f64 / (g - 1) as f64)
                    * ((n - 1) as f64 / df_resid as f64)
            }
        };

        if matches!(cov_type, CovarianceType::NonRobust)
            && sigma2 > 0.0
            && cov_matrix.diag().iter().any(|&v| v <= 0.0)
        {
            return Err(GreenersError::InvalidOperation(
                "Classical coefficient covariance is not representable".into(),
            ));
        }
        validate_covariance(&cov_matrix)?;
        let cov_matrix = &cov_matrix * 0.5 + &cov_matrix.t() * 0.5;
        // Scalar zero-variance inference uses an explicit degenerate convention.
        let std_errors = cov_matrix.diag().mapv(f64::sqrt);
        let t_values = Array1::from_shape_fn(k_clean, |i| {
            if std_errors[i] > 0.0 {
                beta[i] / std_errors[i]
            } else if beta[i].abs() > 0.0 {
                beta[i].signum() * f64::INFINITY
            } else {
                0.0
            }
        });

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
        let sst = if has_intercept {
            let y_mean = y.mean().unwrap_or(0.0);
            y.mapv(|val| (val - y_mean).powi(2)).sum()
        } else {
            y.mapv(|val| val.powi(2)).sum()
        };

        let r_squared = if sst.abs() < 1e-12 {
            0.0
        } else {
            1.0 - (ssr / sst)
        };

        let adj_r_squared = if has_intercept {
            1.0 - (1.0 - r_squared) * ((n as f64 - 1.0) / (df_resid as f64))
        } else {
            1.0 - (1.0 - r_squared) * ((n as f64) / (df_resid as f64))
        };

        let n_f64 = n as f64;
        let log_likelihood =
            -n_f64 / 2.0 * ((2.0 * std::f64::consts::PI).ln() + (ssr / n_f64).ln() + 1.0);
        let aic = 2.0 * (k_clean as f64) - 2.0 * log_likelihood;
        let bic = (k_clean as f64) * n_f64.ln() - 2.0 * log_likelihood;

        let mut result = OlsResult {
            params: beta,
            covariance: cov_matrix,
            std_errors,
            t_values,
            p_values,
            conf_lower,
            conf_upper,
            r_squared,
            adj_r_squared,
            f_statistic: f64::NAN,
            prob_f: f64::NAN,
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
            omitted_vars: omitted_positioned,
            intercept_index,
            x_clean: x_clean_out,
        };
        result.refresh_omnibus()?;
        Ok(result)
    }
}

fn validate_alpha(alpha: f64) -> Result<(), GreenersError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(GreenersError::InvalidOperation(
            "Significance level must lie strictly between zero and one".into(),
        ));
    }
    Ok(())
}

fn validate_covariance(cov: &Array2<f64>) -> Result<(), GreenersError> {
    let k = cov.nrows();
    if k == 0 || cov.ncols() != k || cov.iter().any(|v| !v.is_finite()) {
        return Err(GreenersError::InvalidOperation(
            "Fitted covariance must be finite, square and nonempty".into(),
        ));
    }
    let tolerance = 128.0 * f64::EPSILON * k as f64;
    let mut normalised = Array2::zeros((k, k));
    for i in 0..k {
        if cov[[i, i]] < 0.0 {
            return Err(GreenersError::InvalidOperation(
                "Fitted covariance has negative variance".into(),
            ));
        }
        for j in 0..k {
            let scale = cov[[i, i]].sqrt() * cov[[j, j]].sqrt();
            if scale > 0.0 {
                if (cov[[i, j]] - cov[[j, i]]).abs() > tolerance * scale {
                    return Err(GreenersError::InvalidOperation(
                        "Fitted covariance is materially asymmetric".into(),
                    ));
                }
                normalised[[i, j]] = (0.5 * cov[[i, j]] + 0.5 * cov[[j, i]]) / scale;
            } else if cov[[i, j]].abs() > 0.0 || cov[[j, i]].abs() > 0.0 {
                return Err(GreenersError::InvalidOperation(
                    "Zero covariance variance has nonzero cross-covariance".into(),
                ));
            }
        }
    }
    let (eigenvalues, _) = normalised.eigh(UPLO::Lower)?;
    if eigenvalues
        .iter()
        .any(|&v| !v.is_finite() || v < -tolerance)
    {
        return Err(GreenersError::InvalidOperation(
            "Fitted covariance is materially indefinite".into(),
        ));
    }
    Ok(())
}

fn quadratic_components(cov: &Array2<f64>, r: &Array1<f64>) -> Result<(f64, f64), GreenersError> {
    let mut variance = 0.0;
    let mut magnitude = 0.0;
    for i in 0..r.len() {
        for j in 0..r.len() {
            let term = r[i] * cov[[i, j]] * r[j];
            variance += term;
            magnitude += term.abs();
        }
    }
    let error_budget = 128.0 * f64::EPSILON * r.len().max(1) as f64 * magnitude;
    if !variance.is_finite() || !magnitude.is_finite() || variance < -error_budget {
        return Err(GreenersError::InvalidOperation(
            "Propagated covariance variance is invalid".into(),
        ));
    }
    Ok((variance, error_budget))
}

/// Identify a nonzero constant column before transformations change its values.
pub(crate) fn constant_column(x: &Array2<f64>) -> Option<usize> {
    if x.nrows() == 0 {
        return None;
    }
    (0..x.ncols()).find(|&j| {
        let first = x[[0, j]];
        first.abs() > 0.0
            && x.column(j)
                .iter()
                .all(|&v| (v - first).abs() <= 128.0 * f64::EPSILON * first.abs())
    })
}

/// Prepared scalar propagation is reused across prediction rows. Ordinary
/// quadratic forms must be resolved above their summation-error budget; nearly
/// null contrasts require a factor that reproduces the stored covariance exactly.
struct CovariancePropagation {
    standard_deviations: Array1<f64>,
    correlation: Array2<f64>,
    factor: Lblt<f64>,
    exact_psd_factor: bool,
}

impl CovariancePropagation {
    fn new(cov: &Array2<f64>) -> Self {
        let sd = cov.diag().mapv(f64::sqrt);
        let correlation = Array2::from_shape_fn(cov.dim(), |(i, j)| {
            if sd[i] > 0.0 && sd[j] > 0.0 {
                cov[[i, j]] / sd[i] / sd[j]
            } else {
                0.0
            }
        });
        let matrix = Mat::from_fn(cov.nrows(), cov.ncols(), |i, j| {
            0.5 * cov[[i, j]] + 0.5 * cov[[j, i]]
        });
        let factor = Lblt::new(matrix.as_ref(), Side::Lower);
        let exact_psd_factor = certify_psd_factor(cov, &factor);
        Self {
            standard_deviations: sd,
            correlation,
            factor,
            exact_psd_factor,
        }
    }

    fn standard_error(&self, r: &Array1<f64>) -> Result<f64, GreenersError> {
        let weighted = r * &self.standard_deviations;
        if weighted.iter().any(|v| !v.is_finite())
            || (0..r.len()).any(|i| {
                r[i].abs() > 0.0 && self.standard_deviations[i] > 0.0 && weighted[i].abs() <= 0.0
            })
        {
            return Err(GreenersError::InvalidOperation(
                "Contrast uncertainty cannot be represented in the requested units".into(),
            ));
        }
        let scale = weighted.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        if scale <= 0.0 {
            return Ok(0.0);
        }
        let normalised = &weighted / scale;
        if (0..r.len()).any(|i| weighted[i].abs() > 0.0 && normalised[i].abs() <= 0.0) {
            return Err(GreenersError::InvalidOperation(
                "Contrast coordinates exceed the representable covariance range".into(),
            ));
        }
        let (variance, budget) = quadratic_components(&self.correlation, &normalised)?;
        let se = if variance > budget {
            variance.sqrt() * scale
        } else {
            if !self.exact_psd_factor {
                return Err(GreenersError::InvalidOperation(
                    "Contrast variance is unresolved at the stored covariance precision".into(),
                ));
            }
            let permutation = self.factor.P();
            let (forward, _) = permutation.arrays();
            let mut norm = 0.0_f64;
            for j in 0..r.len() {
                let d = self.factor.B_diag()[j];
                if d > 0.0 {
                    let mut sum = 0.0;
                    let mut correction = 0.0;
                    let mut magnitude = 0.0;
                    let mut rounded = false;
                    for (i, &index) in forward.iter().enumerate() {
                        let coefficient = self.factor.L()[(i, j)];
                        let term = coefficient * r[index];
                        if !term.is_finite()
                            || (term.abs() <= 0.0
                                && coefficient.abs() > 0.0
                                && r[index].abs() > 0.0)
                        {
                            return Err(GreenersError::InvalidOperation(
                                "Factored contrast projection exceeds numerical precision".into(),
                            ));
                        }
                        let product_error = coefficient.mul_add(r[index], -term);
                        let next = sum + term;
                        let addition_error = if sum.abs() >= term.abs() {
                            (sum - next) + term
                        } else {
                            (term - next) + sum
                        };
                        correction += addition_error + product_error;
                        rounded |= addition_error.abs() > 0.0 || product_error.abs() > 0.0;
                        magnitude += term.abs() + product_error.abs();
                        sum = next;
                    }
                    let projection = sum + correction;
                    // Compensated products/sums have a second-order error budget.
                    // A remainder below that budget cannot certify a null direction.
                    let projection_budget =
                        8.0 * f64::EPSILON * f64::EPSILON * r.len() as f64 * magnitude;
                    if !magnitude.is_finite() || (rounded && projection.abs() <= projection_budget)
                    {
                        return Err(GreenersError::InvalidOperation(
                            "Factored contrast projection is unresolved at numerical precision"
                                .into(),
                        ));
                    }
                    let component = projection * d.sqrt();
                    if component.abs() <= 0.0 && projection.abs() > 0.0 {
                        return Err(GreenersError::InvalidOperation(
                            "Factored contrast uncertainty underflows numerical precision".into(),
                        ));
                    }
                    if !component.is_finite() {
                        return Err(GreenersError::InvalidOperation(
                            "Factored contrast is not representable".into(),
                        ));
                    }
                    norm = norm.hypot(component);
                }
            }
            norm
        };
        if !se.is_finite() || (se <= 0.0 && variance > budget) {
            return Err(GreenersError::InvalidOperation(
                "Contrast standard error is not representable".into(),
            ));
        }
        Ok(se)
    }
}

/// Certification uses exact floating-point products and sums, rather than a
/// tolerance that would project unresolved covariance directions onto zero.
/// Failure is conservative: ordinary resolved contrasts remain usable.
fn certify_psd_factor(cov: &Array2<f64>, factor: &Lblt<f64>) -> bool {
    let k = cov.nrows();
    if (0..k).any(|j| {
        !factor.B_diag()[j].is_finite()
            || factor.B_diag()[j] < 0.0
            || factor.B_subdiag()[j].abs() > 0.0
    }) {
        return false;
    }
    let permutation = factor.P();
    let (forward, _) = permutation.arrays();
    for i in 0..k {
        for j in 0..k {
            let mut value = 0.0;
            for t in 0..k {
                let a = factor.L()[(i, t)];
                let b = factor.B_diag()[t];
                let c = factor.L()[(j, t)];
                if !a.is_finite() || !c.is_finite() {
                    return false;
                }
                let term = if a.abs() <= 0.0 || b <= 0.0 || c.abs() <= 0.0 {
                    0.0
                } else {
                    // Binary64 products contain up to 106 significant bits.
                    // A rounded product needs exponent headroom for its lowest
                    // exact bit, including a possible binade carry; otherwise an
                    // FMA residual can underflow even for a normal product.
                    // The conservative floor is 2^-968, not an SE threshold.
                    let first = a * b;
                    if !first.is_finite()
                        || first.abs() < 4.0 * f64::MIN_POSITIVE / f64::EPSILON
                        || a.mul_add(b, -first).abs() > 0.0
                    {
                        return false;
                    }
                    let term = first * c;
                    if !term.is_finite()
                        || term.abs() < 4.0 * f64::MIN_POSITIVE / f64::EPSILON
                        || first.mul_add(c, -term).abs() > 0.0
                    {
                        return false;
                    }
                    term
                };
                let next = value + term;
                let residual = if value.abs() >= term.abs() {
                    (value - next) + term
                } else {
                    (term - next) + value
                };
                if !next.is_finite() || residual.abs() > 0.0 {
                    return false;
                }
                value = next;
            }
            let expected = cov[[forward[i], forward[j]]];
            if value.to_bits() != expected.to_bits()
                && !(value.abs() <= 0.0 && expected.abs() <= 0.0)
            {
                return false;
            }
        }
    }
    true
}

fn contrast_standard_error(cov: &Array2<f64>, r: &Array1<f64>) -> Result<f64, GreenersError> {
    CovariancePropagation::new(cov).standard_error(r)
}

/// Adaptive central differences in coefficient units, with Richardson
/// convergence, subtraction-roundoff and local smoothness/domain checks.
fn nonlinear_derivative<F>(
    g: &F,
    params: &[f64],
    j: usize,
    coordinate_se: f64,
    centre: f64,
) -> Result<f64, GreenersError>
where
    F: Fn(&[f64]) -> f64,
{
    let parameter = params[j];
    let mut step = f64::EPSILON.cbrt() * parameter.abs().max(coordinate_se);
    let mut point = params.to_vec();
    let mut previous_difference = None;
    let mut previous_extrapolation: Option<f64> = None;
    let mut previous_gap = None;
    for _ in 0..20 {
        let plus = parameter + step;
        let minus = parameter - step;
        if !plus.is_finite()
            || !minus.is_finite()
            || plus.to_bits() == parameter.to_bits()
            || minus.to_bits() == parameter.to_bits()
        {
            break;
        }
        point[j] = plus;
        let plus_value = g(&point);
        point[j] = minus;
        let minus_value = g(&point);
        if !plus_value.is_finite() || !minus_value.is_finite() {
            previous_difference = None;
            previous_extrapolation = None;
            previous_gap = None;
            step *= 0.5;
            continue;
        }
        let span = plus - minus;
        let difference = (plus_value - minus_value) / span;
        let gap = ((plus_value - centre) / (plus - parameter)
            - (centre - minus_value) / (parameter - minus))
            .abs();
        let roundoff =
            8.0 * f64::EPSILON * (plus_value.abs() + minus_value.abs() + 2.0 * centre.abs()) / span;
        if !difference.is_finite() || !gap.is_finite() || !roundoff.is_finite() {
            break;
        }
        // Constant sampled values cannot distinguish a constant function from
        // variation lost to rounding. A black-box derivative is then unavailable.
        let flat =
            plus_value.to_bits() == centre.to_bits() && minus_value.to_bits() == centre.to_bits();
        if let Some(previous) = previous_difference {
            let extrapolation: f64 = difference + (difference - previous) / 3.0;
            if let Some(previous_extrapolation) = previous_extrapolation {
                let tolerance = f64::EPSILON.sqrt()
                    * extrapolation.abs().max(previous_extrapolation.abs())
                    + roundoff;
                let smooth = gap <= tolerance
                    || previous_gap.is_some_and(|old| gap <= 0.75 * old + tolerance);
                if !flat && (extrapolation - previous_extrapolation).abs() <= tolerance && smooth {
                    return Ok(extrapolation);
                }
            }
            previous_extrapolation = Some(extrapolation);
        }
        previous_difference = Some(difference);
        previous_gap = Some(gap);
        step *= 0.5;
    }
    Err(GreenersError::InvalidOperation(format!("Nonlinear derivative for coefficient {j} is unresolved in its domain or numerical precision")))
}

/// Covariance before finite-sample correction, from summed coefficient influences.
/// A Gram product preserves symmetry without left/right sandwich cancellation.
fn cluster_score_covariance(
    inverse_design: &Array2<f64>,
    residuals: &Array1<f64>,
    labels: &[usize],
) -> Result<(Array2<f64>, usize), GreenersError> {
    let k = inverse_design.nrows();
    if labels.len() != residuals.len() {
        return Err(GreenersError::ShapeMismatch(
            "Cluster IDs must match fitted observations".into(),
        ));
    }
    let mut sums = indexmap::IndexMap::<usize, Array1<f64>>::new();
    for (i, &label) in labels.iter().enumerate() {
        let score = sums.entry(label).or_insert_with(|| Array1::zeros(k));
        for j in 0..k {
            score[j] += inverse_design[[j, i]] * residuals[i];
        }
    }
    let scores = Array2::from_shape_fn((k, sums.len()), |(j, g)| sums[g][j]);
    let covariance = scores.dot(&scores.t());
    if (0..k).any(|j| covariance[[j, j]] <= 0.0 && scores.row(j).iter().any(|v| v.abs() > 0.0)) {
        return Err(GreenersError::InvalidOperation(
            "Cluster covariance underflows numerical precision".into(),
        ));
    }
    Ok((covariance, sums.len()))
}
