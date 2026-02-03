use crate::error::GreenersError;
use crate::ols::OlsResult;
use crate::{CovarianceType, DataFrame, Formula, OLS};
use ndarray::{Array1, Array2};

/// Weighted Least Squares estimator.
///
/// Transforms the model by sqrt(w): y* = sqrt(w)*y, X* = sqrt(w)*X,
/// then runs OLS on the transformed data.
pub struct WLS;

impl WLS {
    /// Fit WLS from a formula, DataFrame, and weight column name.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        weights: &Array1<f64>,
        cov_type: CovarianceType,
    ) -> Result<OlsResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;

        let mut var_names = Vec::new();
        if formula.intercept {
            var_names.push("const".to_string());
        }
        for var in &formula.independents {
            var_names.push(var.clone());
        }

        Self::fit_with_names(&y, &x, weights, cov_type, Some(var_names))
    }

    /// Fit WLS from arrays.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        weights: &Array1<f64>,
        cov_type: CovarianceType,
    ) -> Result<OlsResult, GreenersError> {
        Self::fit_with_names(y, x, weights, cov_type, None)
    }

    /// Fit WLS with variable names.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        weights: &Array1<f64>,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
    ) -> Result<OlsResult, GreenersError> {
        let n = y.len();
        if weights.len() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "weights length ({}) must match observations ({})",
                weights.len(),
                n
            )));
        }
        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "X rows ({}) must match y length ({})",
                x.nrows(),
                n
            )));
        }

        // Validate weights: must be positive
        if weights.iter().any(|&w| w <= 0.0 || !w.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "Weights must be positive and finite".into(),
            ));
        }

        // Transform: multiply by sqrt(w)
        let sqrt_w = weights.mapv(f64::sqrt);

        let y_star = &sqrt_w * y;
        let mut x_star = x.clone();
        for i in 0..n {
            x_star.row_mut(i).mapv_inplace(|val| val * sqrt_w[i]);
        }

        OLS::fit_with_names(&y_star, &x_star, cov_type, variable_names)
    }
}
