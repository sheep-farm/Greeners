use greeners::{CovarianceType, DataFrame, Formula, OLS};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

#[test]
fn test_ols_basic_estimation() {
    // Simple OLS: y = 2 + 3*x + error
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![5.0, 8.0, 11.0, 14.0, 17.0]; // Perfect fit: y = 2 + 3*x

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust).unwrap();

    // Check coefficients (intercept ≈ 2, slope ≈ 3)
    assert!((result.params[0] - 2.0).abs() < 1e-10);
    assert!((result.params[1] - 3.0).abs() < 1e-10);

    // Check R² = 1 (perfect fit)
    assert!((result.r_squared - 1.0).abs() < 1e-10);
}

#[test]
fn test_ols_multiple_regression() {
    // Test OLS with 2 regressors
    let x1_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2_data = vec![2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5];
    let y_data = vec![4.0, 6.5, 10.0, 12.5, 16.0, 18.5, 22.0, 24.5];

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x1".to_string(), Array1::from(x1_data));
    data.insert("x2".to_string(), Array1::from(x2_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1 + x2").unwrap();
    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust).unwrap();

    // Check basic properties
    assert_eq!(result.params.len(), 3); // Intercept + 2 slopes
    assert!(result.r_squared > 0.90); // Should have good fit
    assert!(result.params.iter().all(|&p| p.is_finite())); // All finite
}

#[test]
fn test_ols_robust_se_hc1() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_data = vec![2.5, 5.2, 7.8, 10.1, 12.9, 15.3, 17.5, 20.2, 22.8, 25.1];

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();

    // Non-robust
    let result_nonrobust = OLS::from_formula(&formula, &df, CovarianceType::NonRobust).unwrap();

    // HC1
    let result_hc1 = OLS::from_formula(&formula, &df, CovarianceType::HC1).unwrap();

    // Coefficients should be identical
    assert!((result_nonrobust.params[0] - result_hc1.params[0]).abs() < 1e-10);
    assert!((result_nonrobust.params[1] - result_hc1.params[1]).abs() < 1e-10);

    // Standard errors may differ (HC1 adjusts for heteroskedasticity)
    assert!(result_hc1.std_errors.len() == 2);
    assert!(result_hc1.std_errors[0] > 0.0);
    assert!(result_hc1.std_errors[1] > 0.0);
}

#[test]
fn test_ols_predictions() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 1 + 2*x

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data.clone()));
    data.insert("x".to_string(), Array1::from(x_data.clone()));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust).unwrap();

    // Get design matrix
    let (y, x) = df.to_design_matrix(&formula).unwrap();

    // Test fitted values
    let fitted = result.fitted_values(&x);
    assert_eq!(fitted.len(), y.len());

    // Check fitted values are close to actual (should be perfect fit)
    for i in 0..fitted.len() {
        assert!((fitted[i] - y[i]).abs() < 1e-10);
    }

    // Test residuals
    let residuals = result.residuals(&y, &x);
    assert_eq!(residuals.len(), y.len());

    // Residuals should be near zero for perfect fit
    for &r in residuals.iter() {
        assert!(r.abs() < 1e-10);
    }

    // Test predictions for new data
    let x_new = Array2::from_shape_vec((2, 2), vec![1.0, 6.0, 1.0, 7.0]).unwrap();
    let predictions = result.predict(&x_new);

    // y = 1 + 2*6 = 13, y = 1 + 2*7 = 15
    assert!((predictions[0] - 13.0).abs() < 1e-10);
    assert!((predictions[1] - 15.0).abs() < 1e-10);
}

#[test]
fn test_ols_no_intercept() {
    // y = 2*x (no intercept)
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x - 1").unwrap(); // -1 removes intercept
    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust).unwrap();

    // Should have only 1 coefficient (no intercept)
    assert_eq!(result.params.len(), 1);
    assert!((result.params[0] - 2.0).abs() < 1e-10);
}

#[test]
fn test_ols_hc3_robust_se() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_data = vec![2.1, 4.3, 6.2, 8.5, 10.1, 12.4, 14.2, 16.8];

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = OLS::from_formula(&formula, &df, CovarianceType::HC3).unwrap();

    // Basic checks
    assert_eq!(result.params.len(), 2);
    assert!(result.r_squared > 0.95);
    assert!(result.std_errors.len() == 2);
}

#[test]
fn test_ols_hc4_robust_se() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_data = vec![2.5, 5.1, 7.3, 9.8, 12.2, 14.9, 17.1, 19.7, 22.3, 24.8];

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = OLS::from_formula(&formula, &df, CovarianceType::HC4).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.std_errors.len() == 2);
    assert!(result.r_squared > 0.90);
}

#[test]
fn test_ols_log_likelihood() {
    // Use data with more substantial noise
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_data = vec![3.5, 4.2, 7.8, 8.1, 11.5, 12.3, 15.9, 16.2, 19.8, 20.5];

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust).unwrap();

    // Log-likelihood should be finite (don't assume sign - depends on variance)
    assert!(result.log_likelihood.is_finite());
}

#[test]
fn test_ols_model_stats() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_data = vec![2.5, 4.8, 7.2, 9.5, 11.9, 14.1];

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust).unwrap();

    let (aic, bic, loglik, adj_r2) = result.model_stats();

    // All should be finite
    assert!(aic.is_finite());
    assert!(bic.is_finite());
    assert!(loglik.is_finite());
    assert!(adj_r2.is_finite());

    // Adjusted R² should be between 0 and 1
    assert!(adj_r2 >= 0.0 && adj_r2 <= 1.0);
}

#[test]
fn test_ols_confidence_intervals() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![3.0, 5.0, 7.0, 9.0, 11.0];

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust).unwrap();

    // Check that confidence intervals are generated (via display)
    assert!(result.params.len() == 2);
    assert!(result.std_errors.len() == 2);

    // CI bounds should satisfy: lower < param < upper
    for i in 0..result.params.len() {
        let param = result.params[i];
        let se = result.std_errors[i];
        let ci_lower = param - 1.96 * se;
        let ci_upper = param + 1.96 * se;

        assert!(ci_lower < param);
        assert!(param < ci_upper);
    }
}
