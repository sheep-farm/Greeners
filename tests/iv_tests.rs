use greeners::{CovarianceType, DataFrame, Formula, IV};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

#[test]
fn test_iv_basic_estimation() {
    // Simple IV test with valid instruments
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let z = x.clone(); // Perfect instruments

    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert!(result.n_obs == 5);
}

#[test]
fn test_iv_from_formula() {
    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.2, 2.3, 2.9, 4.1, 5.2]));
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    );
    data.insert(
        "z1".to_string(),
        Array1::from(vec![1.0, 2.5, 3.0, 3.5, 5.0]),
    );

    let df = DataFrame::new(data).unwrap();
    let endog_formula = Formula::parse("y ~ x1").unwrap();
    // Use a dummy LHS for instrument formula since parser requires it
    let instrument_formula = Formula::parse("z1 ~ z1").unwrap();

    let result = IV::from_formula(
        &endog_formula,
        &instrument_formula,
        &df,
        CovarianceType::HC1,
    )
    .unwrap();

    assert_eq!(result.params.len(), 2); // intercept + x1
    assert!(result.variable_names.is_some());
    let names = result.variable_names.unwrap();
    assert_eq!(names, vec!["const", "x1"]);
}

#[test]
fn test_iv_order_condition() {
    // Should fail: more endogenous vars than instruments
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x =
        Array2::from_shape_vec((3, 3), vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 3.0]).unwrap();
    let z = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust);
    assert!(result.is_err()); // Order condition violated
}

#[test]
fn test_iv_predictions() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let z = x.clone();

    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust).unwrap();

    let x_new = Array2::from_shape_vec((2, 2), vec![1.0, 6.0, 1.0, 7.0]).unwrap();
    let predictions = result.predict(&x_new);

    assert_eq!(predictions.len(), 2);
    assert!(predictions[0].is_finite());
    assert!(predictions[1].is_finite());
}

#[test]
fn test_iv_robust_se() {
    // Add noise to create heteroscedasticity
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9, 9.1, 10.2]);
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();
    let z = x.clone();

    let result_nonrobust = IV::fit(&y, &x, &z, CovarianceType::NonRobust).unwrap();
    let result_hc1 = IV::fit(&y, &x, &z, CovarianceType::HC1).unwrap();

    // Coefficients should be the same
    assert!((result_nonrobust.params[0] - result_hc1.params[0]).abs() < 1e-10);

    // Standard errors may differ depending on heteroscedasticity
    // Both should be positive
    assert!(result_nonrobust.std_errors[0] > 0.0);
    assert!(result_hc1.std_errors[0] > 0.0);
}

#[test]
fn test_iv_residuals_and_fitted() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let z = x.clone();

    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust).unwrap();

    let fitted = result.fitted_values(&x);
    let residuals = result.residuals(&y, &x);

    assert_eq!(fitted.len(), 5);
    assert_eq!(residuals.len(), 5);

    // y = fitted + residuals
    for i in 0..5 {
        assert!((y[i] - (fitted[i] + residuals[i])).abs() < 1e-10);
    }
}
