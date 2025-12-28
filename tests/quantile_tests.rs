use greeners::{DataFrame, Formula, QuantileReg};
use ndarray::Array1;
use std::collections::HashMap;

#[test]
fn test_quantile_median() {
    // Test median regression (tau = 0.5) with realistic data
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9, 9.1, 10.2]);
    let x = ndarray::Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let result = QuantileReg::fit(&y, &x, 0.5, 50).unwrap(); // tau=0.5, n_boot=50

    assert_eq!(result.params.len(), 2);
    assert_eq!(result.tau, 0.5);
    assert!(result.r_squared >= 0.0);
    assert!(result.iterations > 0);
}

#[test]
fn test_quantile_different_taus() {
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9, 9.1, 10.2]);
    let x = ndarray::Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let result_25 = QuantileReg::fit(&y, &x, 0.25, 50).unwrap();
    let result_50 = QuantileReg::fit(&y, &x, 0.50, 50).unwrap();
    let result_75 = QuantileReg::fit(&y, &x, 0.75, 50).unwrap();

    // Different quantiles should give different estimates
    assert!((result_25.params[1] - result_50.params[1]).abs() > 1e-10);
    assert!((result_50.params[1] - result_75.params[1]).abs() > 1e-10);

    // All should be finite
    assert!(result_25.params[1].is_finite());
    assert!(result_50.params[1].is_finite());
    assert!(result_75.params[1].is_finite());
}

#[test]
fn test_quantile_from_formula() {
    let mut data = HashMap::new();
    data.insert(
        "y".to_string(),
        Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9]),
    );
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9]),
    );

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1").unwrap();

    let result = QuantileReg::from_formula(&formula, &df, 0.5, 50).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.variable_names.is_some());
    let names = result.variable_names.unwrap();
    assert_eq!(names, vec!["const", "x1"]);
}

#[test]
fn test_quantile_invalid_tau() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = ndarray::Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

    // Tau must be in (0, 1)
    let result_zero = QuantileReg::fit(&y, &x, 0.0, 50);
    let result_one = QuantileReg::fit(&y, &x, 1.0, 50);
    let result_negative = QuantileReg::fit(&y, &x, -0.5, 50);
    let result_above_one = QuantileReg::fit(&y, &x, 1.5, 50);

    assert!(result_zero.is_err());
    assert!(result_one.is_err());
    assert!(result_negative.is_err());
    assert!(result_above_one.is_err());
}

#[test]
fn test_quantile_extreme_quantiles() {
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9, 9.1, 10.2]);
    let x = ndarray::Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let result_low = QuantileReg::fit(&y, &x, 0.1, 50).unwrap();
    let result_high = QuantileReg::fit(&y, &x, 0.9, 50).unwrap();

    // Both should converge
    assert!(result_low.params[0].is_finite());
    assert!(result_high.params[0].is_finite());
}

#[test]
fn test_quantile_standard_errors() {
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9]);
    let x = ndarray::Array2::from_shape_vec(
        (8, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0],
    )
    .unwrap();

    let result = QuantileReg::fit(&y, &x, 0.5, 100).unwrap();

    // Standard errors should be positive and finite
    for se in result.std_errors.iter() {
        assert!(se > &0.0);
        assert!(se.is_finite());
    }
}

#[test]
fn test_quantile_t_values_and_p_values() {
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9]);
    let x = ndarray::Array2::from_shape_vec(
        (8, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0],
    )
    .unwrap();

    let result = QuantileReg::fit(&y, &x, 0.5, 100).unwrap();

    // t-values should be finite
    for t in result.t_values.iter() {
        assert!(t.is_finite());
    }

    // p-values should be between 0 and 1
    for p in result.p_values.iter() {
        assert!(p >= &0.0 && p <= &1.0);
    }
}

#[test]
fn test_quantile_convergence() {
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1]);
    let x = ndarray::Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = QuantileReg::fit(&y, &x, 0.5, 50).unwrap();

    // Should converge in reasonable iterations
    assert!(result.iterations > 0);
    assert!(result.iterations < 1000);
}

#[test]
fn test_quantile_with_outliers() {
    // Median regression should be robust to outliers
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 100.0]); // Outlier at end
    let x = ndarray::Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    )
    .unwrap();

    let result = QuantileReg::fit(&y, &x, 0.5, 50).unwrap();

    // Should still produce reasonable estimates despite outlier
    assert!(result.params[1].is_finite());
    assert!(result.params[1] < 10.0); // Shouldn't be drastically affected by outlier
}
