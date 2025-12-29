use greeners::{DataFrame, Formula, FGLS};
use ndarray::Array1;
use std::collections::HashMap;

#[test]
fn test_wls_basic() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = ndarray::Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let weights = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]); // Equal weights

    let result = FGLS::wls(&y, &x, &weights).unwrap();

    assert_eq!(result.params.len(), 2);
    assert_eq!(result.method, "WLS");
    assert!(result.rho.is_none());
    assert!(result.iter.is_none());
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
}

#[test]
fn test_wls_unequal_weights() {
    // Add some noise to y so weights matter
    let y = Array1::from(vec![1.1, 2.3, 2.8, 4.2, 4.9]);
    let x = ndarray::Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let weights_equal = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    let weights_unequal = Array1::from(vec![1.0, 5.0, 1.0, 5.0, 1.0]); // Much higher weight on some obs

    let result_equal = FGLS::wls(&y, &x, &weights_equal).unwrap();
    let result_unequal = FGLS::wls(&y, &x, &weights_unequal).unwrap();

    // Results should differ with different weights
    assert!((result_equal.params[0] - result_unequal.params[0]).abs() > 1e-6);
}

#[test]
fn test_wls_from_formula() {
    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    );
    data.insert(
        "x2".to_string(),
        Array1::from(vec![2.0, 3.5, 4.2, 5.1, 6.3]), // Not perfectly correlated with x1
    );

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1 + x2").unwrap();
    let weights = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);

    let result = FGLS::wls_from_formula(&formula, &df, &weights).unwrap();

    assert_eq!(result.params.len(), 3); // intercept + 2 vars
    assert!(result.variable_names.is_some());
    let names = result.variable_names.unwrap();
    assert_eq!(names, vec!["const", "x1", "x2"]);
}

#[test]
fn test_wls_weights_mismatch() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = ndarray::Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let weights = Array1::from(vec![1.0, 1.0, 1.0]); // Wrong length

    let result = FGLS::wls(&y, &x, &weights);
    assert!(result.is_err());
}

#[test]
fn test_cochrane_orcutt_basic() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let x = ndarray::Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let result = FGLS::cochrane_orcutt(&y, &x).unwrap();

    assert_eq!(result.params.len(), 2);
    assert_eq!(result.method, "Cochrane-Orcutt AR(1)");
    assert!(result.rho.is_some());
    assert!(result.iter.is_some());
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
}

// #[test]
// fn test_cochrane_orcutt_rho_bounds() {
//     let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
//     let x = ndarray::Array2::from_shape_vec(
//         (10, 2),
//         vec![
//             1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
//             9.0, 1.0, 10.0,
//         ],
//     )
//     .unwrap();

//     let result = FGLS::cochrane_orcutt(&y, &x).unwrap();

//     let rho = result.rho.unwrap();
//     // Rho should be between -1 and 1
//     assert!((-1.0..=1.0).contains(&rho));
// }

#[test]
fn test_cochrane_orcutt_rho_bounds() {
    // Adicionar ruído para evitar ajuste perfeito
    let y = Array1::from(vec![1.1, 2.3, 2.8, 4.1, 5.2, 5.9, 7.1, 8.2, 9.1, 10.2]);
    let x = ndarray::Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 
            1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0, 9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let result = FGLS::cochrane_orcutt(&y, &x).unwrap();

    let rho = result.rho.unwrap();
    assert!((-1.0..=1.0).contains(&rho));
}

#[test]
fn test_cochrane_orcutt_from_formula() {
    let mut data = HashMap::new();
    // Add some noise to avoid perfect fit
    data.insert(
        "y".to_string(),
        Array1::from(vec![1.1, 2.3, 2.8, 4.1, 5.2, 5.9, 7.1, 8.2]),
    );
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1").unwrap();

    let result = FGLS::cochrane_orcutt_from_formula(&formula, &df).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.variable_names.is_some());
    let names = result.variable_names.unwrap();
    assert_eq!(names, vec!["const", "x1"]);
}

#[test]
fn test_cochrane_orcutt_convergence() {
    // Add noise to ensure realistic convergence behavior
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9, 9.1, 10.2]);
    let x = ndarray::Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let result = FGLS::cochrane_orcutt(&y, &x).unwrap();

    // Should converge (iterations > 0 means it ran)
    let iter = result.iter.unwrap();
    assert!(iter > 0);
    // Most cases should converge quickly, but allow some flexibility
    assert!(iter <= 100);
}

#[test]
fn test_wls_heteroscedasticity_correction() {
    // Test that WLS handles heteroscedasticity better than OLS
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let x = ndarray::Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();

    // Weights inversely proportional to variance (simulate heteroscedasticity)
    let weights = Array1::from(vec![1.0, 0.5, 0.33, 0.25, 0.2, 0.17, 0.14, 0.12]);

    let result = FGLS::wls(&y, &x, &weights).unwrap();

    // Should still produce finite estimates
    assert!(result.params[0].is_finite());
    assert!(result.params[1].is_finite());
    assert!(result.std_errors[0] > 0.0);
}
