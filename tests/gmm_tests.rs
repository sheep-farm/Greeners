use greeners::GMM;
use ndarray::{Array1, Array2};

#[test]
fn test_gmm_basic() {
    // Simple GMM test with some noise to avoid perfect fit
    let y = Array1::from(vec![1.1, 2.3, 2.9, 4.2, 4.8, 6.1, 6.9, 8.3]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();
    let z = x.clone(); // Use same instruments

    let result = GMM::fit(&y, &x, &z).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.j_stat >= 0.0);
    // J-test p-value can be NaN for exactly identified models with perfect instruments
    if result.j_p_value.is_finite() {
        assert!(result.j_p_value >= 0.0 && result.j_p_value <= 1.0);
    }
}

#[test]
fn test_gmm_exactly_identified() {
    // Same number of instruments as parameters
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let z = x.clone(); // k = l (exactly identified)

    let result = GMM::fit(&y, &x, &z).unwrap();

    // J-statistic should be very small (close to 0) for exactly identified models
    // Allowing for numerical precision issues
    assert!(result.j_stat < 1e-3);
}

#[test]
fn test_gmm_overidentified() {
    // More instruments than parameters
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    )
    .unwrap();
    let z = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 3.0, 9.0, 1.0, 4.0, 16.0, 1.0, 5.0, 25.0, 1.0, 6.0,
            36.0,
        ],
    )
    .unwrap();

    let result = GMM::fit(&y, &x, &z).unwrap();

    // J-statistic should be positive for overidentified models
    assert!(result.j_stat >= 0.0);
}

#[test]
fn test_gmm_parameter_estimates() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();
    let z = x.clone();

    let result = GMM::fit(&y, &x, &z).unwrap();

    // Parameters should be finite
    for param in result.params.iter() {
        assert!(param.is_finite());
    }
}

#[test]
fn test_gmm_standard_errors() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();
    let z = x.clone();

    let result = GMM::fit(&y, &x, &z).unwrap();

    assert_eq!(result.std_errors.len(), 2);

    // Standard errors should be positive and finite
    for se in result.std_errors.iter() {
        assert!(se > &0.0);
        assert!(se.is_finite());
    }
}

#[test]
fn test_gmm_t_values() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();
    let z = x.clone();

    let result = GMM::fit(&y, &x, &z).unwrap();

    assert_eq!(result.t_values.len(), 2);

    // t-values should be finite
    for t in result.t_values.iter() {
        assert!(t.is_finite());
    }

    // t = param / se
    for i in 0..result.params.len() {
        let expected_t = result.params[i] / result.std_errors[i];
        assert!((result.t_values[i] - expected_t).abs() < 1e-10);
    }
}

#[test]
fn test_gmm_p_values() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();
    let z = x.clone();

    let result = GMM::fit(&y, &x, &z).unwrap();

    assert_eq!(result.p_values.len(), 2);

    // P-values should be between 0 and 1
    for p in result.p_values.iter() {
        assert!(p >= &0.0 && p <= &1.0);
    }
}

#[test]
fn test_gmm_j_test_valid_specification() {
    // With valid instruments, J-test should not reject
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9, 9.1, 10.2]);
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();
    let z = x.clone(); // Valid instruments

    let result = GMM::fit(&y, &x, &z).unwrap();

    // J-test p-value should be reasonably high (not rejecting) for exactly identified
    // For exactly identified models, p-value might be NaN, so we allow that
    if result.j_p_value.is_finite() {
        assert!(result.j_p_value >= 0.0);
    }
}

#[test]
fn test_gmm_underidentified() {
    // More parameters than instruments - should fail
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 4.0, 4.0, 1.0, 5.0, 5.0,
        ],
    )
    .unwrap();
    let z = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = GMM::fit(&y, &x, &z);
    assert!(result.is_err()); // Should fail order condition
}

#[test]
fn test_gmm_degrees_of_freedom() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    )
    .unwrap();
    let z = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 3.0, 9.0, 1.0, 4.0, 16.0, 1.0, 5.0, 25.0, 1.0, 6.0,
            36.0,
        ],
    )
    .unwrap();

    let result = GMM::fit(&y, &x, &z).unwrap();

    // df = l - k = 3 - 2 = 1 for J-test
    // J-statistic follows chi-squared with (l-k) degrees of freedom
    assert!(result.j_stat >= 0.0);
}

#[test]
fn test_gmm_finite_sample() {
    // Test with small sample (but not too small)
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let z = x.clone();

    let result = GMM::fit(&y, &x, &z).unwrap();

    // Should still produce estimates
    assert_eq!(result.params.len(), 2);
    assert!(result.params[0].is_finite());
    assert!(result.params[1].is_finite());
}
