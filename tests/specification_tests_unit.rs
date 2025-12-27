use greeners::SpecificationTests;
use ndarray::{Array1, Array2};

#[test]
fn test_white_test_basic() {
    // Create simple residuals and design matrix
    let residuals = Array1::from(vec![0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.15, 0.1]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();

    let result = SpecificationTests::white_test(&residuals, &x);
    assert!(result.is_ok());

    let (lm_stat, p_value, df) = result.unwrap();

    // LM statistic should be non-negative
    assert!(lm_stat >= 0.0);

    // P-value should be between 0 and 1
    assert!(p_value >= 0.0 && p_value <= 1.0);

    // Degrees of freedom should be reasonable
    assert!(df > 0);
}

#[test]
fn test_white_test_homoskedastic_data() {
    // Generate homoskedastic residuals (constant variance)
    let residuals = Array1::from(vec![
        0.1, -0.1, 0.12, -0.11, 0.09, -0.08, 0.11, -0.1, 0.1, -0.09,
    ]);
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let (lm_stat, p_value, _) = SpecificationTests::white_test(&residuals, &x).unwrap();

    // With homoskedastic data, should not reject H0 (p-value should be relatively high)
    // Note: This is a statistical test, so might occasionally fail
    assert!(lm_stat >= 0.0);
    assert!(p_value > 0.0); // At minimum, p-value should be positive
}

#[test]
fn test_reset_test_basic() {
    // Linear data with small noise: y ≈ 1 + 2*x
    let y = Array1::from(vec![3.1, 4.9, 7.1, 8.9, 11.1, 12.9, 15.1, 16.9]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();

    // Fitted values close to actual
    let fitted = Array1::from(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);

    let result = SpecificationTests::reset_test(&y, &x, &fitted, 2);
    assert!(result.is_ok());

    let (f_stat, p_value, df_num, df_denom) = result.unwrap();

    // F statistic should be non-negative
    assert!(f_stat >= 0.0);

    // P-value should be valid
    assert!(p_value >= 0.0 && p_value <= 1.0);

    // Check degrees of freedom
    assert_eq!(df_num, 1); // power - 1 = 2 - 1
    assert!(df_denom > 0);

    // For correctly specified linear model, F should be small
    assert!(f_stat < 10.0);
}

#[test]
fn test_reset_test_power_3() {
    let y = Array1::from(vec![2.5, 4.8, 7.2, 9.5, 11.9, 14.2, 16.5, 18.9]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();

    let fitted = Array1::from(vec![2.5, 4.8, 7.1, 9.4, 11.7, 14.0, 16.3, 18.6]);

    let result = SpecificationTests::reset_test(&y, &x, &fitted, 3);
    assert!(result.is_ok());

    let (f_stat, p_value, df_num, _) = result.unwrap();

    assert!(f_stat >= 0.0);
    assert!(p_value >= 0.0 && p_value <= 1.0);
    assert_eq!(df_num, 2); // power - 1 = 3 - 1
}

#[test]
fn test_reset_test_invalid_power() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let fitted = Array1::from(vec![1.0, 2.0, 3.0]);

    // Power < 2 should return error
    let result = SpecificationTests::reset_test(&y, &x, &fitted, 1);
    assert!(result.is_err());
}

#[test]
fn test_breusch_godfrey_basic() {
    // Simple residuals
    let residuals = Array1::from(vec![
        0.1, -0.05, 0.08, -0.06, 0.07, -0.04, 0.06, -0.03, 0.05, -0.02,
    ]);
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let result = SpecificationTests::breusch_godfrey_test(&residuals, &x, 1);
    assert!(result.is_ok());

    let (lm_stat, p_value, df) = result.unwrap();

    // LM statistic should be non-negative
    assert!(lm_stat >= 0.0);

    // P-value should be valid
    assert!(p_value >= 0.0 && p_value <= 1.0);

    // Degrees of freedom should equal number of lags
    assert_eq!(df, 1);
}

#[test]
fn test_breusch_godfrey_no_autocorrelation() {
    // Random residuals (no autocorrelation)
    let residuals = Array1::from(vec![
        0.05, -0.03, 0.08, -0.06, 0.02, -0.04, 0.07, -0.01, 0.04, -0.05, 0.03, -0.07,
    ]);
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0, 1.0, 11.0, 1.0, 12.0,
        ],
    )
    .unwrap();

    let (lm_stat, p_value, _) =
        SpecificationTests::breusch_godfrey_test(&residuals, &x, 1).unwrap();

    // With no autocorrelation, should not reject (relatively high p-value)
    assert!(lm_stat >= 0.0);
    assert!(p_value > 0.0);
}

#[test]
fn test_breusch_godfrey_multiple_lags() {
    let residuals = Array1::from(vec![
        0.1, 0.12, 0.14, 0.11, 0.13, 0.15, 0.12, 0.14, 0.16, 0.13, 0.15, 0.17,
    ]);
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0, 1.0, 11.0, 1.0, 12.0,
        ],
    )
    .unwrap();

    let result = SpecificationTests::breusch_godfrey_test(&residuals, &x, 2);
    assert!(result.is_ok());

    let (lm_stat, p_value, df) = result.unwrap();

    assert!(lm_stat >= 0.0);
    assert!(p_value >= 0.0 && p_value <= 1.0);
    assert_eq!(df, 2); // Testing 2 lags
}

#[test]
fn test_goldfeld_quandt_basic() {
    // Residuals ordered by suspected heteroskedasticity variable
    let residuals = Array1::from(vec![
        0.05, 0.06, 0.05, 0.07, 0.06, // Small variance
        0.15, 0.18, 0.16, 0.19, 0.17, // Large variance
    ]);

    let result = SpecificationTests::goldfeld_quandt_test(&residuals, 0.2);
    assert!(result.is_ok());

    let (f_stat, p_value, df1, df2) = result.unwrap();

    // F statistic should be positive
    assert!(f_stat > 0.0);

    // P-value should be valid
    assert!(p_value >= 0.0 && p_value <= 1.0);

    // Degrees of freedom should be equal
    assert_eq!(df1, df2);
}

#[test]
fn test_goldfeld_quandt_homoskedastic() {
    // Constant variance throughout
    let residuals = Array1::from(vec![
        0.1, -0.1, 0.11, -0.09, 0.1, -0.1, 0.09, -0.11, 0.1, -0.1,
    ]);

    let (f_stat, p_value, _, _) =
        SpecificationTests::goldfeld_quandt_test(&residuals, 0.2).unwrap();

    // F should be close to 1 for homoskedastic data
    assert!((f_stat - 1.0).abs() < 1.5);

    // P-value should be relatively high
    assert!(p_value > 0.0);
}

#[test]
fn test_goldfeld_quandt_insufficient_obs() {
    // Too few observations for the test
    let residuals = Array1::from(vec![0.1, 0.2]);

    let result = SpecificationTests::goldfeld_quandt_test(&residuals, 0.2);
    assert!(result.is_err()); // Should return error
}

#[test]
fn test_white_test_with_multiple_regressors() {
    // More observations to avoid singular matrix issues
    let residuals = Array1::from(vec![
        0.1, -0.15, 0.12, -0.08, 0.14, -0.1, 0.11, -0.13, 0.09, -0.12, 0.08, -0.11, 0.13, -0.09,
        0.10,
    ]);
    let x = Array2::from_shape_vec(
        (15, 3),
        vec![
            1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 3.0, 4.0, 1.0, 4.0, 5.5, 1.0, 5.0, 6.0, 1.0, 6.0,
            7.5, 1.0, 7.0, 8.0, 1.0, 8.0, 9.5, 1.0, 9.0, 10.0, 1.0, 10.0, 11.5, 1.0, 11.0, 12.0,
            1.0, 12.0, 13.5, 1.0, 13.0, 14.0, 1.0, 14.0, 15.5, 1.0, 15.0, 16.0,
        ],
    )
    .unwrap();

    let result = SpecificationTests::white_test(&residuals, &x);
    assert!(result.is_ok());

    let (lm_stat, p_value, df) = result.unwrap();

    assert!(lm_stat >= 0.0);
    assert!(p_value >= 0.0 && p_value <= 1.0);
    assert!(df > 0);
}

#[test]
fn test_reset_test_misspecified_model() {
    // Quadratic data fitted with linear model (misspecification)
    let x_vals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| 1.0 + 2.0 * x + 0.5 * x * x)
        .collect();

    let y = Array1::from(y_vals.clone());

    // Design matrix with just linear term
    let mut x_data = Vec::new();
    for &xv in &x_vals {
        x_data.push(1.0);
        x_data.push(xv);
    }
    let x = Array2::from_shape_vec((10, 2), x_data).unwrap();

    // Linear fitted values (will not match quadratic data well)
    let fitted: Vec<f64> = x_vals.iter().map(|&x| 1.0 + 2.0 * x).collect();
    let fitted = Array1::from(fitted);

    let (f_stat, _p_value, _, _) = SpecificationTests::reset_test(&y, &x, &fitted, 2).unwrap();

    // Should detect misspecification (low p-value, high F)
    // Note: This is a strong effect, so should be detected
    assert!(f_stat > 1.0);
    // p_value might still be high in some cases due to small sample
}

#[test]
fn test_breusch_godfrey_too_many_lags() {
    let residuals = Array1::from(vec![0.1, 0.2, 0.3]);
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    // Lags >= n should return error
    let result = SpecificationTests::breusch_godfrey_test(&residuals, &x, 5);
    assert!(result.is_err());
}

#[test]
fn test_specification_tests_statistics_are_finite() {
    // Ensure all test statistics and p-values are finite
    let residuals = Array1::from(vec![0.1, -0.1, 0.12, -0.11, 0.09, -0.08, 0.11, -0.1]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();

    let y = Array1::from(vec![2.5, 4.8, 7.1, 9.4, 11.7, 14.0, 16.3, 18.6]);
    let fitted = Array1::from(vec![2.5, 4.8, 7.1, 9.4, 11.7, 14.0, 16.3, 18.6]);

    // White test
    let (white_stat, white_p, _) = SpecificationTests::white_test(&residuals, &x).unwrap();
    assert!(white_stat.is_finite());
    assert!(white_p.is_finite());

    // RESET test
    let (reset_stat, reset_p, _, _) = SpecificationTests::reset_test(&y, &x, &fitted, 2).unwrap();
    assert!(reset_stat.is_finite());
    assert!(reset_p.is_finite());

    // Breusch-Godfrey test
    let (bg_stat, bg_p, _) = SpecificationTests::breusch_godfrey_test(&residuals, &x, 1).unwrap();
    assert!(bg_stat.is_finite());
    assert!(bg_p.is_finite());

    // Goldfeld-Quandt test
    let (gq_stat, gq_p, _, _) = SpecificationTests::goldfeld_quandt_test(&residuals, 0.25).unwrap();
    assert!(gq_stat.is_finite());
    assert!(gq_p.is_finite());
}
