use greeners::{Bootstrap, HypothesisTest};
use ndarray::{Array1, Array2};

#[test]
fn test_bootstrap_pairs_basic() {
    // Simple data: y = 2 + 3*x
    let y = Array1::from(vec![5.0, 8.0, 11.0, 14.0, 17.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let n_bootstrap = 100;
    let boot_coefs = Bootstrap::pairs_bootstrap(&y, &x, n_bootstrap).unwrap();

    // Check dimensions
    assert_eq!(boot_coefs.nrows(), n_bootstrap);
    assert_eq!(boot_coefs.ncols(), 2); // Intercept + slope

    // All coefficients should be finite
    for i in 0..n_bootstrap {
        assert!(boot_coefs[[i, 0]].is_finite());
        assert!(boot_coefs[[i, 1]].is_finite());
    }
}

#[test]
fn test_bootstrap_standard_errors() {
    let y = Array1::from(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0]);
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    )
    .unwrap();

    let boot_coefs = Bootstrap::pairs_bootstrap(&y, &x, 200).unwrap();
    let boot_se = Bootstrap::bootstrap_se(&boot_coefs);

    // Check dimensions
    assert_eq!(boot_se.len(), 2);

    // Standard errors should be positive
    assert!(boot_se[0] > 0.0);
    assert!(boot_se[1] > 0.0);

    // Standard errors should be finite
    assert!(boot_se.iter().all(|&se| se.is_finite()));
}

#[test]
fn test_bootstrap_percentile_ci() {
    let y = Array1::from(vec![2.5, 4.8, 7.1, 9.3, 11.7, 14.2, 16.5]);
    let x = Array2::from_shape_vec(
        (7, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0,
        ],
    )
    .unwrap();

    let boot_coefs = Bootstrap::pairs_bootstrap(&y, &x, 500).unwrap();
    let (lower, upper) = Bootstrap::percentile_ci(&boot_coefs, 0.05);

    // Check dimensions
    assert_eq!(lower.len(), 2);
    assert_eq!(upper.len(), 2);

    // Lower bounds should be less than upper bounds
    assert!(lower[0] < upper[0]);
    assert!(lower[1] < upper[1]);

    // All bounds should be finite
    assert!(lower.iter().all(|&l| l.is_finite()));
    assert!(upper.iter().all(|&u| u.is_finite()));
}

#[test]
fn test_bootstrap_confidence_interval_coverage() {
    // True parameters: intercept = 1, slope = 2
    let y = Array1::from(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let boot_coefs = Bootstrap::pairs_bootstrap(&y, &x, 1000).unwrap();
    let (lower, upper) = Bootstrap::percentile_ci(&boot_coefs, 0.05);

    // For perfect data, true parameters should be within CI
    // Intercept ≈ 1, Slope ≈ 2
    let true_intercept = 1.0;
    let true_slope = 2.0;

    // Note: This might fail occasionally due to sampling variability
    // but should pass most of the time for this simple case
    assert!(lower[0] <= true_intercept + 0.5 && upper[0] >= true_intercept - 0.5);
    assert!(lower[1] <= true_slope + 0.5 && upper[1] >= true_slope - 0.5);
}

#[test]
fn test_wald_test_basic() {
    // Test H0: β₁ = 0
    let beta = Array1::from(vec![2.0, 3.5]); // Intercept, slope
    let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.25, 0.0, 0.0, 0.16]).unwrap();

    // Test β₁ = 0
    let r = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
    let q = Array1::from(vec![0.0]);

    let (wald_stat, p_value, df) = HypothesisTest::wald_test(&beta, &cov_matrix, &r, &q).unwrap();

    // Check degrees of freedom
    assert_eq!(df, 1);

    // Wald statistic should be positive
    assert!(wald_stat > 0.0);

    // P-value should be between 0 and 1
    assert!((0.0..=1.0).contains(&p_value));

    // For β₁ = 3.5 with SE = 0.4, should reject H0: β₁ = 0
    // Wald = (3.5 - 0)² / 0.16 = 76.5625
    assert!((wald_stat - 76.5625).abs() < 0.01);
    assert!(p_value < 0.01); // Strong rejection
}

#[test]
fn test_wald_test_joint_hypothesis() {
    // Test H0: β₁ = 0 AND β₂ = 0
    let beta = Array1::from(vec![1.0, 2.0, 3.0]); // Intercept + 2 slopes
    let cov_matrix =
        Array2::from_shape_vec((3, 3), vec![0.25, 0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.0, 0.09])
            .unwrap();

    // Test both slopes = 0
    let r = Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let q = Array1::from(vec![0.0, 0.0]);

    let (wald_stat, p_value, df) = HypothesisTest::wald_test(&beta, &cov_matrix, &r, &q).unwrap();

    assert_eq!(df, 2); // Testing 2 restrictions
    assert!(wald_stat > 0.0);
    assert!((0.0..=1.0).contains(&p_value));
}

#[test]
fn test_joint_significance_test() {
    let beta = Array1::from(vec![1.5, 2.5, 3.5]); // Intercept + 2 slopes
    let cov_matrix =
        Array2::from_shape_vec((3, 3), vec![0.25, 0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.0, 0.09])
            .unwrap();

    let (wald_stat, p_value, df) =
        HypothesisTest::joint_significance(&beta, &cov_matrix, true).unwrap();

    // Testing 2 slopes (excluding intercept)
    assert_eq!(df, 2);
    assert!(wald_stat > 0.0);
    assert!((0.0..=1.0).contains(&p_value));

    // With large coefficients and small SEs, should strongly reject
    assert!(p_value < 0.01);
}

#[test]
fn test_f_test_nested_models() {
    // Restricted model: SSR = 150, k = 2
    // Full model: SSR = 100, k = 4
    // n = 50

    let ssr_r = 150.0;
    let ssr_f = 100.0;
    let n = 50;
    let k_f = 4;
    let k_r = 2;

    let (f_stat, p_value, df_num, df_denom) =
        HypothesisTest::f_test_nested(ssr_r, ssr_f, n, k_f, k_r).unwrap();

    // Check degrees of freedom
    assert_eq!(df_num, 2); // k_f - k_r = 4 - 2
    assert_eq!(df_denom, 46); // n - k_f = 50 - 4

    // F should be positive
    assert!(f_stat > 0.0);

    // P-value should be valid
    assert!((0.0..=1.0).contains(&p_value));

    // Manual calculation: F = ((150-100)/2) / (100/46) = 25 / 2.174 ≈ 11.5
    let expected_f = ((ssr_r - ssr_f) / (k_f - k_r) as f64) / (ssr_f / (n - k_f) as f64);
    assert!((f_stat - expected_f).abs() < 0.01);
}

#[test]
fn test_f_test_no_improvement() {
    // Test where full model doesn't improve fit (SSR same)
    let ssr_r = 100.0;
    let ssr_f = 100.0;
    let n = 50;
    let k_f = 4;
    let k_r = 2;

    let (f_stat, p_value, _, _) = HypothesisTest::f_test_nested(ssr_r, ssr_f, n, k_f, k_r).unwrap();

    // F should be 0 (no improvement)
    assert!(f_stat.abs() < 1e-10);

    // P-value should be close to 1 (fail to reject)
    assert!(p_value > 0.95);
}

#[test]
fn test_wald_test_specific_value() {
    // Test H0: β₁ = 5 (not zero)
    let beta = Array1::from(vec![2.0, 5.2]);
    let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.25, 0.0, 0.0, 0.04]).unwrap();

    // Test β₁ = 5
    let r = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
    let q = Array1::from(vec![5.0]);

    let (wald_stat, p_value, _) = HypothesisTest::wald_test(&beta, &cov_matrix, &r, &q).unwrap();

    // β₁ = 5.2 is close to 5, so should not reject strongly
    // Wald = (5.2 - 5)² / 0.04 = 0.04 / 0.04 = 1.0
    assert!((wald_stat - 1.0).abs() < 0.01);
    assert!(p_value > 0.10); // Should not reject at 10% level
}

#[test]
fn test_bootstrap_reproduces_coefficients() {
    // With many bootstrap samples, mean should be close to original estimate
    let y = Array1::from(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();

    let boot_coefs = Bootstrap::pairs_bootstrap(&y, &x, 5000).unwrap();

    // Calculate mean of bootstrap coefficients
    let mean_intercept: f64 = boot_coefs.column(0).mean().unwrap();
    let mean_slope: f64 = boot_coefs.column(1).mean().unwrap();

    // True values: intercept = 1, slope = 2
    // Bootstrap means should be close (within 10%)
    assert!((mean_intercept - 1.0).abs() < 0.2);
    assert!((mean_slope - 2.0).abs() < 0.2);
}
