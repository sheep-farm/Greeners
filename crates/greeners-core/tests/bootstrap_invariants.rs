use greeners_core::bootstrap::Bootstrap;
use greeners_core::bootstrap::HypothesisTest;
use ndarray::{Array1, Array2};

/// Pairs bootstrap yields finite coefficients and ordered percentile intervals.
#[test]
fn test_bootstrap_pairs_invariants() {
    let n = 30;
    let x: Array2<f64> =
        Array2::from_shape_vec((n, 2), (0..n).flat_map(|i| vec![1.0, i as f64]).collect()).unwrap();
    let y: Array1<f64> = x.column(1).mapv(|v| 1.0 + 2.0 * v);

    let n_boot = 50;
    let boot = Bootstrap::pairs_bootstrap(&y, &x, n_boot).unwrap();

    assert_eq!(boot.shape(), [n_boot, 2]);
    assert!(boot.iter().all(|v| v.is_finite()));

    let se = Bootstrap::bootstrap_se(&boot);
    assert_eq!(se.len(), 2);
    assert!(se.iter().all(|v| v.is_finite() && *v >= 0.0));

    let (lower, upper) = Bootstrap::percentile_ci(&boot, 0.05);
    assert_eq!(lower.len(), 2);
    assert_eq!(upper.len(), 2);
    for j in 0..2 {
        assert!(lower[j].is_finite());
        assert!(upper[j].is_finite());
        assert!(lower[j] <= upper[j]);
    }
}

/// Hypothesis tests return finite statistics and valid p-values.
#[test]
fn test_hypothesis_tests_invariants() {
    let beta = Array1::from_vec(vec![0.5, 2.0, 0.0]);
    let cov = Array2::eye(3);
    let r = Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let q = Array1::zeros(2);

    let (wald_stat, p_value, df) = HypothesisTest::wald_test(&beta, &cov, &r, &q).unwrap();
    assert!(wald_stat.is_finite());
    assert!(p_value >= 0.0 && p_value <= 1.0);
    assert_eq!(df, 2);

    let (f_stat, p, df_num, df_denom) =
        HypothesisTest::f_test_nested(120.0, 80.0, 20, 3, 1).unwrap();
    assert!(f_stat.is_finite());
    assert!(p >= 0.0 && p <= 1.0);
    assert!(df_num > 0);
    assert!(df_denom > 0);

    let (joint_stat, joint_p, joint_df) =
        HypothesisTest::joint_significance(&beta, &cov, true).unwrap();
    assert!(joint_stat.is_finite());
    assert!(joint_p >= 0.0 && joint_p <= 1.0);
    assert_eq!(joint_df, 2);
}

/// Input validation rejects mismatched shapes and invalid model comparisons.
#[test]
fn test_bootstrap_input_validation() {
    let y = Array1::from_vec(vec![1.0; 5]);
    let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
    assert!(Bootstrap::pairs_bootstrap(&y, &x, 10).is_err());

    // Singular covariance matrix makes the Wald test non-invertible
    let beta = Array1::from_vec(vec![0.5, 2.0, 0.0]);
    let cov_zero = Array2::zeros((3, 3));
    let r = Array2::eye(3);
    let q = Array1::zeros(3);
    assert!(HypothesisTest::wald_test(&beta, &cov_zero, &r, &q).is_err());

    // Nested models with the same number of parameters are invalid
    assert!(HypothesisTest::f_test_nested(100.0, 90.0, 20, 2, 2).is_err());

    // No slope coefficients to test when only an intercept is present
    let b0 = Array1::from_vec(vec![1.0]);
    let c0 = Array2::eye(1);
    assert!(HypothesisTest::joint_significance(&b0, &c0, true).is_err());
}
