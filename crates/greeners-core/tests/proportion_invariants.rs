use greeners_core::proportion::ProportionTests;

/// One-sample z-test recovers the expected value for a simple case.
#[test]
fn test_proportion_ztest_1samp_invariants() {
    let (z, p) = ProportionTests::proportions_ztest_1samp(60, 100, 0.5).unwrap();
    assert!(z.is_finite());
    assert!(p >= 0.0 && p <= 1.0);
    assert!((z - 2.0).abs() < 1e-6);
}

/// Two-sample z-test and confidence intervals return finite, ordered results.
#[test]
fn test_proportion_confint_and_2samp_invariants() {
    let (z, p) = ProportionTests::proportions_ztest_2samp(45, 100, 55, 100).unwrap();
    assert!(z.is_finite());
    assert!(p >= 0.0 && p <= 1.0);

    let (lower, upper) = ProportionTests::proportion_confint(30, 100, 0.05).unwrap();
    assert!(lower.is_finite());
    assert!(upper.is_finite());
    assert!(lower >= 0.0);
    assert!(upper <= 1.0);
    assert!(lower < upper);
}

/// Contingency table chi-square is consistent with a manual computation.
#[test]
fn test_proportion_chi2_invariants() {
    let table = [[10, 20], [30, 40]];
    let (chi2, p) = ProportionTests::chi2_contingency(&table).unwrap();
    assert!(chi2.is_finite());
    assert!(p >= 0.0 && p <= 1.0);
    assert!((chi2 - 0.7937).abs() < 1e-3);
}
