use greeners::ProportionTests;

#[test]
fn test_1samp_ztest() {
    // 60 successes out of 100, test H0: p = 0.5
    let (z, pval) = ProportionTests::proportions_ztest_1samp(60, 100, 0.5).unwrap();

    // z = (0.6 - 0.5) / sqrt(0.5 * 0.5 / 100) = 0.1 / 0.05 = 2.0
    assert!((z - 2.0).abs() < 1e-10);
    // two-sided p-value for z=2.0 ~ 0.0455
    assert!((pval - 0.04550026).abs() < 1e-4);
}

#[test]
fn test_2samp_ztest() {
    // 45/100 vs 55/100
    let (z, pval) = ProportionTests::proportions_ztest_2samp(45, 100, 55, 100).unwrap();

    // pooled p = 100/200 = 0.5
    // se = sqrt(0.5 * 0.5 * (1/100 + 1/100)) = sqrt(0.005) ~ 0.07071
    // z = (0.45 - 0.55) / 0.07071 ~ -1.4142
    assert!((z - (-std::f64::consts::SQRT_2)).abs() < 1e-4);
    // p ~ 0.1573
    assert!((pval - 0.1573).abs() < 1e-3);
}

#[test]
fn test_wilson_confint() {
    // 30/100 at alpha=0.05
    let (lower, upper) = ProportionTests::proportion_confint(30, 100, 0.05).unwrap();

    // Wilson interval for p=0.3, n=100: roughly (0.217, 0.394)
    assert!(lower > 0.21 && lower < 0.23);
    assert!(upper > 0.39 && upper < 0.40);
    assert!(lower < upper);
}

#[test]
fn test_chi2_contingency() {
    // Table: [[10, 20], [30, 40]]
    let table = [[10, 20], [30, 40]];
    let (chi2, pval) = ProportionTests::chi2_contingency(&table).unwrap();

    // Expected: row_totals = [30, 70], col_totals = [40, 60], n = 100
    // E = [[12, 18], [28, 42]]
    // chi2 = (10-12)^2/12 + (20-18)^2/18 + (30-28)^2/28 + (40-42)^2/42
    //      = 4/12 + 4/18 + 4/28 + 4/42 = 0.3333 + 0.2222 + 0.1429 + 0.0952 = 0.7937
    assert!((chi2 - 0.7937).abs() < 1e-3);
    // p-value for chi2=0.7937, df=1 ~ 0.373
    assert!((pval - 0.373).abs() < 0.01);
}

#[test]
fn test_1samp_errors() {
    assert!(ProportionTests::proportions_ztest_1samp(0, 0, 0.5).is_err());
    assert!(ProportionTests::proportions_ztest_1samp(101, 100, 0.5).is_err());
    assert!(ProportionTests::proportions_ztest_1samp(50, 100, 0.0).is_err());
}
