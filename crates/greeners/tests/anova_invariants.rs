use greeners::Stats;
use ndarray::Array1;

/// One-way ANOVA returns a valid decomposition of sums of squares.
#[test]
fn test_anova_oneway_invariants() {
    let data: Array1<f64> = (0..15)
        .map(|i| (i / 5) as f64 * 2.0 + (i % 5) as f64 * 0.3)
        .collect();
    let groups: Array1<usize> = (0..15).map(|i| i / 5).collect();

    let r = Stats::anova_oneway(&data, &groups).unwrap();
    assert_eq!(r.n_obs, 15);
    assert_eq!(r.n_groups, 3);
    assert_eq!(r.df_between, 2);
    assert_eq!(r.df_within, 12);
    assert!((r.ss_total - (r.ss_between + r.ss_within)).abs() < 1e-10);
    assert!(r.f_statistic.is_finite());
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    assert!(r.ms_between >= 0.0);
    assert!(r.ms_within >= 0.0);
}

/// ANOVA for regression decomposes total variation into model and residual parts.
#[test]
fn test_anova_regression_invariants() {
    let n = 20;
    let x: Array1<f64> = (0..n).map(|i| i as f64).collect();
    let y = x.mapv(|v| 1.0 + 2.0 * v);
    // Fitted perfectly by the linear model: residuals are negligible
    let residuals = Array1::zeros(n);

    let r = Stats::anova_regression(&y, &residuals, 1).unwrap();
    assert!((r.ss_total - (r.ss_model + r.ss_resid)).abs() < 1e-10);
    assert_eq!(r.df_model, 1);
    assert_eq!(r.df_resid, n - 2);
    assert!(r.f_statistic.is_finite());
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    assert!(r.ms_model >= 0.0);
    assert!(r.ms_resid >= 0.0);
}

/// Input validation rejects mismatched dimensions, too few groups and empty data.
#[test]
fn test_anova_input_validation() {
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let groups_bad = Array1::from_vec(vec![0; 4]);
    assert!(Stats::anova_oneway(&data, &groups_bad).is_err());

    let one_group = Array1::from_vec(vec![0; 5]);
    assert!(Stats::anova_oneway(&data, &one_group).is_err());

    let empty_data = Array1::<f64>::from_vec(vec![]);
    let empty_groups = Array1::<usize>::from_vec(vec![]);
    assert!(Stats::anova_oneway(&empty_data, &empty_groups).is_err());
}
