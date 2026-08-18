use greeners::Stats;
use ndarray::Array1;

/// CompareMeans returns finite statistics and the expected sign of the mean difference.
#[test]
fn test_compare_means_invariants() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0]);
    let r = Stats::compare_means(&a, &b, true).unwrap();
    assert!(r.mean1.is_finite());
    assert!(r.mean2.is_finite());
    assert!(r.diff.is_finite());
    assert!(r.t_statistic.is_finite());
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    assert!((r.diff - (r.mean1 - r.mean2)).abs() < 1e-10);
    assert!(r.diff < 0.0);
    assert!(r.n1 == 5);
    assert!(r.n2 == 5);
}

/// TTest 1-sample and 2-sample functions return consistent full results.
#[test]
fn test_ttest_full_invariants() {
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let r = Stats::ttest_1samp_full(&data, 3.0).unwrap();
    assert!(r.t_statistic.is_finite());
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    assert!(r.mean.is_finite());
    assert_eq!(r.n, 5);

    let data2 = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    let (t, p) = Stats::ttest_ind(&data, &data2, true).unwrap();
    assert!(t.is_finite());
    assert!(p >= 0.0 && p <= 1.0);
}

/// Input validation rejects too few observations or mismatched paired data.
#[test]
fn test_stats_input_validation() {
    let a = Array1::from_vec(vec![1.0]);
    let b = Array1::from_vec(vec![2.0]);
    assert!(Stats::compare_means(&a, &b, true).is_err());

    let a2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b2 = Array1::from_vec(vec![1.0, 2.0]);
    assert!(Stats::ttest_paired(&a2, &b2).is_err());
}
