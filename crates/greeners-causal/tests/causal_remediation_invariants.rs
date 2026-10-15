use greeners_causal::{
    dr_learner::DRLearner,
    rd::{RdKernel, RD},
};
use ndarray::{array, Array1, Array2};

#[test]
fn dr_correct_outcomes_recover_ate_with_misspecified_propensity() {
    let n = 30_000;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| (i / 10_000) as f64 - 1.0);
    let d: Vec<bool> = (0..n)
        .map(|i| i % 10_000 < [2000, 8000, 8000][i / 10_000])
        .collect();
    let y = Array1::from_shape_fn(n, |i| if d[i] { 1.0 + x[(i, 0)] } else { 0.0 });
    let r = DRLearner::fit(&y, &d, &x, None, None).unwrap();
    assert!((r.ate - 1.0).abs() < 1e-10, "ATE {}", r.ate);
    for i in 0..n {
        assert!((r.cate[i] - (1.0 + x[(i, 0)])).abs() < 1e-10);
        assert!((r.outcome_reg[i] - y[i]).abs() < 1e-10);
    }
}

#[test]
fn dr_rejects_unidentified_or_nonfinite_nuisances() {
    let x = Array2::ones((60, 1));
    let d: Vec<bool> = (0..60).map(|i| i % 2 == 0).collect();
    let y = Array1::from_shape_fn(60, |i| i as f64);
    assert!(DRLearner::fit(&y, &d, &x, None, None).is_err());
    let x = Array2::from_shape_fn((60, 1), |(i, _)| i as f64);
    let mut bad = y.clone();
    bad[1] = f64::NAN;
    assert!(DRLearner::fit(&bad, &d, &x, None, None).is_err());
}

#[test]
fn dr_folds_are_reproducible_independent_of_call_order() {
    let x = Array2::from_shape_fn((90, 1), |(i, _)| i as f64 / 90.0);
    let d: Vec<bool> = (0..90).map(|i| i % 3 == 0).collect();
    let y = Array1::from_shape_fn(90, |i| x[(i, 0)].powi(2) + if d[i] { 2.0 } else { 0.0 });
    let first = DRLearner::fit(&y, &d, &x, None, None).unwrap();
    let _ = DRLearner::fit(&y.mapv(|v| v * 2.0), &d, &x, Some(2), None).unwrap();
    let again = DRLearner::fit(&y, &d, &x, None, None).unwrap();
    assert_eq!(first.ate, again.ate);
    assert_eq!(first.propensity, again.propensity);
}

#[test]
fn dr_correct_bounded_propensity_recovers_ate_with_nonlinear_outcome() {
    // Balanced deterministic support: true propensity is 1/2 and the outcome
    // contains x², outside the linear nuisance model. True treatment effect=2.
    let n = 10_100;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| (i % 101) as f64 / 50.0 - 1.0);
    let d: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let y = Array1::from_shape_fn(n, |i| x[(i, 0)].powi(2) + if d[i] { 2.0 } else { 0.0 });
    let r = DRLearner::fit(&y, &d, &x, None, None).unwrap();
    // Finite cross-fit nuisance error is allowed; this is a point-recovery check,
    // not a claim of misspecification-robust interval coverage.
    assert!((r.ate - 2.0).abs() < 0.02, "{}", r.ate);
    assert!(r.propensity.iter().all(|&v| v > 0.4 && v < 0.6));
    let linear_y = Array1::from_shape_fn(n, |i| 1.0 + x[(i, 0)] + if d[i] { 2.0 } else { 0.0 });
    let both_correct = DRLearner::fit(&linear_y, &d, &x, None, None).unwrap();
    assert!((both_correct.ate - 2.0).abs() < 1e-10);
}

#[test]
fn dr_correct_outcomes_remain_valid_with_active_propensity_clipping() {
    // True propensities .005 and .995 preserve positivity but lie outside the
    // truncation bounds. This tests the outcome-correct branch only.
    let n = 20_000;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| if i < 10_000 { -1.0 } else { 1.0 });
    let d: Vec<bool> = (0..n)
        .map(|i| i % 10_000 < if i < 10_000 { 50 } else { 9950 })
        .collect();
    let y = Array1::from_shape_fn(n, |i| {
        1.0 + x[(i, 0)] + if d[i] { 2.0 + 0.2 * x[(i, 0)] } else { 0.0 }
    });
    let r = DRLearner::fit(&y, &d, &x, None, None).unwrap();
    assert!(r.propensity.iter().any(|&v| v == 0.01));
    assert!(r.propensity.iter().any(|&v| v == 0.99));
    assert!((r.ate - 2.0).abs() < 1e-9);
}

#[test]
fn dr_standard_error_respects_very_small_and_large_outcome_units() {
    let x = Array2::from_shape_fn((90, 1), |(i, _)| i as f64 / 90.0);
    let d: Vec<bool> = (0..90).map(|i| i % 3 == 0).collect();
    let y = Array1::from_shape_fn(90, |i| x[(i, 0)].powi(2) + if d[i] { 2.0 } else { 0.0 });
    let reference = DRLearner::fit(&y, &d, &x, None, None).unwrap();
    assert!(reference.ate_se > 0.0);
    for scale in [1e-200, 1e200] {
        let r = DRLearner::fit(&y.mapv(|v| v * scale), &d, &x, None, None).unwrap();
        assert!((r.ate_se / (reference.ate_se * scale) - 1.0).abs() < 1e-10);
        assert!((r.ate / (reference.ate * scale) - 1.0).abs() < 1e-10);
    }
}

#[test]
fn fuzzy_rd_ratio_covariance_cancels_for_y_equal_c_d() {
    let x = array![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    let d = array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    for c in [1.0, -2.0, 0.0] {
        let r = RD::fit_fuzzy(
            &d.mapv(|v| v * c),
            &d,
            &x,
            0.0,
            Some(5.0),
            1,
            RdKernel::Uniform,
            None,
        )
        .unwrap();
        assert!((r.tau - c).abs() < 1e-12);
        assert!(r.se < 1e-12, "SE {} for c={c}", r.se);
        if c == 0.0 {
            assert_eq!(r.z, 0.0);
            assert_eq!(r.p_value, 1.0);
        }
    }
}

// Independent NumPy joint HC1 sandwich with triangular weights, n_side/(n_side-2).
#[test]
fn fuzzy_rd_matches_joint_sandwich_for_either_covariance_sign() {
    let x = array![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    let d = array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let y = array![1.0, -1.0, 2.0, 0.0, 4.0, 1.0, 5.0, 3.0];
    for sign in [-1.0, 1.0] {
        let r = RD::fit_fuzzy(
            &y.mapv(|v| v * sign),
            &d,
            &x,
            0.0,
            Some(5.0),
            1,
            RdKernel::Triangular,
            None,
        )
        .unwrap();
        assert!((r.tau - 3.0 * sign).abs() < 1e-12);
        assert!((r.se.powi(2) - 4.506666666666657).abs() < 1e-10, "{}", r.se);
    }
}

#[test]
fn fuzzy_rd_stage_guard_respects_treatment_units() {
    let x = array![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    let d = array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let r = RD::fit_fuzzy(
        &d,
        &d.mapv(|v| v * 1e-12),
        &x,
        0.0,
        Some(5.0),
        1,
        RdKernel::Uniform,
        None,
    )
    .unwrap();
    assert!((r.tau / 1e12 - 1.0).abs() < 1e-12);
    assert!(RD::fit_fuzzy(
        &d,
        &Array1::ones(8),
        &x,
        0.0,
        Some(5.0),
        1,
        RdKernel::Uniform,
        None
    )
    .is_err());
}

#[test]
fn fuzzy_stage_guard_distinguishes_numerical_resolution_from_strength() {
    let x = array![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    for jump in [1e-8, 1e-15] {
        let d = x.mapv(|v| if v < 0.0 { 0.5 } else { 0.5 + jump });
        let fit = RD::fit_fuzzy(&d, &d, &x, 0.0, Some(5.0), 1, RdKernel::Uniform, None);
        if jump > 1e-10 {
            assert!((fit.unwrap().tau - 1.0).abs() < 1e-12);
        } else {
            assert!(fit.is_err());
        }
    }
}

#[test]
fn rd_preserves_sharp_hc1_and_rejects_rank_deficient_local_design() {
    let x = array![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let r = RD::fit(&y, &x, 0.0, Some(5.0), 1, RdKernel::Uniform, None).unwrap();
    // Independent original-scale OLS/HC1 on each side: jump=.5, variance=.69.
    assert!((r.tau - 0.5).abs() < 1e-12);
    assert!((r.se.powi(2) - 0.69).abs() < 1e-12);
    let repeated_x = array![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
    assert!(RD::fit(&y, &repeated_x, 0.0, Some(5.0), 1, RdKernel::Uniform, None).is_err());
    assert!(RD::fit(&y, &x, f64::NAN, Some(5.0), 1, RdKernel::Uniform, None).is_err());
}

#[test]
fn fuzzy_rd_rejects_unrepresentable_final_interval() {
    let x = array![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    let d = array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0].mapv(|v| v * 3.9e-158);
    let y = array![1.0, -1.0, 2.0, 0.0, 4.0, 1.0, 5.0, 3.0].mapv(|v| v * 1e150);
    assert!(RD::fit_fuzzy(&y, &d, &x, 0.0, Some(5.0), 1, RdKernel::Triangular, None).is_err());
}

#[test]
fn rd_rejects_zero_residual_degrees_of_freedom_and_invalid_bandwidth() {
    let x = array![-2.0, -1.0, 1.0, 2.0];
    let y = array![0.0, 1.0, 2.0, 3.0];
    assert!(RD::fit(&y, &x, 0.0, Some(3.0), 1, RdKernel::Uniform, None).is_err());
    for h in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(RD::fit(&y, &x, 0.0, Some(h), 0, RdKernel::Uniform, None).is_err());
    }
}
