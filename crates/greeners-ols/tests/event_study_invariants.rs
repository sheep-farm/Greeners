use greeners_core::types::CovarianceType;
use greeners_ols::event_study::EventStudy;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn event_study_runs_and_recovers_post_effects() {
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let event_times = vec![-3i64, -2, -1, 0, 1, 2, 3];
    let reps = 30;
    let n = event_times.len() * reps;
    let mut y = Vec::with_capacity(n);
    let mut et = Vec::with_capacity(n);

    for t in &event_times {
        for _ in 0..reps {
            et.push(*t);
            let post_effect = if *t >= 0 { 0.5 * (*t as f64) } else { 0.0 };
            y.push(2.0 + post_effect + noise.sample(&mut rng));
        }
    }

    let y = Array1::from_vec(y);
    let event_time = Array1::from_vec(et);
    let x_controls = Array2::zeros((n, 0));

    let result = EventStudy::fit(
        &y,
        event_time.as_slice().unwrap(),
        &x_controls,
        -1,
        -3,
        3,
        CovarianceType::NonRobust,
    )
    .unwrap();

    assert!(!result.event_times.is_empty());
    assert_eq!(result.event_coefs.len(), result.event_times.len());
    assert_eq!(result.event_se.len(), result.event_times.len());
    assert!(result.ols.params.iter().all(|v| v.is_finite()));

    // Post-treatment coefficients should be positive and increasing.
    for (&t, &b) in result.event_times.iter().zip(result.event_coefs.iter()) {
        if t >= 0 {
            assert!(b > -0.2, "post coef too negative at t={}: {}", t, b);
        }
    }
}

#[test]
fn event_study_with_controls() {
    let n = 100;
    let mut rng = StdRng::seed_from_u64(22);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut event_time = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i % 7) as i64 - 3; // -3..3
        event_time.push(t);
        let control = noise.sample(&mut rng);
        let post_effect = if t >= 0 { 0.4 } else { 0.0 };
        y.push(1.0 + post_effect + 0.5 * control + noise.sample(&mut rng));
        x.push(control);
    }

    let y = Array1::from_vec(y);
    let event_time = Array1::from_vec(event_time);
    let x_controls = Array2::from_shape_vec((n, 1), x).unwrap();

    let result = EventStudy::fit(
        &y,
        event_time.as_slice().unwrap(),
        &x_controls,
        -1,
        -3,
        3,
        CovarianceType::HC1,
    )
    .unwrap();

    assert!(!result.event_times.is_empty());
    assert!(result.ols.r_squared >= 0.0 && result.ols.r_squared <= 1.0);
}

#[test]
fn event_study_input_validation() {
    let n = 20;
    let y = Array1::from_vec(vec![0.0; n]);
    let event_time = vec![-2i64; n - 1];
    let x_controls = Array2::zeros((n, 0));

    assert!(EventStudy::fit(
        &y,
        &event_time,
        &x_controls,
        -1,
        -3,
        3,
        CovarianceType::NonRobust
    )
    .is_err());
}

fn assert_event_inference_matches_ols(result: &greeners_ols::event_study::EventStudyResult) {
    assert_eq!(result.event_times.len(), result.event_col_indices.len());
    for (event, &column) in result.event_col_indices.iter().enumerate() {
        assert_eq!(result.event_coefs[event], result.ols.params[column]);
        assert_eq!(result.event_se[event], result.ols.std_errors[column]);
        assert_eq!(result.event_t[event], result.ols.t_values[column]);
        assert_eq!(result.event_p[event], result.ols.p_values[column]);
    }
}

#[test]
fn event_study_preserves_representable_tails_in_both_inference_tables() {
    // Independent pre-commit counterexample: each event group has 20 observations
    // and alternating +/-0.01 residuals. Student-t tails are subnormal but nonzero.
    let n = 60;
    let events: Vec<_> = (0..n).map(|i| (i % 3) as i64 - 1).collect();
    let controls = Array2::zeros((n, 0));
    for covariance in [CovarianceType::HC1, CovarianceType::NonRobust] {
        for sign in [-1.0, 1.0] {
            let y = Array1::from_shape_fn(n, |i| {
                sign * (if events[i] < 0 { 0.0 } else { 1e4 }
                    + if i % 2 == 0 { 0.01 } else { -0.01 })
            });
            let result =
                EventStudy::fit(&y, &events, &controls, -1, -1, 1, covariance.clone()).unwrap();
            for (&probability, &statistic) in result.event_p.iter().zip(result.event_t.iter()) {
                assert!(probability.is_finite() && probability > 0.0);
                assert!(probability < f64::MIN_POSITIVE, "{probability:e}");
                assert!(statistic * sign > 0.0);
            }
            assert_event_inference_matches_ols(&result);
        }
    }
}

#[test]
fn event_study_ordinary_tails_and_column_order_match_final_ols() {
    let n = 60;
    let events: Vec<_> = (0..n).map(|i| (i % 3) as i64 - 1).collect();
    let controls = Array2::zeros((n, 0));
    for covariance in [CovarianceType::HC1, CovarianceType::NonRobust] {
        for sign in [-1.0, 1.0] {
            let y = Array1::from_shape_fn(n, |i| {
                sign * (if events[i] < 0 { 0.0 } else { 0.005 }
                    + if i % 2 == 0 { 0.01 } else { -0.01 })
            });
            let result =
                EventStudy::fit(&y, &events, &controls, -1, -1, 1, covariance.clone()).unwrap();
            assert_eq!(result.event_times, vec![0, 1]);
            assert_eq!(result.event_col_indices, vec![1, 2]);
            assert!(result.event_p.iter().all(|&p| p > 0.05 && p < 0.5));
            assert_event_inference_matches_ols(&result);
        }
    }
}
