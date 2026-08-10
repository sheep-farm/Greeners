use greeners::{CoxPH, KaplanMeier};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Exp;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Kaplan-Meier survival probabilities are monotone, in [0,1] and match the
/// product-limit formula for uncensored data.
#[test]
fn test_kaplan_meier_no_censoring() {
    let n = 10;
    let times = Array1::from_vec((1..=n).map(|i| i as f64).collect());
    let events = Array1::from_vec(vec![1u8; n]);

    let result = KaplanMeier::fit(&times, &events).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_events, n);
    assert_eq!(result.times.len(), n);

    for j in 0..result.times.len() {
        let expected = (n - j - 1) as f64 / n as f64;
        approx_zero((result.survival_probs[j] - expected).abs(), 1e-10);
        assert!(result.survival_probs[j] >= 0.0 && result.survival_probs[j] <= 1.0);
        assert!(result.conf_lower[j] >= -1e-10);
        assert!(result.conf_upper[j] <= 1.0 + 1e-10);
    }

    // Survival is non-increasing.
    for j in 1..result.survival_probs.len() {
        assert!(result.survival_probs[j] <= result.survival_probs[j - 1]);
    }

    // Median survival for this data is 5 (first time where S <= 0.5).
    approx_zero((result.median_survival - 5.0).abs(), 1e-10);
}

/// Kaplan-Meier with censoring: censored observations do not decrease the
/// survival estimate at their time.
#[test]
fn test_kaplan_meier_with_censoring() {
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let events = Array1::from_vec(vec![1u8, 1, 0, 1, 1, 0, 1, 1, 1, 1]);

    let result = KaplanMeier::fit(&times, &events).unwrap();
    assert_eq!(result.n_events, 8);
    assert!(result.survival_probs.iter().all(|&v| v >= 0.0 && v <= 1.0));
    for j in 1..result.survival_probs.len() {
        assert!(result.survival_probs[j] <= result.survival_probs[j - 1]);
    }
}

/// Cox PH recovers the log hazard ratio on simulated exponential survival data.
#[test]
fn test_cox_ph_recovery() {
    let n = 200;
    let mut rng = StdRng::seed_from_u64(6001);
    let exp = Exp::new(1.0).unwrap();

    let true_beta = 0.5;

    let mut times = Vec::with_capacity(n);
    let mut events = Vec::with_capacity(n);
    let mut x_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i % 2) as f64;
        x_vec.push(x);
        let rate = (true_beta * x).exp();
        times.push(exp.sample(&mut rng) / rate);
        events.push(1u8);
    }
    let times = Array1::from_vec(times);
    let events = Array1::from_vec(events);
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();

    let result = CoxPH::fit(&times, &events, &x).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - true_beta).abs(), 0.3);
    approx_zero((result.hazard_ratios[0] - true_beta.exp()).abs(), 0.2);
    assert!(result.log_likelihood.is_finite());
    assert!(result.concordance > 0.5 && result.concordance <= 1.0);
    assert!(result.n_events > 0);
}

/// Input validation.
#[test]
fn test_survival_input_validation() {
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let events = Array1::from_vec(vec![1u8, 0]);
    assert!(KaplanMeier::fit(&times, &events).is_err());

    let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
    assert!(CoxPH::fit(&times, &events, &x).is_err());

    let no_events = Array1::from_vec(vec![0u8, 0, 0]);
    let x3 = Array2::from_shape_vec((3, 1), vec![1.0; 3]).unwrap();
    assert!(CoxPH::fit(&times, &no_events, &x3).is_err());
}
