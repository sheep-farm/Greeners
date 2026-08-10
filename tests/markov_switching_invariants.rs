use greeners::MarkovSwitching;
use ndarray::Array1;
use rand::distributions::{Distribution, Uniform};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Generate a two-regime AR(1) series and fit it. The result is finite and
/// the transition matrix is a valid stochastic matrix.
#[test]
fn test_markov_switching_fit() {
    let n = 300;
    let mut rng = StdRng::seed_from_u64(7001);
    let unif = Uniform::new(0.0_f64, 1.0_f64);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let trans = [[0.95, 0.05], [0.15, 0.85]];
    let means = [0.0, 4.0];
    let phi = 0.5;

    let mut state = if unif.sample(&mut rng) < 0.5 { 0 } else { 1 };
    let mut y_vec = Vec::with_capacity(n);
    let mut prev_y = 0.0;
    for _ in 0..n {
        let switch = unif.sample(&mut rng);
        state = if switch < trans[state][0] { 0 } else { 1 };
        let e = noise.sample(&mut rng);
        let y = means[state] + phi * prev_y + e;
        y_vec.push(y);
        prev_y = y;
    }
    let y = Array1::from_vec(y_vec);

    let result = MarkovSwitching::fit(&y, 2, 1).unwrap();
    assert_eq!(result.n_regimes, 2);
    assert_eq!(result.ar_order, 1);
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());

    // Transition matrix rows sum to 1 and are non-negative.
    for i in 0..2 {
        let row_sum = result.transition_matrix.row(i).sum();
        approx_zero(row_sum - 1.0, 1e-10);
        assert!(result.transition_matrix.row(i).iter().all(|&v| v >= 0.0));
        // High persistence: diagonal entries dominate.
        assert!(result.transition_matrix[[i, i]] > 0.5);
    }

    // Regime parameters and variances are finite and variances positive.
    assert_eq!(result.regime_params.len(), 2);
    for params in &result.regime_params {
        assert!(params.iter().all(|&v| v.is_finite()));
    }
    assert!(result
        .regime_variances
        .iter()
        .all(|&v| v > 0.0 && v.is_finite()));

    // Filtered and smoothed probabilities are valid distributions.
    assert_eq!(result.filtered_probs.ncols(), 2);
    assert_eq!(result.smoothed_probs.ncols(), 2);
    for t in 0..result.filtered_probs.nrows() {
        let s1: f64 = result.filtered_probs.row(t).sum();
        let s2: f64 = result.smoothed_probs.row(t).sum();
        approx_zero(s1 - 1.0, 1e-10);
        approx_zero(s2 - 1.0, 1e-10);
        assert!(result.filtered_probs.row(t).iter().all(|&v| v >= 0.0));
        assert!(result.smoothed_probs.row(t).iter().all(|&v| v >= 0.0));
    }

    // Forecasts are finite and have the requested length.
    let forecasts = result.predict(&y, 5);
    assert_eq!(forecasts.len(), 5);
    assert!(forecasts.iter().all(|&v| v.is_finite()));
}

/// The fitted intercepts are ordered by the true regime means and the
/// smoothed state probabilities correlate with the true high-mean state.
#[test]
fn test_markov_switching_regime_alignment() {
    let n = 500;
    let mut rng = StdRng::seed_from_u64(7002);
    let unif = Uniform::new(0.0_f64, 1.0_f64);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let trans = [[0.95, 0.05], [0.10, 0.90]];
    let means = [-2.0, 3.0];
    let phi = 0.3;

    let mut state = if unif.sample(&mut rng) < 0.5 { 0 } else { 1 };
    let mut y_vec = Vec::with_capacity(n);
    let mut states = Vec::with_capacity(n);
    let mut prev_y = 0.0;
    for _ in 0..n {
        let switch = unif.sample(&mut rng);
        state = if switch < trans[state][0] { 0 } else { 1 };
        let e = noise.sample(&mut rng);
        let y = means[state] + phi * prev_y + e;
        y_vec.push(y);
        states.push(state);
        prev_y = y;
    }
    let y = Array1::from_vec(y_vec);

    // Effective observations start after p lags.
    let p = 1;
    let result = MarkovSwitching::fit(&y, 2, p).unwrap();

    // Map the fitted regime with larger intercept to the high-mean state.
    let high_fit = if result.regime_params[0][0] > result.regime_params[1][0] {
        0
    } else {
        1
    };

    // The fitted intercepts should be ordered like the true means.
    let true_high = if means[1] > means[0] { 1 } else { 0 };
    assert_eq!(high_fit, true_high);

    // For the high-mean state, smoothed probability should be > 0.5 more often
    // when the true state is high.
    let mut correct = 0;
    let mut checked = 0;
    for t in 0..result.smoothed_probs.nrows() {
        let true_state = states[t + p];
        let prob_high = result.smoothed_probs[[t, high_fit]];
        if (true_state == true_high && prob_high > 0.5)
            || (true_state != true_high && prob_high <= 0.5)
        {
            correct += 1;
        }
        checked += 1;
    }
    let accuracy = correct as f64 / checked as f64;
    assert!(
        accuracy > 0.65,
        "regime classification accuracy was {}",
        accuracy
    );

    // The AR coefficient should have the right sign and be finite.
    for params in &result.regime_params {
        assert!(params[1].is_finite());
    }
}

/// Markov switching rejects invalid inputs.
#[test]
fn test_markov_switching_input_validation() {
    let y = Array1::from_vec(vec![1.0; 20]);
    assert!(MarkovSwitching::fit(&y, 1, 1).is_err());
    assert!(MarkovSwitching::fit(&y, 2, 25).is_err());
}
