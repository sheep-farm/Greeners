use greeners_timeseries::markov_autoreg::MarkovAutoregression;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn generate_markov_ar(seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.4).unwrap();
    let n = 250;
    let mut y = vec![0.0; n];

    // 2-regime AR(1): intercept 0 or 3, phi = 0.3, switch every 50 obs.
    let phi = 0.3;
    let mut regime = 0;
    for i in 1..n {
        if i % 50 == 0 {
            regime = 1 - regime;
        }
        let mu = if regime == 0 { 0.0 } else { 3.0 };
        y[i] = mu + phi * y[i - 1] + noise.sample(&mut rng);
    }
    Array1::from_vec(y)
}

#[test]
fn test_markov_autoreg_runs_and_produces_finite_output() {
    let y = generate_markov_ar(12001);

    let result = MarkovAutoregression::fit(&y, 2, 1).unwrap();

    assert_eq!(result.k_regimes, 2);
    assert_eq!(result.ar_order, 1);
    assert_eq!(result.n_obs, y.len() - 1);
    assert_eq!(result.regime_means.len(), 2);
    assert_eq!(result.ar_params.shape(), &[2, 1]);
    assert_eq!(result.regime_sigmas.len(), 2);
    assert_eq!(result.transition_matrix.shape(), &[2, 2]);
    assert_eq!(result.smoothed_probs.shape(), &[result.n_obs, 2]);
    assert_eq!(result.filtered_probs.shape(), &[result.n_obs, 2]);
    assert!(result.regime_means.iter().all(|&v| v.is_finite()));
    assert!(result.ar_params.iter().all(|&v| v.is_finite()));
    assert!(result.log_likelihood.is_finite());

    // Transition matrix rows should be probabilities.
    for i in 0..2 {
        let s: f64 = result.transition_matrix.row(i).sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(result.transition_matrix.row(i).iter().all(|&v| v >= 0.0));
    }
}

#[test]
fn test_markov_autoreg_separates_regimes_and_predicts() {
    let y = generate_markov_ar(12002);

    let result = MarkovAutoregression::fit(&y, 2, 1).unwrap();

    // AR parameters should be stable.
    assert!(result.ar_params[[0, 0]].abs() < 1.0);
    assert!(result.ar_params[[1, 0]].abs() < 1.0);

    // Regime means should be ordered: the low-regime mean is below the high-regime mean.
    assert!(result.regime_means[0] < result.regime_means[1]);

    // Predicted regime labels should be in {0, 1} and alternate.
    let regimes = result.predict_regime();
    assert_eq!(regimes.len(), result.n_obs);
    assert!(regimes.iter().all(|&v| v == 0 || v == 1));
    assert!(regimes.iter().any(|&v| v == 0));
    assert!(regimes.iter().any(|&v| v == 1));
}

#[test]
fn test_markov_autoreg_input_validation() {
    let y = Array1::from(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
    ]);

    // Too few regimes.
    assert!(MarkovAutoregression::fit(&y, 1, 1).is_err());

    // Too short for p+10.
    let short = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(MarkovAutoregression::fit(&short, 2, 1).is_err());
}
