use greeners_timeseries::setar::SETAR;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn generate_setar_data(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.3).unwrap();
    let mut y = vec![0.0; n];
    for i in 1..n {
        let y_lag = y[i - 1];
        let y_t = if y_lag <= 0.0 {
            0.5 * y_lag
        } else {
            -0.3 * y_lag
        };
        y[i] = y_t + noise.sample(&mut rng);
    }
    Array1::from_vec(y)
}

#[test]
fn test_setar_runs_and_produces_finite_output() {
    let y = generate_setar_data(300, 13001);

    let result = SETAR::fit(&y, 1, 1).unwrap();

    assert_eq!(result.ar_order, 1);
    assert_eq!(result.delay, 1);
    assert_eq!(result.beta_low.len(), 2);
    assert_eq!(result.beta_high.len(), 2);
    assert_eq!(result.se_low.len(), 2);
    assert_eq!(result.se_high.len(), 2);
    assert_eq!(result.t_low.len(), 2);
    assert_eq!(result.t_high.len(), 2);
    assert_eq!(result.p_low.len(), 2);
    assert_eq!(result.p_high.len(), 2);
    assert!(result.threshold.is_finite());
    assert!(result.n_low > 0);
    assert!(result.n_high > 0);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert!(result.sigma.is_finite() && result.sigma > 0.0);
    assert!(result.log_likelihood.is_finite());
    assert!(result.beta_low[0].is_finite());
    assert!(result.beta_high[0].is_finite());

    // Both regimes together should account for all effective observations.
    assert_eq!(result.n_low + result.n_high, result.n_obs);
}

#[test]
fn test_setar_recovers_threshold_and_regime_coefficients() {
    let y = generate_setar_data(400, 13002);

    let result = SETAR::fit(&y, 1, 1).unwrap();

    // Threshold should be near the data threshold of 0 (with sampling variation).
    assert!(result.threshold.abs() < 1.0);

    // Low-regime AR coefficient should be positive and high-regime negative.
    assert!(result.beta_low[1] > 0.0 && result.beta_low[1] < 0.8);
    assert!(result.beta_high[1] > -0.6 && result.beta_high[1] < 0.0);

    // R-squared should be non-trivially positive for this deterministic DGP.
    assert!(result.r_squared > 0.15);
}

#[test]
fn test_setar_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Too few observations for ar_order=2, delay=2.
    assert!(SETAR::fit(&y, 2, 2).is_err());

    // ar_order=1, delay=1 with n < (ar_order + delay) * 3.
    let short = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    assert!(SETAR::fit(&short, 1, 1).is_err());
}
