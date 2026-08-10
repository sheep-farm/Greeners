use greeners::{state_space_estimate, KalmanFilter, KalmanSmoother, LocalLevel, StateSpaceModel};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_local_level_series(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut y = vec![noise.sample(&mut rng)];
    for _ in 1..n {
        let prev = y.last().copied().unwrap();
        y.push(prev + noise.sample(&mut rng));
    }
    y
}

fn make_random_walk_obs(seed: u64, n: usize) -> (Vec<Array1<f64>>, StateSpaceModel) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut state = 0.0;
    let mut obs = Vec::with_capacity(n);
    for _ in 0..n {
        state += noise.sample(&mut rng);
        let y_i = state + noise.sample(&mut rng);
        obs.push(Array1::from_vec(vec![y_i]));
    }

    let model = StateSpaceModel {
        h: Array2::from_elem((1, 1), 1.0),
        f: Array2::from_elem((1, 1), 1.0),
        r: Array2::from_elem((1, 1), 1.0),
        q: Array2::from_elem((1, 1), 0.25),
        r_obs: Array2::from_elem((1, 1), 1.0),
        s0: Array1::from_vec(vec![0.0]),
        p0: Array2::from_elem((1, 1), 1.0),
    };

    (obs, model)
}

/// State-space estimation returns filtered and smoothed states with the expected shapes.
#[test]
fn test_state_space_estimate_invariants() {
    let (y, model) = make_random_walk_obs(12345, 60);
    let result = state_space_estimate(&model, &y).unwrap();

    assert_eq!(result.n_obs, 60);
    assert_eq!(result.n_states, 1);
    assert_eq!(result.filtered_states.len(), 60);
    assert_eq!(result.smoothed_states.len(), 60);
    assert_eq!(result.filtered_cov.len(), 60);
    assert_eq!(result.smoothed_cov.len(), 60);
    assert_eq!(result.innovations.len(), 60);
    assert!(result.log_likelihood.is_finite());
    assert!(result.filtered_states.iter().all(|s| s[0].is_finite()));
    assert!(result.smoothed_states.iter().all(|s| s[0].is_finite()));

    // Forecasts from the last filtered state remain one-dimensional and finite.
    let forecasts = result.predict(&model, 5);
    assert_eq!(forecasts.len(), 5);
    assert!(forecasts.iter().all(|f| f[0].is_finite()));
}

/// Kalman filter and smoother can be run independently and agree on state count.
#[test]
fn test_kalman_filter_smoother_invariants() {
    let (y, model) = make_random_walk_obs(23456, 60);
    let filter = KalmanFilter::filter(&model, &y).unwrap();

    assert_eq!(filter.n_obs, 60);
    assert_eq!(filter.n_states, 1);
    assert_eq!(filter.filtered_states.len(), 60);
    assert_eq!(filter.predicted_states.len(), 60);
    assert_eq!(filter.innovations.len(), 60);
    assert!(filter.log_likelihood.is_finite());

    let smooth = KalmanSmoother::smooth(&model, &filter).unwrap();
    assert_eq!(smooth.n_obs, 60);
    assert_eq!(smooth.smoothed_states.len(), 60);
    assert!(smooth.smoothed_states.iter().all(|s| s[0].is_finite()));
}

/// Local-level model estimates finite variances and state paths.
#[test]
fn test_local_level_invariants() {
    let y = make_local_level_series(34567, 120);
    let result = LocalLevel::fit(&y).unwrap();

    assert_eq!(result.n_obs, 120);
    assert!(result.sigma_obs > 0.0 && result.sigma_obs.is_finite());
    assert!(result.sigma_state > 0.0 && result.sigma_state.is_finite());
    assert!(result.log_likelihood.is_finite());
    assert_eq!(result.filtered_states.len(), 120);
    assert_eq!(result.smoothed_states.len(), 120);
    assert!(result.filtered_states.iter().all(|s| s[0].is_finite()));
    assert!(result.smoothed_states.iter().all(|s| s[0].is_finite()));
}

/// Input validation catches too few observations for local-level estimation.
#[test]
fn test_statespace_input_validation() {
    let y_short = make_local_level_series(11111, 3);
    assert!(LocalLevel::fit(&y_short).is_err());
}
