use greeners::{KalmanFilter, KalmanSmoother, LocalLevel, StateSpaceModel};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Local-level model on a random walk: variances positive and states have
/// correct length.
#[test]
fn test_local_level_basic_properties() {
    let n = 60;
    let mut rng = StdRng::seed_from_u64(2025);
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut y = vec![0.0];
    for _ in 1..n {
        y.push(y.last().unwrap() + norm.sample(&mut rng));
    }

    let result = LocalLevel::fit(&y).unwrap();

    assert!(result.sigma_obs > 0.0);
    assert!(result.sigma_state > 0.0);
    assert!(result.log_likelihood.is_finite());
    assert_eq!(result.n_obs, n);
    assert_eq!(result.filtered_states.len(), n);
    assert_eq!(result.smoothed_states.len(), n);

    // The first and last filtered states are just the observations (initial
    // value and final update close to y_T), so they are finite.
    assert!(result.filtered_states[0][0].is_finite());
    assert!(result.filtered_states[n - 1][0].is_finite());

    // Smoothed and filtered states at the last time point are identical.
    approx_zero(
        (result.smoothed_states[n - 1][0] - result.filtered_states[n - 1][0]).abs(),
        1e-10,
    );
}

#[test]
fn test_local_level_input_validation() {
    assert!(LocalLevel::fit(&[1.0, 2.0]).is_err());
}

/// Kalman filter on a constant state with observation noise: the final
/// filtered state converges to the true constant.
#[test]
fn test_kalman_filter_constant_state() {
    let n = 100;
    let true_mu = 5.0;
    let sigma_obs = 0.5;

    let mut rng = StdRng::seed_from_u64(123);
    let norm = Normal::new(0.0, sigma_obs).unwrap();
    let y: Vec<Array1<f64>> = (0..n)
        .map(|_| Array1::from_vec(vec![true_mu + norm.sample(&mut rng)]))
        .collect();

    let model = StateSpaceModel {
        h: Array2::from_elem((1, 1), 1.0),
        f: Array2::from_elem((1, 1), 1.0),
        r: Array2::from_elem((1, 1), 0.0), // no state noise
        q: Array2::from_elem((1, 1), 1e-12),
        r_obs: Array2::from_elem((1, 1), sigma_obs * sigma_obs),
        s0: Array1::from_vec(vec![0.0]),
        p0: Array2::from_elem((1, 1), 100.0),
    };

    let filter = KalmanFilter::filter(&model, &y).unwrap();
    let final_state = filter.filtered_states[n - 1][0];
    approx_zero((final_state - true_mu).abs(), 0.2);

    // Smoother should give the same last state.
    let smooth = KalmanSmoother::smooth(&model, &filter).unwrap();
    approx_zero(
        (smooth.smoothed_states[n - 1][0] - filter.filtered_states[n - 1][0]).abs(),
        1e-10,
    );
}
