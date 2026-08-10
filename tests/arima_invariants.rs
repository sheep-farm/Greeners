use greeners::ARIMA;
use ndarray::Array1;
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Noisy AR(1): y_t = c + phi y_{t-1} + eps_t. Hannan-Rissanen recovers
/// phi and c approximately for long series.
#[test]
fn test_arima_ar1_recovery() {
    let n = 200;
    let phi = 0.6;
    let c = 2.0;
    let mut rng = StdRng::seed_from_u64(789);
    let norm = Normal::new(0.0, 0.5).unwrap();
    let mut y = vec![5.0];
    for t in 1..n {
        let y_t = c + phi * y[t - 1] + norm.sample(&mut rng);
        y.push(y_t);
    }
    let y = Array1::from(y);

    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();
    approx_zero((result.ar_params[0] - phi).abs(), 0.05);
    approx_zero((result.intercept - c).abs(), 0.3);
}

/// ARIMA(0,1,0) on a random walk: the intercept on the differenced series
/// should be close to the mean of the increments.
#[test]
fn test_arima_random_walk_i10() {
    let n = 120;
    let mut rng = StdRng::seed_from_u64(2025);
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut y = vec![0.0];
    let mut steps = Vec::with_capacity(n - 1);
    for _ in 1..n {
        let step = norm.sample(&mut rng);
        steps.push(step);
        y.push(y.last().unwrap() + step);
    }
    let y = Array1::from(y);

    let result = ARIMA::fit(&y, (0, 1, 0)).unwrap();
    let mean_step: f64 = steps.iter().sum::<f64>() / steps.len() as f64;
    approx_zero((result.intercept - mean_step).abs(), 0.15);
}

/// Input validation.
#[test]
fn test_arima_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(ARIMA::fit(&y, (1, 0, 0)).is_err()); // too short

    let y = Array1::from(vec![1.0; 30]);
    assert!(ARIMA::fit(&y, (0, 30, 0)).is_err()); // too much differencing
}
