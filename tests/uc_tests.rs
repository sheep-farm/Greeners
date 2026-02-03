use greeners::{UCLevel, UCSeasonal, UnobservedComponents};
use ndarray::Array1;

/// Generate a random walk: y_t = y_{t-1} + e_t
fn make_random_walk(n: usize, seed: u64) -> Array1<f64> {
    let mut y = vec![0.0; n];
    let mut state = seed;
    for i in 1..n {
        // Simple LCG pseudo-random
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        y[i] = y[i - 1] + u * 0.5;
    }
    Array1::from_vec(y)
}

/// Generate random walk with drift: y_t = y_{t-1} + 0.1 + e_t
fn make_trend_data(n: usize, seed: u64) -> Array1<f64> {
    let mut y = vec![0.0; n];
    let mut state = seed;
    for i in 1..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        y[i] = y[i - 1] + 0.1 + u * 0.3;
    }
    Array1::from_vec(y)
}

/// Generate seasonal data with period 12
fn make_seasonal_data(n: usize, seed: u64) -> Array1<f64> {
    let mut y = vec![0.0; n];
    let mut state = seed;
    let seasonal_pattern = [
        2.0, 1.5, 0.5, -0.5, -1.5, -2.0, -1.5, -0.5, 0.5, 1.5, 2.0, 1.0,
    ];
    let mut level = 10.0;
    for i in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        level += u * 0.1;
        y[i] = level + seasonal_pattern[i % 12] + u * 0.2;
    }
    Array1::from_vec(y)
}

#[test]
fn test_local_level() {
    let y = make_random_walk(200, 42);
    let result = UnobservedComponents::fit(&y, UCLevel::LocalLevel, UCSeasonal::None).unwrap();

    assert_eq!(result.n_obs, 200);
    assert_eq!(result.level.len(), 200);
    assert!(result.trend.is_none());
    assert!(result.seasonal.is_none());
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
    assert_eq!(result.params.len(), 2); // sigma2_irregular + sigma2_level
    assert!(result.params.iter().all(|p| *p > 0.0));

    // Forecasts should work
    let fc = result.predict(5);
    assert_eq!(fc.len(), 5);
    assert!(fc.iter().all(|v| v.is_finite()));

    // Display should work
    let display = format!("{}", result);
    assert!(display.contains("Unobserved Components"));
}

#[test]
fn test_local_linear_trend() {
    let y = make_trend_data(200, 123);
    let result =
        UnobservedComponents::fit(&y, UCLevel::LocalLinearTrend, UCSeasonal::None).unwrap();

    assert_eq!(result.n_obs, 200);
    assert!(result.trend.is_some());
    let trend = result.trend.as_ref().unwrap();
    assert_eq!(trend.len(), 200);
    // Trend should generally be positive for data with drift
    let mean_trend = trend.mean().unwrap_or(0.0);
    assert!(
        mean_trend > -1.0,
        "Expected positive trend, got {}",
        mean_trend
    );
    assert_eq!(result.params.len(), 3); // irregular + level + trend
}

#[test]
fn test_smooth_trend() {
    let y = make_trend_data(150, 456);
    let result = UnobservedComponents::fit(&y, UCLevel::SmoothTrend, UCSeasonal::None).unwrap();

    assert_eq!(result.n_obs, 150);
    assert!(result.trend.is_some());
    // SmoothTrend: no sigma2_level, so params = [sigma2_irregular, sigma2_trend]
    assert_eq!(result.params.len(), 2);
}

#[test]
fn test_random_walk() {
    let y = make_random_walk(100, 789);
    let result = UnobservedComponents::fit(&y, UCLevel::RandomWalk, UCSeasonal::None).unwrap();

    assert_eq!(result.n_obs, 100);
    // RandomWalk: no observation noise, so params = [sigma2_level]
    assert_eq!(result.params.len(), 1);
    assert!(result.log_likelihood.is_finite());
}

#[test]
fn test_stochastic_seasonal() {
    let y = make_seasonal_data(120, 321);
    let result =
        UnobservedComponents::fit(&y, UCLevel::LocalLevel, UCSeasonal::Stochastic(12)).unwrap();

    assert_eq!(result.n_obs, 120);
    assert!(result.seasonal.is_some());
    let seasonal = result.seasonal.as_ref().unwrap();
    assert_eq!(seasonal.len(), 120);
    // sigma2_irregular + sigma2_level + sigma2_seasonal
    assert_eq!(result.params.len(), 3);

    // Forecasts should show periodicity
    let fc = result.predict(24);
    assert_eq!(fc.len(), 24);
    assert!(fc.iter().all(|v| v.is_finite()));
}

#[test]
fn test_deterministic_seasonal() {
    let y = make_seasonal_data(120, 654);
    let result =
        UnobservedComponents::fit(&y, UCLevel::LocalLevel, UCSeasonal::Deterministic(12)).unwrap();

    assert_eq!(result.n_obs, 120);
    assert!(result.seasonal.is_some());
    // Deterministic seasonal: no sigma2_seasonal, so params = [sigma2_irregular, sigma2_level]
    assert_eq!(result.params.len(), 2);
}

#[test]
fn test_too_few_observations() {
    let y = Array1::from_vec(vec![1.0, 2.0]);
    let result = UnobservedComponents::fit(&y, UCLevel::LocalLevel, UCSeasonal::None);
    assert!(result.is_err());
}
