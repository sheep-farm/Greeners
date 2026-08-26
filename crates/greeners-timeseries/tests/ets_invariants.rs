use greeners_timeseries::ets::ExponentialSmoothing;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn test_ets_level_and_predict() {
    let n = 60;
    let mut rng = StdRng::seed_from_u64(16001);
    let noise = Normal::new(0.0, 0.5).unwrap();

    // Constant level with noise.
    let y = Array1::from(
        (0..n)
            .map(|_| 5.0 + noise.sample(&mut rng))
            .collect::<Vec<_>>(),
    );

    let result = ExponentialSmoothing::fit(&y, None, None, 0, false).unwrap();

    assert_eq!(result.n_obs, n);
    assert!(result.alpha > 0.0 && result.alpha < 1.0);
    assert!(result.beta.is_none());
    assert!(result.gamma.is_none());
    assert!(result.phi.is_none());
    assert_eq!(result.level.len(), n);
    assert_eq!(result.fitted_values.len(), n);
    assert_eq!(result.residuals.len(), n);
    assert!(result.fitted_values.iter().all(|&v| v.is_finite()));
    assert!(result.residuals.iter().all(|&v| v.is_finite()));
    assert!(result.sse.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());

    let forecasts = result.predict(5);
    assert_eq!(forecasts.len(), 5);
    assert!(forecasts.iter().all(|&v| v.is_finite()));
    // Forecasts should hover near the level.
    assert!(forecasts.iter().all(|&v| v > 3.0 && v < 7.0));
}

#[test]
fn test_ets_trend_and_seasonal() {
    let n = 80;
    let mut rng = StdRng::seed_from_u64(16002);
    let noise = Normal::new(0.0, 0.3).unwrap();

    // Linear trend + additive seasonal period 4.
    let y = Array1::from(
        (0..n)
            .map(|i| {
                let trend = 0.5 * i as f64;
                let seasonal = [2.0, 1.0, -1.0, -2.0][i % 4];
                trend + seasonal + noise.sample(&mut rng)
            })
            .collect::<Vec<_>>(),
    );

    let result = ExponentialSmoothing::fit(&y, Some("add"), Some("add"), 4, false).unwrap();

    assert_eq!(result.n_obs, n);
    assert_eq!(result.trend_type, "add");
    assert_eq!(result.seasonal_type, "add");
    assert_eq!(result.seasonal_periods, 4);
    assert!(result.beta.is_some());
    assert!(result.gamma.is_some());
    assert!(result.trend.len() == n);
    assert!(result.seasonal.len() == n);
    assert!(result.beta.unwrap() > 0.0);
    assert!(result.gamma.unwrap() > 0.0);
    assert!(result.sse.is_finite());

    let forecasts = result.predict(4);
    assert_eq!(forecasts.len(), 4);
    assert!(forecasts.iter().all(|&v| v.is_finite()));
    // Trend should be estimated as positive.
    assert!(result.last_trend > 0.0);
}

#[test]
fn test_ets_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);

    // Too short.
    assert!(ExponentialSmoothing::fit(&y, None, None, 0, false).is_err());

    // Seasonal requires at least 2 full periods.
    let y_long = Array1::from(vec![1.0; 20]);
    assert!(ExponentialSmoothing::fit(&y_long, None, Some("add"), 12, false).is_err());
}
