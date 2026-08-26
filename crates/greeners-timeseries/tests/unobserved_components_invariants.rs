use greeners_timeseries::unobserved_components::UCLevel;
use greeners_timeseries::unobserved_components::UCSeasonal;
use greeners_timeseries::unobserved_components::UnobservedComponents;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn test_uc_local_level_runs_and_produces_finite_output() {
    let n = 50;
    let mut rng = StdRng::seed_from_u64(20001);
    let noise = Normal::new(0.0, 0.5).unwrap();

    // Random walk + noise.
    let mut y = vec![0.0; n];
    for i in 1..n {
        y[i] = y[i - 1] + noise.sample(&mut rng);
    }
    for i in 0..n {
        y[i] += noise.sample(&mut rng);
    }
    let y = Array1::from_vec(y);

    let result = UnobservedComponents::fit(&y, UCLevel::LocalLevel, UCSeasonal::None).unwrap();

    assert_eq!(result.n_obs, n);
    assert_eq!(result.level.len(), n);
    assert_eq!(result.residuals.len(), n);
    assert!(result.level.iter().all(|&v| v.is_finite()));
    assert!(result.residuals.iter().all(|&v| v.is_finite()));
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
    assert!(!result.params.is_empty());
    assert_eq!(result.params.len(), result.param_names.len());

    let forecasts = result.predict(5);
    assert_eq!(forecasts.len(), 5);
    assert!(forecasts.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_uc_local_linear_trend() {
    let n = 50;
    let mut rng = StdRng::seed_from_u64(20002);
    let noise = Normal::new(0.0, 0.3).unwrap();

    // Linear trend with noise.
    let y = Array1::from(
        (0..n)
            .map(|i| 0.5 * i as f64 + noise.sample(&mut rng))
            .collect::<Vec<_>>(),
    );

    let result =
        UnobservedComponents::fit(&y, UCLevel::LocalLinearTrend, UCSeasonal::None).unwrap();

    assert_eq!(result.n_obs, n);
    assert!(result.trend.is_some());
    assert_eq!(result.trend.as_ref().unwrap().len(), n);
    assert!(result
        .trend
        .as_ref()
        .unwrap()
        .iter()
        .all(|&v| v.is_finite()));
    assert!(result.level.iter().all(|&v| v.is_finite()));
    assert!(result.residuals.iter().all(|&v| v.is_finite()));

    // Forecasts should be increasing for a positive trend.
    let forecasts = result.predict(5);
    assert!(forecasts.iter().all(|&v| v.is_finite()));
    assert!(forecasts[4] > forecasts[0]);
}

#[test]
fn test_uc_input_validation() {
    let y = Array1::from(vec![1.0, 2.0]);

    // Too few observations.
    assert!(UnobservedComponents::fit(&y, UCLevel::LocalLevel, UCSeasonal::None).is_err());
}
