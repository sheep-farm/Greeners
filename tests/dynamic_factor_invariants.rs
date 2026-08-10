use greeners::DynamicFactor;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn generate_dynamic_factor_data(
    t: usize,
    n_vars: usize,
    n_factors: usize,
    seed: u64,
) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    // n_factors independent AR(1) factors.
    let mut factors = Array2::<f64>::zeros((t, n_factors));
    for i in 1..t {
        for f in 0..n_factors {
            factors[[i, f]] = 0.6 * factors[[i - 1, f]] + noise.sample(&mut rng);
        }
    }

    // Observed series are linear combinations of factors.
    let mut data = Array2::<f64>::zeros((t, n_vars));
    for i in 0..t {
        for j in 0..n_vars {
            let loading = 0.2 + 0.1 * ((j % n_factors) as f64);
            data[[i, j]] = loading * factors[[i, j % n_factors]] + noise.sample(&mut rng);
        }
    }

    data
}

#[test]
fn test_dynamic_factor_runs_and_produces_finite_output() {
    let data = generate_dynamic_factor_data(60, 5, 2, 18001);

    let result = DynamicFactor::fit(&data, 2, 1).unwrap();

    assert_eq!(result.n_obs, 60);
    assert_eq!(result.n_vars, 5);
    assert_eq!(result.n_factors, 2);
    assert_eq!(result.factor_order, 1);
    assert_eq!(result.factor_loadings.shape(), &[5, 2]);
    assert_eq!(result.factors.shape(), &[60, 2]);
    assert_eq!(result.factor_ar_params.len(), 1);
    assert_eq!(result.factor_ar_params[0].shape(), &[2, 2]);
    assert_eq!(result.sigma_obs.len(), 5);
    assert_eq!(result.sigma_factor.shape(), &[2, 2]);
    assert!(result.factor_loadings.iter().all(|&v| v.is_finite()));
    assert!(result.factors.iter().all(|&v| v.is_finite()));
    assert!(result
        .factor_ar_params
        .iter()
        .all(|m| m.iter().all(|&v| v.is_finite())));
    assert!(result.sigma_obs.iter().all(|&v| v.is_finite() && v > 0.0));
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_dynamic_factor_predict_and_ar_stability() {
    let data = generate_dynamic_factor_data(80, 5, 2, 18002);

    let result = DynamicFactor::fit(&data, 2, 1).unwrap();

    // Forecasts should be finite and of the right shape.
    let forecasts = result.predict(5);
    assert_eq!(forecasts.shape(), &[5, 5]);
    assert!(forecasts.iter().all(|&v| v.is_finite()));

    // Factor AR matrix should be stable.
    let ar = &result.factor_ar_params[0];
    let det = ar[[0, 0]] * ar[[1, 1]] - ar[[0, 1]] * ar[[1, 0]];
    let trace = ar[[0, 0]] + ar[[1, 1]];
    let disc = trace * trace - 4.0 * det;
    let (e1, e2) = if disc >= 0.0 {
        let s = disc.sqrt();
        ((trace + s) / 2.0, (trace - s) / 2.0)
    } else {
        let m = trace / 2.0;
        let r = (-disc).sqrt() / 2.0;
        (m.hypot(r), m.hypot(r))
    };
    assert!(e1 < 1.0 && e2 < 1.0);
}

#[test]
fn test_dynamic_factor_input_validation() {
    let data = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();

    // k_factors = 0.
    assert!(DynamicFactor::fit(&data, 0, 1).is_err());

    // k_factors >= n_vars.
    assert!(DynamicFactor::fit(&data, 3, 1).is_err());

    // factor_order = 0.
    assert!(DynamicFactor::fit(&data, 1, 0).is_err());

    // Too few observations.
    let short = Array2::from_shape_vec((3, 3), vec![1.0; 9]).unwrap();
    assert!(DynamicFactor::fit(&short, 1, 1).is_err());
}
