use greeners::DynamicFactor;
use ndarray::Array2;

fn generate_factor_data(t: usize) -> Array2<f64> {
    // Generate 1 common factor driving 3 observed series + noise
    let mut data = Array2::<f64>::zeros((t, 3));

    // AR(1) factor: f_t = 0.8 * f_{t-1} + u_t
    let mut factor = 0.0_f64;
    let mut seed: u64 = 42;

    for i in 0..t {
        // Simple LCG pseudo-random
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
        factor = 0.8 * factor + u * 0.5;

        // y1 = 1.0 * f + noise
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e1 = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.2;
        data[[i, 0]] = 1.0 * factor + e1;

        // y2 = 0.7 * f + noise
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e2 = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.2;
        data[[i, 1]] = 0.7 * factor + e2;

        // y3 = 0.5 * f + noise
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e3 = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.2;
        data[[i, 2]] = 0.5 * factor + e3;
    }

    data
}

#[test]
fn test_dynamic_factor_basic() {
    let data = generate_factor_data(200);
    let result = DynamicFactor::fit(&data, 1, 1).unwrap();

    // Check shapes
    assert_eq!(result.factor_loadings.nrows(), 3);
    assert_eq!(result.factor_loadings.ncols(), 1);
    assert_eq!(result.factors.nrows(), 200);
    assert_eq!(result.factors.ncols(), 1);
    assert_eq!(result.factor_ar_params.len(), 1);
    assert_eq!(result.factor_ar_params[0].nrows(), 1);
    assert_eq!(result.factor_ar_params[0].ncols(), 1);
    assert_eq!(result.sigma_obs.len(), 3);
    assert_eq!(result.n_obs, 200);
    assert_eq!(result.n_vars, 3);
    assert_eq!(result.n_factors, 1);
    assert_eq!(result.factor_order, 1);

    // Log-likelihood should be finite
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_dynamic_factor_forecast() {
    let data = generate_factor_data(200);
    let result = DynamicFactor::fit(&data, 1, 1).unwrap();

    let forecast = result.predict(5);
    assert_eq!(forecast.nrows(), 5);
    assert_eq!(forecast.ncols(), 3);

    // Forecasts should be finite
    for val in forecast.iter() {
        assert!(val.is_finite(), "Forecast contains non-finite value");
    }
}

#[test]
fn test_dynamic_factor_display() {
    let data = generate_factor_data(200);
    let result = DynamicFactor::fit(&data, 1, 1).unwrap();
    let display = format!("{}", result);
    assert!(display.contains("Dynamic Factor Model"));
    assert!(display.contains("Factor Loadings"));
}

#[test]
fn test_dynamic_factor_higher_order() {
    let data = generate_factor_data(200);
    let result = DynamicFactor::fit(&data, 1, 2).unwrap();

    assert_eq!(result.factor_ar_params.len(), 2);
    assert_eq!(result.factor_order, 2);
}

#[test]
fn test_dynamic_factor_invalid_params() {
    let data = generate_factor_data(50);

    // k_factors >= k should fail
    assert!(DynamicFactor::fit(&data, 3, 1).is_err());
    // k_factors = 0 should fail
    assert!(DynamicFactor::fit(&data, 0, 1).is_err());
    // factor_order = 0 should fail
    assert!(DynamicFactor::fit(&data, 1, 0).is_err());
}
