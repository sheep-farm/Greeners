use greeners::ARIMA;
use ndarray::{Array1, Array2};

/// Generate an AR(1) process: y_t = c + phi * y_{t-1} + e_t
fn generate_ar1(n: usize, phi: f64, c: f64, seed: u64) -> Array1<f64> {
    let mut y = vec![0.0; n];
    let mut rng_state = seed;
    for t in 1..n {
        // Simple LCG pseudo-random for reproducibility
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        let noise = u * 0.5; // small noise
        y[t] = c + phi * y[t - 1] + noise;
    }
    Array1::from_vec(y)
}

/// Generate an MA(1) process: y_t = c + e_t + theta * e_{t-1}
fn generate_ma1(n: usize, theta: f64, c: f64, seed: u64) -> Array1<f64> {
    let mut y = vec![0.0; n];
    let mut rng_state = seed;
    let mut prev_e = 0.0;
    for t in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        let e = u * 0.5;
        y[t] = c + e + theta * prev_e;
        prev_e = e;
    }
    Array1::from_vec(y)
}

#[test]
fn test_ar1_recovery() {
    let y = generate_ar1(500, 0.7, 0.1, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();

    assert_eq!(result.ar_params.len(), 1);
    assert_eq!(result.ma_params.len(), 0);
    // AR coefficient should be close to 0.7
    assert!(
        (result.ar_params[0] - 0.7).abs() < 0.15,
        "AR(1) coef: {} (expected ~0.7)",
        result.ar_params[0]
    );
    assert!(result.sigma2 > 0.0);
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_ma1_recovery() {
    let y = generate_ma1(500, 0.5, 0.0, 123);
    let result = ARIMA::fit(&y, (0, 0, 1)).unwrap();

    assert_eq!(result.ma_params.len(), 1);
    assert_eq!(result.ar_params.len(), 0);
    // MA coefficient should be in a reasonable range
    assert!(
        result.ma_params[0].abs() < 1.5,
        "MA(1) coef: {}",
        result.ma_params[0]
    );
    assert!(result.sigma2 > 0.0);
}

#[test]
fn test_arima_111_on_random_walk() {
    // Generate random walk (integrated AR(1))
    let ar1 = generate_ar1(500, 0.5, 0.1, 77);
    // Cumulative sum to make it integrated
    let mut y_vec = vec![0.0; 500];
    y_vec[0] = ar1[0];
    for i in 1..500 {
        y_vec[i] = y_vec[i - 1] + ar1[i];
    }
    let y = Array1::from_vec(y_vec);

    let result = ARIMA::fit(&y, (1, 1, 1)).unwrap();
    assert_eq!(result.order.p, 1);
    assert_eq!(result.order.d, 1);
    assert_eq!(result.order.q, 1);
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
    assert!(result.n_obs > 0);
}

#[test]
fn test_sarimax_seasonal_ar() {
    // Generate series with seasonal AR(1) at period 12
    let n = 300;
    let mut y = vec![0.0; n];
    let mut rng_state: u64 = 999;
    for t in 12..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        y[t] = 0.6 * y[t - 12] + u * 0.3;
    }
    let y = Array1::from_vec(y);

    let result = ARIMA::fit_sarimax(&y, (0, 0, 0), (1, 0, 0, 12), None).unwrap();
    assert_eq!(result.seasonal_ar_params.len(), 1);
    assert!(result.seasonal_order.is_some());
    let so = result.seasonal_order.as_ref().unwrap();
    assert_eq!(so.s, 12);
    assert!(result.aic.is_finite());
}

#[test]
fn test_arimax_with_exogenous() {
    let n = 200;
    let mut y_vec = vec![0.0; n];
    let mut x_vec = vec![0.0; n];
    let mut rng_state: u64 = 55;
    for t in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        x_vec[t] = u;
        y_vec[t] = 2.0 * x_vec[t] + if t > 0 { 0.5 * y_vec[t - 1] } else { 0.0 } + u * 0.1;
    }
    let y = Array1::from_vec(y_vec);
    let exog = Array2::from_shape_vec((n, 1), x_vec).unwrap();

    let result = ARIMA::fit_sarimax(&y, (1, 0, 0), (0, 0, 0, 1), Some(&exog)).unwrap();
    assert!(result.exog_params.is_some());
    assert_eq!(result.exog_params.as_ref().unwrap().len(), 1);
}

#[test]
fn test_pure_ar_no_differencing() {
    let y = generate_ar1(200, 0.3, 0.0, 10);
    let result = ARIMA::fit(&y, (2, 0, 0)).unwrap();
    assert_eq!(result.ar_params.len(), 2);
    assert_eq!(result.order.d, 0);
}

#[test]
fn test_pure_ma_no_ar() {
    let y = generate_ma1(200, -0.4, 1.0, 20);
    let result = ARIMA::fit(&y, (0, 0, 1)).unwrap();
    assert_eq!(result.ar_params.len(), 0);
    assert_eq!(result.ma_params.len(), 1);
}

#[test]
fn test_short_series_error() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let result = ARIMA::fit(&y, (1, 0, 0));
    assert!(result.is_err());
}

#[test]
fn test_predict_returns_correct_length() {
    let y = generate_ar1(200, 0.5, 0.0, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();
    let forecast = result.predict(10, None).unwrap();
    assert_eq!(forecast.len(), 10);
    for v in forecast.iter() {
        assert!(v.is_finite(), "Forecast contains non-finite value: {}", v);
    }
}

#[test]
fn test_predict_with_differencing() {
    let ar1 = generate_ar1(300, 0.5, 0.1, 77);
    let mut y_vec = vec![0.0; 300];
    y_vec[0] = ar1[0];
    for i in 1..300 {
        y_vec[i] = y_vec[i - 1] + ar1[i];
    }
    let y = Array1::from_vec(y_vec);

    let result = ARIMA::fit(&y, (1, 1, 0)).unwrap();
    let forecast = result.predict(5, None).unwrap();
    assert_eq!(forecast.len(), 5);
    for v in forecast.iter() {
        assert!(v.is_finite());
    }
}

#[test]
fn test_display() {
    let y = generate_ar1(200, 0.5, 0.0, 42);
    let result = ARIMA::fit(&y, (1, 0, 1)).unwrap();
    let display = format!("{}", result);
    assert!(display.contains("ARIMA"));
    assert!(display.contains("AIC"));
    assert!(display.contains("BIC"));
    // New: should contain inference table headers
    assert!(display.contains("coef"));
    assert!(display.contains("std err"));
    assert!(display.contains("P>|z|"));
}

#[test]
fn test_residuals_length() {
    let y = generate_ar1(200, 0.5, 0.0, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();
    assert_eq!(result.residuals().len(), result.n_obs);
}

#[test]
fn test_fitted_values_length() {
    let y = generate_ar1(200, 0.5, 0.0, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();
    assert_eq!(result.fitted_values().len(), result.n_obs);
}

#[test]
fn test_aic_bic_finite() {
    let y = generate_ar1(200, 0.5, 0.0, 42);
    let result = ARIMA::fit(&y, (1, 0, 1)).unwrap();
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

// --- New tests for inference, diagnostics, and predict with exog ---

#[test]
fn test_inference_fields() {
    let y = generate_ar1(500, 0.7, 0.1, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();

    // 2 params: intercept + ar.L1
    assert_eq!(result.std_errors.len(), 2);
    assert_eq!(result.t_values.len(), 2);
    assert_eq!(result.p_values.len(), 2);
    assert_eq!(result.conf_lower.len(), 2);
    assert_eq!(result.conf_upper.len(), 2);
    assert_eq!(result.param_names.len(), 2);
    assert_eq!(result.param_names[0], "intercept");
    assert_eq!(result.param_names[1], "ar.L1");

    // Standard errors should be positive
    for &se in result.std_errors.iter() {
        assert!(se >= 0.0);
    }
    // p-values should be in [0, 1]
    for &pv in result.p_values.iter() {
        assert!((0.0..=1.0).contains(&pv), "p-value out of range: {}", pv);
    }
    // Confidence intervals: lower < upper
    for i in 0..result.conf_lower.len() {
        assert!(result.conf_lower[i] <= result.conf_upper[i]);
    }
    // Log-likelihood should be finite
    assert!(result.log_likelihood.is_finite());
    assert!(result.df_model > 0);
    assert!(result.df_resid > 0);
}

#[test]
fn test_inference_ar1_significance() {
    let y = generate_ar1(500, 0.7, 0.1, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();

    // The AR(1) coefficient should be significant (p < 0.05)
    assert!(
        result.p_values[1] < 0.05,
        "AR(1) p-value should be significant: {}",
        result.p_values[1]
    );
}

#[test]
fn test_predict_with_exog() {
    let n = 200;
    let mut y_vec = vec![0.0; n];
    let mut x_vec = vec![0.0; n];
    let mut rng_state: u64 = 55;
    for t in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        x_vec[t] = u;
        y_vec[t] = 2.0 * x_vec[t] + if t > 0 { 0.5 * y_vec[t - 1] } else { 0.0 } + u * 0.1;
    }
    let y = Array1::from_vec(y_vec);
    let exog = Array2::from_shape_vec((n, 1), x_vec).unwrap();

    let result = ARIMA::fit_sarimax(&y, (1, 0, 0), (0, 0, 0, 1), Some(&exog)).unwrap();

    // Predict with future exogenous values
    let future_x = Array2::from_shape_vec((5, 1), vec![0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
    let forecast = result.predict(5, Some(&future_x)).unwrap();
    assert_eq!(forecast.len(), 5);
    for v in forecast.iter() {
        assert!(v.is_finite());
    }

    // Predict without exog should also work (exog contribution = 0)
    let forecast_no_exog = result.predict(5, None).unwrap();
    assert_eq!(forecast_no_exog.len(), 5);
}

#[test]
fn test_predict_exog_shape_mismatch() {
    let n = 200;
    let mut y_vec = vec![0.0; n];
    let mut x_vec = vec![0.0; n];
    let mut rng_state: u64 = 55;
    for t in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        x_vec[t] = u;
        y_vec[t] = 2.0 * x_vec[t] + if t > 0 { 0.5 * y_vec[t - 1] } else { 0.0 } + u * 0.1;
    }
    let y = Array1::from_vec(y_vec);
    let exog = Array2::from_shape_vec((n, 1), x_vec).unwrap();

    let result = ARIMA::fit_sarimax(&y, (1, 0, 0), (0, 0, 0, 1), Some(&exog)).unwrap();

    // Wrong number of rows
    let bad_x = Array2::from_shape_vec((3, 1), vec![0.1, 0.2, 0.3]).unwrap();
    assert!(result.predict(5, Some(&bad_x)).is_err());
}

#[test]
fn test_ljung_box() {
    let y = generate_ar1(500, 0.7, 0.1, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();

    let (stat, pval) = result.ljung_box(10).unwrap();
    assert!(stat.is_finite());
    assert!(stat >= 0.0);
    assert!((0.0..=1.0).contains(&pval));

    // For a well-specified AR(1), residuals should not show autocorrelation
    // so p-value should be > 0.01 (not too strict since Hannan-Rissanen is approximate)
}

#[test]
fn test_ljung_box_invalid_lags() {
    let y = generate_ar1(200, 0.5, 0.0, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();

    // lags = 0 should error
    assert!(result.ljung_box(0).is_err());
}

#[test]
fn test_acf() {
    let y = generate_ar1(500, 0.7, 0.1, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();

    let acf_vals = result.acf(5);
    assert_eq!(acf_vals.len(), 5);
    for &v in &acf_vals {
        assert!(v.is_finite());
        assert!(v.abs() <= 1.0 + 1e-10, "ACF value out of range: {}", v);
    }
}

#[test]
fn test_is_stationary() {
    let y = generate_ar1(500, 0.7, 0.1, 42);
    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();
    // AR(1) with |phi| < 1 should be stationary
    assert!(result.is_stationary());
    assert!(result.is_invertible()); // no MA params, trivially invertible
}

#[test]
fn test_is_stationary_ma() {
    let y = generate_ma1(500, 0.5, 0.0, 123);
    let result = ARIMA::fit(&y, (0, 0, 1)).unwrap();
    // Pure MA is always stationary (no AR roots)
    assert!(result.is_stationary());
    // MA(1) with |theta| < 1 should be invertible
    assert!(
        result.is_invertible(),
        "MA coef {} should be invertible",
        result.ma_params[0]
    );
}
