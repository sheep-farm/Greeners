use greeners::*;
use ndarray::{Array1, Array2};

// ─── Helper ─────────────────────────────────────────────────────────────────

fn sine_with_trend(n: usize) -> Array1<f64> {
    Array1::from_vec(
        (0..n)
            .map(|i| {
                let t = i as f64;
                0.5 * t + 10.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin() + 100.0
            })
            .collect(),
    )
}

fn seasonal_series(n: usize, period: usize) -> Array1<f64> {
    Array1::from_vec(
        (0..n)
            .map(|i| {
                let t = i as f64;
                let trend = 0.1 * t + 50.0;
                let seasonal = 5.0 * (2.0 * std::f64::consts::PI * t / period as f64).sin();
                trend + seasonal + 0.5 * ((i * 7 + 3) % 11) as f64 / 11.0
            })
            .collect(),
    )
}

fn ar1_series(n: usize, phi: f64) -> Array1<f64> {
    let mut y = vec![0.0; n];
    let seed_vals = [0.3, -0.2, 0.5, -0.1, 0.4, -0.3, 0.1, 0.2, -0.4, 0.6];
    for i in 1..n {
        y[i] = phi * y[i - 1] + seed_vals[i % seed_vals.len()] * 0.5;
    }
    Array1::from_vec(y)
}

fn var_data(n: usize) -> Array2<f64> {
    let mut data = Array2::<f64>::zeros((n, 2));
    let seed_a = [0.3, -0.1, 0.4, 0.2, -0.3, 0.1, -0.2, 0.5, -0.4, 0.35];
    let seed_b = [0.1, 0.3, -0.2, 0.4, -0.1, 0.2, 0.3, -0.3, 0.15, -0.25];
    for i in 1..n {
        data[[i, 0]] =
            0.5 * data[[i - 1, 0]] + 0.2 * data[[i - 1, 1]] + seed_a[i % seed_a.len()] * 0.5;
        data[[i, 1]] =
            0.1 * data[[i - 1, 0]] + 0.4 * data[[i - 1, 1]] + seed_b[i % seed_b.len()] * 0.5;
    }
    data
}

// ─── 1. HP/BK/CF Filters ────────────────────────────────────────────────────

#[test]
fn test_hp_filter() {
    let y = sine_with_trend(120);
    let (trend, cycle) = TimeSeries::hp_filter(&y, 1600.0).unwrap();

    assert_eq!(trend.len(), y.len());
    assert_eq!(cycle.len(), y.len());

    // Trend + cycle should reconstruct original
    for i in 0..y.len() {
        assert!((trend[i] + cycle[i] - y[i]).abs() < 1e-10);
    }

    // Trend should be smoother than original
    let y_var: f64 = y
        .iter()
        .map(|&v| (v - y.mean().unwrap()).powi(2))
        .sum::<f64>();
    let t_var: f64 = trend
        .iter()
        .map(|&v| (v - trend.mean().unwrap()).powi(2))
        .sum::<f64>();
    // With lambda=1600, trend should have some variation (it's not constant)
    assert!(t_var > 0.0);
    assert!(t_var < y_var); // trend is smoother
}

#[test]
fn test_hp_filter_short_series() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(TimeSeries::hp_filter(&y, 1600.0).is_err());
}

#[test]
fn test_bk_filter() {
    let y = sine_with_trend(120);
    let cycle = TimeSeries::bk_filter(&y, 6, 32, 12).unwrap();

    assert_eq!(cycle.len(), y.len());

    // First and last K values should be NaN
    assert!(cycle[0].is_nan());
    assert!(cycle[11].is_nan());
    assert!(cycle[119].is_nan());
    assert!(cycle[108].is_nan());

    // Interior values should be finite
    assert!(cycle[12].is_finite());
    assert!(cycle[107].is_finite());
}

#[test]
fn test_cf_filter() {
    let y = sine_with_trend(120);
    let cycle = TimeSeries::cf_filter(&y, 6, 32, false).unwrap();

    assert_eq!(cycle.len(), y.len());
    // All values should be finite
    assert!(cycle.iter().all(|v| v.is_finite()));
}

#[test]
fn test_cf_filter_with_drift() {
    let y = sine_with_trend(120);
    let cycle = TimeSeries::cf_filter(&y, 6, 32, true).unwrap();
    assert_eq!(cycle.len(), y.len());
    assert!(cycle.iter().all(|v| v.is_finite()));
}

// ─── 2. Seasonal Decomposition / STL ────────────────────────────────────────

#[test]
fn test_seasonal_decompose_additive() {
    let y = seasonal_series(120, 12);
    let result = Decomposition::seasonal_decompose(&y, 12, "additive").unwrap();

    assert_eq!(result.observed.len(), 120);
    assert_eq!(result.trend.len(), 120);
    assert_eq!(result.seasonal.len(), 120);
    assert_eq!(result.residual.len(), 120);
    assert_eq!(result.model, "additive");

    // Interior trend values should be finite
    assert!(result.trend[10].is_finite());

    // Seasonal should be periodic
    // Check that seasonal values at same position are equal
    let s0 = result.seasonal[12]; // position 0 in second cycle
    let s1 = result.seasonal[24]; // position 0 in third cycle
    assert!((s0 - s1).abs() < 1e-10);

    // Display works
    let display = format!("{}", result);
    assert!(display.contains("additive"));
}

#[test]
fn test_seasonal_decompose_multiplicative() {
    // Need positive data for multiplicative
    let y = Array1::from_vec(
        (0..120)
            .map(|i| {
                let t = i as f64;
                (50.0 + 0.1 * t) * (1.0 + 0.1 * (2.0 * std::f64::consts::PI * t / 12.0).sin())
            })
            .collect(),
    );
    let result = Decomposition::seasonal_decompose(&y, 12, "multiplicative").unwrap();
    assert_eq!(result.model, "multiplicative");
}

#[test]
fn test_stl() {
    let y = seasonal_series(120, 12);
    let result = Decomposition::stl(&y, 12, 7, 0).unwrap();

    assert_eq!(result.observed.len(), 120);
    assert_eq!(result.model, "STL");

    // Additive decomposition: observed ≈ trend + seasonal + residual
    for i in 0..120 {
        let reconstructed = result.trend[i] + result.seasonal[i] + result.residual[i];
        assert!(
            (reconstructed - y[i]).abs() < 1e-6,
            "STL reconstruction failed at i={}: {} vs {}",
            i,
            reconstructed,
            y[i]
        );
    }
}

// ─── 3. Exponential Smoothing ───────────────────────────────────────────────

#[test]
fn test_ses() {
    let y = ar1_series(100, 0.5);
    let result = ExponentialSmoothing::fit(&y, None, None, 0, false).unwrap();

    assert_eq!(result.n_obs, 100);
    assert!(result.alpha > 0.0 && result.alpha < 1.0);
    assert!(result.beta.is_none());
    assert!(result.gamma.is_none());
    assert!(result.sse.is_finite());
    assert!(result.aic.is_finite());

    // Forecast
    let forecast = result.predict(5);
    assert_eq!(forecast.len(), 5);
    // SES forecasts should be constant (flat)
    assert!((forecast[0] - forecast[4]).abs() < 1e-6);
}

#[test]
fn test_holt() {
    let y = Array1::from_vec((0..50).map(|i| 10.0 + 0.5 * i as f64).collect());
    let result = ExponentialSmoothing::fit(&y, Some("add"), None, 0, false).unwrap();

    assert!(result.beta.is_some());
    assert_eq!(result.trend_type, "add");

    let forecast = result.predict(3);
    assert_eq!(forecast.len(), 3);
    // Forecast should increase for trending data
    assert!(forecast[2] > forecast[0]);
}

#[test]
fn test_holt_winters_additive() {
    let y = seasonal_series(96, 12);
    let result = ExponentialSmoothing::fit(&y, Some("add"), Some("add"), 12, false).unwrap();

    assert!(result.gamma.is_some());
    assert_eq!(result.seasonal_type, "add");
    assert_eq!(result.seasonal_periods, 12);

    let forecast = result.predict(12);
    assert_eq!(forecast.len(), 12);
    assert!(forecast.iter().all(|v| v.is_finite()));

    // Display works
    let display = format!("{}", result);
    assert!(display.contains("ETS"));
}

#[test]
fn test_damped_trend() {
    let y = Array1::from_vec((0..50).map(|i| 10.0 + 0.5 * i as f64).collect());
    let result = ExponentialSmoothing::fit(&y, Some("add"), None, 0, true).unwrap();

    assert!(result.damped);
    assert!(result.phi.is_some());
    let phi = result.phi.unwrap();
    assert!(phi > 0.5 && phi <= 1.0);
}

// ─── 4. AutoReg / ARDL ─────────────────────────────────────────────────────

#[test]
fn test_autoreg() {
    let y = ar1_series(200, 0.7);
    let result = AutoReg::fit(&y, 2, None, "c").unwrap();

    assert!(result.r_squared >= 0.0);
    assert_eq!(result.lags, 2);
    assert_eq!(result.trend, "c");

    // First AR coefficient should be near 0.7
    // param_names: ["const", "y.L1", "y.L2"]
    assert_eq!(result.param_names[1], "y.L1");

    // Forecast
    let forecast = result.predict(&y, 5, None);
    assert_eq!(forecast.len(), 5);
    assert!(forecast.iter().all(|v| v.is_finite()));

    // Display
    let display = format!("{}", result);
    assert!(display.contains("AutoReg"));
}

#[test]
fn test_autoreg_no_trend() {
    let y = ar1_series(100, 0.5);
    let result = AutoReg::fit(&y, 1, None, "n").unwrap();
    assert_eq!(result.param_names[0], "y.L1");
}

#[test]
fn test_ardl() {
    let n = 200;
    let y = ar1_series(n, 0.5);
    let x = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.3 * y[i] + 0.2 * ((i * 7 + 5) % 13) as f64 / 13.0
    });

    let result = ARDL::fit(&y, &x, 2, 1).unwrap();

    assert!(result.r_squared >= 0.0);
    assert_eq!(result.y_lags, 2);
    assert_eq!(result.x_lags, 1);
    assert!(result.param_names.contains(&"x1".to_string()));
    assert!(result.param_names.contains(&"x1.L1".to_string()));

    let display = format!("{}", result);
    assert!(display.contains("ARDL"));
}

// ─── 5. VARMAX ──────────────────────────────────────────────────────────────

#[test]
fn test_varma_basic() {
    let data = var_data(100);
    let result = VARMA::fit(&data, 1, 1).unwrap();

    assert_eq!(result.n_vars, 2);
    assert_eq!(result.p_lags, 1);
    assert_eq!(result.q_lags, 1);
    assert!(result.aic.is_finite());
    assert!(result.exog_params.is_none());
    assert_eq!(result.n_exog, 0);
}

#[test]
fn test_varmax_with_exog() {
    let data = var_data(100);
    let exog = Array2::from_shape_fn((100, 1), |(i, _)| (i as f64) / 100.0);

    let result = VARMA::fit_with_exog(&data, 1, 1, Some(&exog)).unwrap();

    assert_eq!(result.n_exog, 1);
    assert!(result.exog_params.is_some());
    let ep = result.exog_params.as_ref().unwrap();
    assert_eq!(ep.nrows(), 1);
    assert_eq!(ep.ncols(), 2);

    let display = format!("{}", result);
    assert!(display.contains("VARMAX"));
}

// ─── 6. SVAR ────────────────────────────────────────────────────────────────

#[test]
fn test_svar_cholesky() {
    let data = var_data(100);
    let result = SVAR::fit(&data, 2, SVarIdentification::Cholesky).unwrap();

    assert_eq!(result.identification, "Cholesky");
    assert_eq!(result.a_matrix.nrows(), 2);
    assert_eq!(result.b_matrix.nrows(), 2);

    // A should be identity for Cholesky
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((result.a_matrix[[i, j]] - expected).abs() < 1e-10);
        }
    }

    // B should be lower triangular
    assert!(result.b_matrix[[0, 1]].abs() < 1e-10);

    // Structural IRF
    let irf = result.structural_irf(10).unwrap();
    assert_eq!(irf.shape(), &[10, 2, 2]);

    // Structural FEVD
    let fevd = result.structural_fevd(10).unwrap();
    assert_eq!(fevd.shape(), &[10, 2, 2]);
    // FEVD rows should sum to ~1
    for h in 0..10 {
        for i in 0..2 {
            let row_sum = fevd[[h, i, 0]] + fevd[[h, i, 1]];
            assert!(
                (row_sum - 1.0).abs() < 0.01,
                "FEVD row sum = {} at h={}, i={}",
                row_sum,
                h,
                i
            );
        }
    }

    let display = format!("{}", result);
    assert!(display.contains("SVAR"));
}

#[test]
fn test_svar_long_run() {
    let data = var_data(100);
    let c_mask = Array2::from_elem((2, 2), f64::NAN);
    let result = SVAR::fit(&data, 2, SVarIdentification::LongRun(c_mask)).unwrap();

    assert_eq!(result.identification, "Long-run restrictions");
    assert!(result.a_matrix[[0, 0]] > 0.0); // Identity
}

// ─── 7. State Space / Kalman Filter ─────────────────────────────────────────

#[test]
fn test_kalman_local_level() {
    // Local level model: y_t = s_t + e_t, s_t = s_{t-1} + u_t
    let model = StateSpaceModel {
        h: Array2::from_elem((1, 1), 1.0),
        f: Array2::from_elem((1, 1), 1.0),
        r: Array2::from_elem((1, 1), 1.0),
        q: Array2::from_elem((1, 1), 0.1),
        r_obs: Array2::from_elem((1, 1), 1.0),
        s0: Array1::zeros(1),
        p0: Array2::from_elem((1, 1), 10.0),
    };

    let y: Vec<Array1<f64>> = (0..50)
        .map(|i| Array1::from_vec(vec![5.0 + 0.3 * ((i * 7 + 3) % 11) as f64 / 11.0]))
        .collect();

    let filter_result = KalmanFilter::filter(&model, &y).unwrap();

    assert_eq!(filter_result.n_obs, 50);
    assert_eq!(filter_result.n_states, 1);
    assert_eq!(filter_result.filtered_states.len(), 50);
    assert!(filter_result.log_likelihood.is_finite());

    // Filtered states should converge toward observations
    let last_state = filter_result.filtered_states.last().unwrap()[0];
    assert!(last_state.is_finite());

    // Smoother
    let smooth_result = KalmanSmoother::smooth(&model, &filter_result).unwrap();
    assert_eq!(smooth_result.smoothed_states.len(), 50);
}

#[test]
fn test_state_space_estimate() {
    let model = StateSpaceModel {
        h: Array2::from_elem((1, 1), 1.0),
        f: Array2::from_elem((1, 1), 0.9),
        r: Array2::from_elem((1, 1), 1.0),
        q: Array2::from_elem((1, 1), 0.5),
        r_obs: Array2::from_elem((1, 1), 1.0),
        s0: Array1::zeros(1),
        p0: Array2::eye(1),
    };

    let y: Vec<Array1<f64>> = (0..30)
        .map(|i| Array1::from_vec(vec![3.0 + 0.1 * i as f64]))
        .collect();

    let result = state_space_estimate(&model, &y).unwrap();

    assert_eq!(result.n_obs, 30);
    assert!(result.log_likelihood.is_finite());
    assert_eq!(result.smoothed_states.len(), 30);

    // Predict
    let forecasts = result.predict(&model, 5);
    assert_eq!(forecasts.len(), 5);
    assert!(forecasts[0][0].is_finite());

    let display = format!("{}", result);
    assert!(display.contains("State Space"));
}

// ─── 8. Markov Switching ────────────────────────────────────────────────────

#[test]
fn test_markov_switching_2_regimes() {
    // Create regime-switching data
    let n = 200;
    let mut y = vec![0.0; n];
    let seed = [0.3, -0.2, 0.5, -0.1, 0.4, -0.3, 0.1, 0.2, -0.4, 0.6];

    for i in 1..n {
        // Low volatility regime for first half, high for second
        let regime_mean = if i < n / 2 { 2.0 } else { -1.0 };
        let regime_vol = if i < n / 2 { 0.3 } else { 0.8 };
        y[i] = regime_mean + 0.3 * y[i - 1] + regime_vol * seed[i % seed.len()];
    }
    let y = Array1::from_vec(y);

    let result = MarkovSwitching::fit(&y, 2, 1).unwrap();

    assert_eq!(result.n_regimes, 2);
    assert_eq!(result.ar_order, 1);
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());

    // Transition matrix should be K x K
    assert_eq!(result.transition_matrix.shape(), &[2, 2]);
    // Rows should sum to 1
    for i in 0..2 {
        let row_sum = result.transition_matrix[[i, 0]] + result.transition_matrix[[i, 1]];
        assert!((row_sum - 1.0).abs() < 1e-6);
    }

    // Filtered and smoothed probs should be T x K
    assert_eq!(result.filtered_probs.ncols(), 2);
    assert_eq!(result.smoothed_probs.ncols(), 2);

    // Expected durations
    let durations = result.expected_durations();
    assert_eq!(durations.len(), 2);
    assert!(durations[0] > 1.0);
    assert!(durations[1] > 1.0);

    // Forecast
    let forecast = result.predict(&y, 5);
    assert_eq!(forecast.len(), 5);
    assert!(forecast.iter().all(|v| v.is_finite()));

    // Display
    let display = format!("{}", result);
    assert!(display.contains("Markov Switching"));
    assert!(display.contains("Regime 0"));
    assert!(display.contains("Regime 1"));
}

#[test]
fn test_markov_switching_no_ar() {
    let n = 150;
    let mut y = vec![0.0; n];
    let seed = [0.3, -0.2, 0.5, -0.1, 0.4, -0.3, 0.1, 0.2, -0.4, 0.6];
    for i in 0..n {
        let mean = if i < n / 2 { 5.0 } else { -2.0 };
        y[i] = mean + seed[i % seed.len()];
    }
    let y = Array1::from_vec(y);

    let result = MarkovSwitching::fit(&y, 2, 0).unwrap();
    assert_eq!(result.ar_order, 0);
    assert_eq!(result.n_regimes, 2);
}
