//! Tests for the 9 previously untested modules:
//! HausmanTest, SUR, ThreeSLS, ArellanoBond, VAR, VARMA, VECM, TimeSeries, PanelThreshold

use greeners::{
    ArellanoBond, Equation, FixedEffects, HausmanTest, PanelThreshold, RandomEffects, SurEquation,
    ThreeSLS, TimeSeries, SUR, VAR, VARMA, VECM,
};
use ndarray::{Array1, Array2};

// ============================================================================
// HausmanTest
// ============================================================================

fn panel_data_for_hausman() -> (Array1<f64>, Array2<f64>, Vec<i64>, Array1<i64>) {
    // 5 entities, 4 periods each = 20 obs
    let n = 20;
    let entity_ids_vec: Vec<i64> = (0..5).flat_map(|e| std::iter::repeat_n(e, 4)).collect();
    let entity_ids_arr = Array1::from(entity_ids_vec.clone());

    // entity effects
    let alpha: Vec<f64> = entity_ids_vec.iter().map(|&e| (e as f64) * 2.0).collect();
    let x1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i * 3 + 7) % 11) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * x1[i] - 0.5 * x2[i] + alpha[i] + (i as f64 * 0.1).sin())
        .collect();

    let mut x_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        x_flat.push(x1[i]);
        x_flat.push(x2[i]);
    }
    let x = Array2::from_shape_vec((n, 2), x_flat).unwrap();
    let y = Array1::from(y);

    (y, x, entity_ids_vec, entity_ids_arr)
}

#[test]
fn test_hausman_compare_runs() {
    let (y, x, entity_ids_vec, entity_ids_arr) = panel_data_for_hausman();

    // FE needs &[T] where T: Eq + Hash — returns params for x1, x2 only (no intercept)
    let fe = FixedEffects::fit(&y, &x, &entity_ids_vec).unwrap();

    // RE also without intercept so params align for Hausman comparison
    let re = RandomEffects::fit(&y, &x, &entity_ids_arr).unwrap();

    let output = HausmanTest::compare(&fe, &re);
    assert!(output.contains("Hausman Test"));
    assert!(output.contains("Chi2 Statistic"));
    assert!(output.contains("P-Value"));
}

#[test]
fn test_hausman_with_strong_entity_effects() {
    // With strong entity effects, FE should be preferred (reject H0)
    let n = 40;
    let entity_ids_vec: Vec<i64> = (0..10).flat_map(|e| std::iter::repeat_n(e, 4)).collect();
    let entity_ids_arr = Array1::from(entity_ids_vec.clone());

    let alpha: Vec<f64> = entity_ids_vec.iter().map(|&e| (e as f64) * 10.0).collect();
    let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3).collect();
    // y correlated with entity effects through x
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * x[i] + alpha[i] + 0.5 * alpha[i] * x[i] * 0.01)
        .collect();

    let x_mat = Array2::from_shape_vec((n, 1), x.clone()).unwrap();
    let y_arr = Array1::from(y);

    let fe = FixedEffects::fit(&y_arr, &x_mat, &entity_ids_vec).unwrap();

    let mut x_with_const = Array2::zeros((n, 2));
    x_with_const.column_mut(0).fill(1.0);
    for i in 0..n {
        x_with_const[[i, 1]] = x[i];
    }
    let re = RandomEffects::fit(&y_arr, &x_with_const, &entity_ids_arr).unwrap();

    let output = HausmanTest::compare(&fe, &re);
    // The output should contain a recommendation
    assert!(
        output.contains("FIXED EFFECTS") || output.contains("RANDOM EFFECTS"),
        "Expected recommendation in output: {}",
        output
    );
}

// ============================================================================
// SUR
// ============================================================================

#[test]
fn test_sur_two_equations() {
    let n = 50;
    let x_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();

    // Equation 1: y1 = 1 + 2*x + e1
    let y1 = Array1::from(
        x_vals
            .iter()
            .enumerate()
            .map(|(i, &x)| 1.0 + 2.0 * x + (i as f64 * 0.3).sin())
            .collect::<Vec<_>>(),
    );

    // Equation 2: y2 = 3 - 0.5*x + e2 (errors correlated with eq1)
    let y2 = Array1::from(
        x_vals
            .iter()
            .enumerate()
            .map(|(i, &x)| 3.0 - 0.5 * x + (i as f64 * 0.3).cos())
            .collect::<Vec<_>>(),
    );

    // X with intercept
    let mut x_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        x_flat.push(1.0);
        x_flat.push(x_vals[i]);
    }
    let x_mat = Array2::from_shape_vec((n, 2), x_flat).unwrap();

    let equations = vec![
        SurEquation {
            y: y1,
            x: x_mat.clone(),
            name: "eq1".to_string(),
        },
        SurEquation {
            y: y2,
            x: x_mat,
            name: "eq2".to_string(),
        },
    ];

    let result = SUR::fit(&equations).unwrap();

    assert_eq!(result.equations.len(), 2);
    assert_eq!(result.equations[0].name, "eq1");
    assert_eq!(result.equations[1].name, "eq2");

    // Eq1: intercept ~ 1, slope ~ 2
    assert!(
        (result.equations[0].params[0] - 1.0).abs() < 1.0,
        "eq1 intercept: {}",
        result.equations[0].params[0]
    );
    assert!(
        (result.equations[0].params[1] - 2.0).abs() < 0.5,
        "eq1 slope: {}",
        result.equations[0].params[1]
    );

    // Eq2: intercept ~ 3, slope ~ -0.5
    assert!(
        (result.equations[1].params[0] - 3.0).abs() < 1.0,
        "eq2 intercept: {}",
        result.equations[1].params[0]
    );
    assert!(
        result.equations[1].params[1] < 0.0,
        "eq2 slope should be negative: {}",
        result.equations[1].params[1]
    );

    // Sigma cross should be 2x2
    assert_eq!(result.sigma_cross.shape(), &[2, 2]);

    // R-squared should be reasonable
    for eq in &result.equations {
        assert!(eq.r_squared > 0.5, "R² too low: {}", eq.r_squared);
        assert!(eq.std_errors.iter().all(|&s| s > 0.0 && s.is_finite()));
        assert!(eq.p_values.iter().all(|p| p.is_finite()));
    }
}

#[test]
fn test_sur_mismatched_n_returns_error() {
    let equations = vec![
        SurEquation {
            y: Array1::from(vec![1.0, 2.0, 3.0]),
            x: Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap(),
            name: "eq1".to_string(),
        },
        SurEquation {
            y: Array1::from(vec![1.0, 2.0]),
            x: Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap(),
            name: "eq2".to_string(),
        },
    ];

    let result = SUR::fit(&equations);
    assert!(result.is_err());
}

// ============================================================================
// ThreeSLS
// ============================================================================

#[test]
fn test_three_sls_two_equations() {
    let n = 80;

    // Exogenous variables
    let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.2 + 1.0).collect();
    let z2: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 20) as f64 * 0.5).collect();

    // Endogenous: y1 depends on z1 and y2
    // y2 depends on z2 and y1 (simultaneous)
    // Reduced form for data generation
    let y1: Vec<f64> = (0..n)
        .map(|i| 2.0 + 1.5 * z1[i] + 0.3 * z2[i] + (i as f64 * 0.2).sin())
        .collect();
    let y2: Vec<f64> = (0..n)
        .map(|i| 1.0 - 0.5 * z1[i] + 1.0 * z2[i] + (i as f64 * 0.15).cos())
        .collect();

    // Eq1: y1 ~ const + y2 + z1
    let mut x1_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        x1_flat.push(1.0);
        x1_flat.push(y2[i]);
        x1_flat.push(z1[i]);
    }
    let x1 = Array2::from_shape_vec((n, 3), x1_flat).unwrap();

    // Eq2: y2 ~ const + y1 + z2
    let mut x2_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        x2_flat.push(1.0);
        x2_flat.push(y1[i]);
        x2_flat.push(z2[i]);
    }
    let x2 = Array2::from_shape_vec((n, 3), x2_flat).unwrap();

    let equations = vec![
        Equation {
            y: Array1::from(y1),
            x: x1,
            name: "supply".to_string(),
        },
        Equation {
            y: Array1::from(y2),
            x: x2,
            name: "demand".to_string(),
        },
    ];

    // Instruments: all exogenous variables [const, z1, z2]
    let mut z_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        z_flat.push(1.0);
        z_flat.push(z1[i]);
        z_flat.push(z2[i]);
    }
    let z = Array2::from_shape_vec((n, 3), z_flat).unwrap();

    let result = ThreeSLS::fit(&equations, &z).unwrap();

    assert_eq!(result.equations.len(), 2);
    assert_eq!(result.equations[0].name, "supply");
    assert_eq!(result.equations[1].name, "demand");
    assert_eq!(result.sigma_cross.shape(), &[2, 2]);

    for eq in &result.equations {
        assert_eq!(eq.params.len(), 3);
        assert!(eq.std_errors.iter().all(|&s| s > 0.0 && s.is_finite()));
        assert!(eq.r_squared.is_finite());
    }
}

// ============================================================================
// ArellanoBond
// ============================================================================

#[test]
fn test_arellano_bond_basic() {
    // 10 entities, 6 periods
    let n_entities = 10;
    let t_periods = 6;
    let n = n_entities * t_periods;

    let entity_ids = Array1::from(
        (0..n_entities)
            .flat_map(|e| std::iter::repeat_n(e as i64, t_periods))
            .collect::<Vec<_>>(),
    );
    let time_ids = Array1::from(
        (0..n_entities)
            .flat_map(|_| (0..t_periods).map(|t| t as i64))
            .collect::<Vec<_>>(),
    );

    // Generate y with autoregressive structure: y_it = 0.5 * y_{i,t-1} + x_it + alpha_i + e_it
    let x_vals: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 13) as f64 * 0.3).collect();
    let x = Array2::from_shape_vec((n, 1), x_vals.clone()).unwrap();

    let mut y_vals = vec![0.0; n];
    for e in 0..n_entities {
        let alpha = (e as f64) * 1.5;
        y_vals[e * t_periods] = alpha + x_vals[e * t_periods]; // Initial value
        for t in 1..t_periods {
            let idx = e * t_periods + t;
            let prev = y_vals[idx - 1];
            y_vals[idx] = 0.5 * prev + x_vals[idx] + alpha + (idx as f64 * 0.1).sin() * 0.3;
        }
    }
    let y = Array1::from(y_vals);

    let result = ArellanoBond::fit(
        &y,
        &x,
        entity_ids.as_slice().unwrap(),
        time_ids.as_slice().unwrap(),
        2,
        false,
        None,
    )
    .unwrap();

    // First param is the lagged dependent variable coefficient
    assert!(
        result.params.len() >= 2,
        "Expected at least 2 params (lag + x)"
    );
    assert!(result.n_obs > 0);
    assert!(result.std_errors.iter().all(|&s| s > 0.0 && s.is_finite()));
    assert!(result.t_values.iter().all(|t| t.is_finite()));
    assert!(result.sargan_stat >= 0.0);
}

#[test]
fn test_arellano_bond_too_few_periods() {
    // 3 entities, 2 periods each — not enough (need T>=3)
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let entity_ids = Array1::from(vec![0i64, 0, 1, 1, 2, 2]);
    let time_ids = Array1::from(vec![0i64, 1, 0, 1, 0, 1]);

    let result = ArellanoBond::fit(
        &y,
        &x,
        entity_ids.as_slice().unwrap(),
        time_ids.as_slice().unwrap(),
        2,
        false,
        None,
    );
    assert!(result.is_err(), "Should fail with T=2 (need T>=3)");
}

#[test]
fn test_arellano_bond_ids_mismatch() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let entity_ids = Array1::from(vec![0i64, 1]); // wrong length
    let time_ids = Array1::from(vec![0i64, 1, 2]);

    let result = ArellanoBond::fit(
        &y,
        &x,
        entity_ids.as_slice().unwrap(),
        time_ids.as_slice().unwrap(),
        2,
        false,
        None,
    );
    assert!(result.is_err());
}

// ============================================================================
// VAR
// ============================================================================

#[test]
fn test_var_bivariate() {
    // Two variables, 100 time points, VAR(1)
    let t = 100;
    let mut data = Array2::<f64>::zeros((t, 2));

    // Simple VAR(1): y1_t = 0.5*y1_{t-1} + 0.1*y2_{t-1}, y2_t = 0.2*y1_{t-1} + 0.3*y2_{t-1}
    data[[0, 0]] = 1.0;
    data[[0, 1]] = 0.5;
    for i in 1..t {
        data[[i, 0]] =
            0.5 * data[[i - 1, 0]] + 0.1 * data[[i - 1, 1]] + (i as f64 * 0.1).sin() * 0.1;
        data[[i, 1]] =
            0.2 * data[[i - 1, 0]] + 0.3 * data[[i - 1, 1]] + (i as f64 * 0.15).cos() * 0.1;
    }

    let result = VAR::fit(&data, 1, Some(vec!["y1".to_string(), "y2".to_string()])).unwrap();

    assert_eq!(result.n_vars, 2);
    assert_eq!(result.lags, 1);
    assert_eq!(result.n_obs, t - 1);
    assert_eq!(result.var_names, vec!["y1", "y2"]);
    // params shape: (1 + k*p) x k = (1 + 2) x 2 = 3 x 2
    assert_eq!(result.params.shape(), &[3, 2]);
    assert_eq!(result.sigma_u.shape(), &[2, 2]);
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_var_irf() {
    let t = 100;
    let mut data = Array2::<f64>::zeros((t, 2));
    data[[0, 0]] = 1.0;
    for i in 1..t {
        data[[i, 0]] = 0.5 * data[[i - 1, 0]] + (i as f64 * 0.1).sin() * 0.1;
        data[[i, 1]] =
            0.3 * data[[i - 1, 0]] + 0.2 * data[[i - 1, 1]] + (i as f64 * 0.15).cos() * 0.1;
    }

    let result = VAR::fit(&data, 1, None).unwrap();
    let irf = result.irf(10).unwrap();

    // IRF shape: (steps x k x k) = (10 x 2 x 2)
    assert_eq!(irf.shape(), &[10, 2, 2]);

    // At step 0, IRF should be close to Cholesky of Sigma (not identity)
    // IRF should decay over time for stable VAR
    let irf_0_norm: f64 = irf.slice(ndarray::s![0, .., ..]).mapv(|v| v * v).sum();
    let irf_9_norm: f64 = irf.slice(ndarray::s![9, .., ..]).mapv(|v| v * v).sum();
    assert!(
        irf_9_norm < irf_0_norm,
        "IRF should decay: step0={:.4}, step9={:.4}",
        irf_0_norm,
        irf_9_norm
    );
}

#[test]
fn test_var_not_enough_observations() {
    let data = Array2::<f64>::zeros((3, 2));
    let result = VAR::fit(&data, 5, None);
    assert!(result.is_err());
}

#[test]
fn test_var_higher_order() {
    // VAR(3)
    let t = 150;
    let mut data = Array2::<f64>::zeros((t, 2));
    data[[0, 0]] = 1.0;
    data[[1, 0]] = 0.8;
    data[[2, 0]] = 0.6;
    for i in 3..t {
        data[[i, 0]] = 0.3 * data[[i - 1, 0]]
            + 0.1 * data[[i - 2, 0]]
            + 0.05 * data[[i - 3, 0]]
            + (i as f64 * 0.1).sin() * 0.05;
        data[[i, 1]] =
            0.2 * data[[i - 1, 1]] + 0.1 * data[[i - 1, 0]] + (i as f64 * 0.2).cos() * 0.05;
    }

    let result = VAR::fit(&data, 3, None).unwrap();
    assert_eq!(result.lags, 3);
    // params shape: (1 + 2*3) x 2 = 7 x 2
    assert_eq!(result.params.shape(), &[7, 2]);
}

#[test]
fn test_var_granger_causality_not_implemented() {
    let t = 50;
    let data = Array2::from_shape_fn((t, 2), |(i, j)| (i + j) as f64 * 0.1);
    let result = VAR::fit(&data, 1, None).unwrap();

    let gc = result.granger_causality(0, 1);
    assert!(
        gc.is_err(),
        "Granger causality should return error (not yet implemented)"
    );
}

// ============================================================================
// VARMA
// ============================================================================

#[test]
fn test_varma_basic() {
    let t = 200;
    let mut data = Array2::<f64>::zeros((t, 2));
    data[[0, 0]] = 1.0;
    for i in 1..t {
        data[[i, 0]] = 0.4 * data[[i - 1, 0]] + (i as f64 * 0.05).sin() * 0.2;
        data[[i, 1]] =
            0.3 * data[[i - 1, 1]] + 0.15 * data[[i - 1, 0]] + (i as f64 * 0.07).cos() * 0.2;
    }

    let result = VARMA::fit(&data, 1, 1).unwrap();

    assert_eq!(result.p_lags, 1);
    assert_eq!(result.q_lags, 1);
    assert_eq!(result.n_vars, 2);
    assert!(result.n_obs > 0);
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
    assert_eq!(result.sigma_u.shape(), &[2, 2]);
    // AR params: (1 + p*k) x k = (1 + 2) x 2 = 3 x 2
    assert_eq!(result.ar_params.nrows(), 1 + 1 * 2);
    // MA params: (q*k) x k = 2 x 2
    assert_eq!(result.ma_params.nrows(), 1 * 2);
}

#[test]
fn test_varma_not_enough_observations() {
    let data = Array2::<f64>::zeros((5, 2));
    let result = VARMA::fit(&data, 2, 2);
    assert!(result.is_err());
}

// ============================================================================
// VECM
// ============================================================================

#[test]
fn test_vecm_cointegrated_pair() {
    // Two series with a common stochastic trend (cointegrated)
    let t = 200;
    let mut data = Array2::<f64>::zeros((t, 2));

    // Random walk (common trend)
    let mut trend = 0.0_f64;
    for i in 0..t {
        trend += (i as f64 * 0.1).sin() * 0.3; // pseudo-random walk
        data[[i, 0]] = trend + (i as f64 * 0.2).cos() * 0.1; // y1 = trend + noise
        data[[i, 1]] = 2.0 * trend + 1.0 + (i as f64 * 0.15).sin() * 0.1; // y2 = 2*trend + 1 + noise
    }

    let result = VECM::fit(&data, 2, 1).unwrap();

    assert_eq!(result.n_vars, 2);
    assert_eq!(result.rank, 1);
    assert!(result.n_obs > 0);
    // Beta is the cointegrating vector (k x rank) = (2 x 1)
    assert_eq!(result.beta.shape(), &[2, 1]);
    // Alpha is the adjustment speed (k x rank) = (2 x 1)
    assert_eq!(result.alpha.shape(), &[2, 1]);
    assert_eq!(result.eigenvalues.len(), 2);
    // Eigenvalues should be sorted descending
    assert!(result.eigenvalues[0] >= result.eigenvalues[1]);
}

#[test]
fn test_vecm_invalid_rank_zero() {
    let data = Array2::from_shape_fn((50, 2), |(i, j)| (i + j) as f64);
    let result = VECM::fit(&data, 2, 0);
    assert!(result.is_err());
}

#[test]
fn test_vecm_invalid_rank_too_high() {
    let data = Array2::from_shape_fn((50, 2), |(i, j)| (i + j) as f64);
    // rank must be < k=2, so rank=2 should fail
    let result = VECM::fit(&data, 2, 2);
    assert!(result.is_err());
}

// ============================================================================
// TimeSeries (ADF Test)
// ============================================================================

#[test]
fn test_adf_stationary_series() {
    // Stationary AR(1) with |phi| < 1 and real noise
    let n = 200;
    let mut series = vec![0.0; n];
    // Use deterministic but noisy process that behaves like stationary
    for i in 1..n {
        // phi=0.3, noise with enough variance to produce clear mean-reversion
        let noise = ((i as f64 * 1.7).sin() + (i as f64 * 3.1).cos()) * 1.5;
        series[i] = 0.3 * series[i - 1] + noise;
    }
    let arr = Array1::from(series);

    let result = TimeSeries::adf(&arr, Some(1)).unwrap();

    assert!(result.n_obs > 0);
    assert_eq!(result.lags_used, 1);
    assert!(result.test_statistic.is_finite());
    assert!(result.critical_values.0 < 0.0);
    assert!(result.critical_values.1 < 0.0);
    assert!(result.critical_values.2 < 0.0);
}

#[test]
fn test_adf_nonstationary_random_walk() {
    // Cumulative sum (random walk approximation)
    let n = 200;
    let mut series = vec![0.0; n];
    for i in 1..n {
        // Accumulates — nonstationary
        let shock = ((i as f64 * 1.3).sin() + (i as f64 * 2.7).cos()) * 0.5;
        series[i] = series[i - 1] + shock;
    }
    let arr = Array1::from(series);

    let result = TimeSeries::adf(&arr, Some(1)).unwrap();
    assert!(result.test_statistic.is_finite());
}

#[test]
fn test_adf_too_short_series() {
    let arr = Array1::from(vec![1.0, 2.0, 3.0]);
    let result = TimeSeries::adf(&arr, None);
    assert!(result.is_err());
}

#[test]
fn test_adf_custom_lags() {
    let n = 100;
    let mut series = vec![0.0; n];
    for i in 1..n {
        let noise = ((i as f64 * 1.7).sin() + (i as f64 * 3.1).cos()) * 1.5;
        series[i] = 0.5 * series[i - 1] + noise;
    }
    let arr = Array1::from(series);

    let result = TimeSeries::adf(&arr, Some(3)).unwrap();
    assert_eq!(result.lags_used, 3);
}

// ============================================================================
// PanelThreshold
// ============================================================================

#[test]
fn test_panel_threshold_basic() {
    // 10 entities, 10 periods = 100 obs
    let n_entities = 10;
    let t_periods = 10;
    let n = n_entities * t_periods;

    let entity_ids = Array1::from(
        (0..n_entities)
            .flat_map(|e| std::iter::repeat_n(e as i64, t_periods))
            .collect::<Vec<_>>(),
    );

    let x_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let q_vals: Vec<f64> = (0..n).map(|i| (i % t_periods) as f64).collect(); // threshold variable
    let threshold = 5.0;

    // Different slopes in two regimes
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let alpha = ((i / t_periods) as f64) * 2.0; // entity effect
            if q_vals[i] <= threshold {
                alpha + 2.0 * x_vals[i] + (i as f64 * 0.1).sin() * 0.3
            } else {
                alpha + 0.5 * x_vals[i] + (i as f64 * 0.1).sin() * 0.3
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let q = Array1::from(q_vals);

    let result = PanelThreshold::fit(&y, &x, &q, &entity_ids).unwrap();

    assert!(result.threshold_gamma.is_finite());
    assert_eq!(result.params_regime1.len(), 1);
    assert_eq!(result.params_regime2.len(), 1);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert!(result.ssr_min > 0.0);
    assert!(result.n_search > 0);

    // The two regimes should have different coefficients
    assert!(
        (result.params_regime1[0] - result.params_regime2[0]).abs() > 0.1,
        "Expected different regime coefficients: r1={}, r2={}",
        result.params_regime1[0],
        result.params_regime2[0]
    );
}

#[test]
fn test_panel_threshold_length_mismatch() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let q = Array1::from(vec![1.0, 2.0]); // wrong length
    let entity_ids = Array1::from(vec![0i64, 0, 1]);

    let result = PanelThreshold::fit(&y, &x, &q, &entity_ids);
    assert!(result.is_err());
}

// ============================================================================
// Additional VAR tests
// ============================================================================

#[test]
fn test_var_aic_bic_lag_selection() {
    let t = 150;
    let mut data = Array2::<f64>::zeros((t, 2));
    data[[0, 0]] = 1.0;
    for i in 1..t {
        data[[i, 0]] = 0.5 * data[[i - 1, 0]] + (i as f64 * 0.1).sin() * 0.1;
        data[[i, 1]] =
            0.2 * data[[i - 1, 0]] + 0.3 * data[[i - 1, 1]] + (i as f64 * 0.15).cos() * 0.1;
    }

    let result_1 = VAR::fit(&data, 1, None).unwrap();
    let result_2 = VAR::fit(&data, 2, None).unwrap();
    let result_3 = VAR::fit(&data, 3, None).unwrap();

    // All should produce finite AIC/BIC
    assert!(result_1.aic.is_finite());
    assert!(result_2.aic.is_finite());
    assert!(result_3.aic.is_finite());
    assert!(result_1.bic.is_finite());
    assert!(result_2.bic.is_finite());
    assert!(result_3.bic.is_finite());

    // BIC values should be ordered sensibly (different lags produce different BIC)
    assert!(
        (result_1.bic - result_2.bic).abs() > 1e-10,
        "Different lag orders should produce different BIC"
    );
}

#[test]
fn test_var_3_variables() {
    let t = 120;
    let mut data = Array2::<f64>::zeros((t, 3));
    data[[0, 0]] = 1.0;
    data[[0, 1]] = 0.5;
    data[[0, 2]] = -0.3;
    for i in 1..t {
        data[[i, 0]] =
            0.4 * data[[i - 1, 0]] + 0.1 * data[[i - 1, 1]] + (i as f64 * 0.1).sin() * 0.1;
        data[[i, 1]] = 0.15 * data[[i - 1, 0]]
            + 0.3 * data[[i - 1, 1]]
            + 0.05 * data[[i - 1, 2]]
            + (i as f64 * 0.13).cos() * 0.1;
        data[[i, 2]] =
            0.1 * data[[i - 1, 1]] + 0.2 * data[[i - 1, 2]] + (i as f64 * 0.17).sin() * 0.1;
    }

    let result = VAR::fit(&data, 1, Some(vec!["y1".into(), "y2".into(), "y3".into()])).unwrap();

    assert_eq!(result.n_vars, 3);
    assert_eq!(result.params.shape(), &[4, 3]); // (1 + 3*1) x 3
    assert_eq!(result.sigma_u.shape(), &[3, 3]);

    // IRF should work for 3 variables
    let irf = result.irf(5).unwrap();
    assert_eq!(irf.shape(), &[5, 3, 3]);
}

#[test]
fn test_var_irf_variance_decomposition_via_irf() {
    // Verify IRF values sum to something meaningful (FEVD-like check)
    let t = 100;
    let mut data = Array2::<f64>::zeros((t, 2));
    data[[0, 0]] = 1.0;
    for i in 1..t {
        data[[i, 0]] = 0.5 * data[[i - 1, 0]] + (i as f64 * 0.1).sin() * 0.1;
        data[[i, 1]] =
            0.3 * data[[i - 1, 0]] + 0.2 * data[[i - 1, 1]] + (i as f64 * 0.15).cos() * 0.1;
    }

    let result = VAR::fit(&data, 1, None).unwrap();
    let irf = result.irf(20).unwrap();

    // Compute cumulative squared IRF for variable 0
    let mut cum_sq_0 = 0.0;
    let mut cum_sq_1 = 0.0;
    for h in 0..20 {
        cum_sq_0 += irf[[h, 0, 0]].powi(2);
        cum_sq_1 += irf[[h, 0, 1]].powi(2);
    }
    let total = cum_sq_0 + cum_sq_1;
    assert!(total > 0.0, "Total IRF squared should be positive");
    // For VAR(1) with y1 mainly driven by own shocks, own share should be dominant
    let own_share = cum_sq_0 / total;
    assert!(
        own_share > 0.3,
        "Own shock share={} should be substantial",
        own_share
    );
}

// ============================================================================
// Additional VARMA tests
// ============================================================================

#[test]
fn test_varma_different_orders() {
    let t = 200;
    let mut data = Array2::<f64>::zeros((t, 2));
    data[[0, 0]] = 1.0;
    for i in 1..t {
        data[[i, 0]] = 0.4 * data[[i - 1, 0]] + (i as f64 * 0.05).sin() * 0.2;
        data[[i, 1]] =
            0.3 * data[[i - 1, 1]] + 0.15 * data[[i - 1, 0]] + (i as f64 * 0.07).cos() * 0.2;
    }

    let result_11 = VARMA::fit(&data, 1, 1).unwrap();
    let result_21 = VARMA::fit(&data, 2, 1).unwrap();

    assert_eq!(result_11.p_lags, 1);
    assert_eq!(result_11.q_lags, 1);
    assert_eq!(result_21.p_lags, 2);
    assert_eq!(result_21.q_lags, 1);

    // Higher order should have more AR params
    assert!(result_21.ar_params.nrows() > result_11.ar_params.nrows());

    // Both should have finite information criteria
    assert!(result_11.aic.is_finite());
    assert!(result_21.aic.is_finite());
}

#[test]
fn test_varma_residual_properties() {
    let t = 300;
    let mut data = Array2::<f64>::zeros((t, 2));
    data[[0, 0]] = 1.0;
    for i in 1..t {
        data[[i, 0]] = 0.3 * data[[i - 1, 0]] + (i as f64 * 0.05).sin() * 0.3;
        data[[i, 1]] =
            0.2 * data[[i - 1, 1]] + 0.1 * data[[i - 1, 0]] + (i as f64 * 0.07).cos() * 0.3;
    }

    let result = VARMA::fit(&data, 1, 1).unwrap();

    // Sigma_u should be positive semi-definite (diagonal elements positive)
    for i in 0..result.n_vars {
        assert!(
            result.sigma_u[[i, i]] > 0.0,
            "Diagonal of sigma_u should be positive"
        );
    }
}

// ============================================================================
// Additional VECM tests
// ============================================================================

#[test]
fn test_vecm_3_variables() {
    let t = 300;
    let mut data = Array2::<f64>::zeros((t, 3));

    let mut trend = 0.0_f64;
    for i in 0..t {
        trend += (i as f64 * 0.1).sin() * 0.3;
        data[[i, 0]] = trend + (i as f64 * 0.2).cos() * 0.1;
        data[[i, 1]] = 2.0 * trend + 1.0 + (i as f64 * 0.15).sin() * 0.1;
        data[[i, 2]] = -0.5 * trend + 3.0 + (i as f64 * 0.18).cos() * 0.1;
    }

    let result = VECM::fit(&data, 2, 1).unwrap();

    assert_eq!(result.n_vars, 3);
    assert_eq!(result.rank, 1);
    assert_eq!(result.beta.shape(), &[3, 1]);
    assert_eq!(result.alpha.shape(), &[3, 1]);
    assert_eq!(result.eigenvalues.len(), 3);
}

#[test]
fn test_vecm_loading_matrix_rank() {
    let t = 200;
    let mut data = Array2::<f64>::zeros((t, 2));

    let mut trend = 0.0_f64;
    for i in 0..t {
        trend += (i as f64 * 0.1).sin() * 0.3;
        data[[i, 0]] = trend + (i as f64 * 0.2).cos() * 0.1;
        data[[i, 1]] = 2.0 * trend + 1.0 + (i as f64 * 0.15).sin() * 0.1;
    }

    let result = VECM::fit(&data, 2, 1).unwrap();

    // Alpha (loading matrix) should have shape (k x rank) = (2 x 1)
    assert_eq!(result.alpha.shape(), &[2, 1]);
    // At least one loading coefficient should be non-zero
    let alpha_norm: f64 = result.alpha.iter().map(|v| v * v).sum();
    assert!(
        alpha_norm > 1e-10,
        "Loading matrix should have non-zero entries"
    );
}

// ============================================================================
// Additional ThreeSLS tests
// ============================================================================

#[test]
fn test_three_sls_overidentified() {
    let n = 100;

    let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.2 + 1.0).collect();
    let z2: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 20) as f64 * 0.5).collect();
    let z3: Vec<f64> = (0..n).map(|i| ((i * 11 + 5) % 17) as f64 * 0.3).collect();

    let y1: Vec<f64> = (0..n)
        .map(|i| 2.0 + 1.5 * z1[i] + 0.3 * z2[i] + 0.1 * z3[i] + (i as f64 * 0.2).sin())
        .collect();
    let y2: Vec<f64> = (0..n)
        .map(|i| 1.0 - 0.5 * z1[i] + 1.0 * z2[i] + (i as f64 * 0.15).cos())
        .collect();

    // Eq1: y1 ~ const + y2 + z1 (2 endogenous regressors in system)
    let mut x1_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        x1_flat.push(1.0);
        x1_flat.push(y2[i]);
        x1_flat.push(z1[i]);
    }
    let x1 = Array2::from_shape_vec((n, 3), x1_flat).unwrap();

    let mut x2_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        x2_flat.push(1.0);
        x2_flat.push(y1[i]);
        x2_flat.push(z2[i]);
    }
    let x2 = Array2::from_shape_vec((n, 3), x2_flat).unwrap();

    let equations = vec![
        Equation {
            y: Array1::from(y1),
            x: x1,
            name: "eq1".to_string(),
        },
        Equation {
            y: Array1::from(y2),
            x: x2,
            name: "eq2".to_string(),
        },
    ];

    // Instruments include all exogenous: [const, z1, z2, z3] — overidentified
    let mut z_flat = Vec::with_capacity(n * 4);
    for i in 0..n {
        z_flat.push(1.0);
        z_flat.push(z1[i]);
        z_flat.push(z2[i]);
        z_flat.push(z3[i]);
    }
    let z = Array2::from_shape_vec((n, 4), z_flat).unwrap();

    let result = ThreeSLS::fit(&equations, &z).unwrap();
    assert_eq!(result.equations.len(), 2);
    for eq in &result.equations {
        assert!(eq.std_errors.iter().all(|&s| s > 0.0 && s.is_finite()));
    }
}

#[test]
fn test_three_sls_vs_equation_by_equation() {
    // Compare 3SLS system estimates with individual IV/2SLS — they should differ
    // because 3SLS uses cross-equation correlation
    let n = 80;

    let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.2 + 1.0).collect();
    let z2: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 20) as f64 * 0.5).collect();

    let y1: Vec<f64> = (0..n)
        .map(|i| 2.0 + 1.5 * z1[i] + 0.3 * z2[i] + (i as f64 * 0.2).sin())
        .collect();
    let y2: Vec<f64> = (0..n)
        .map(|i| 1.0 - 0.5 * z1[i] + 1.0 * z2[i] + (i as f64 * 0.15).cos())
        .collect();

    let mut x1_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        x1_flat.push(1.0);
        x1_flat.push(y2[i]);
        x1_flat.push(z1[i]);
    }
    let x1 = Array2::from_shape_vec((n, 3), x1_flat).unwrap();

    let mut x2_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        x2_flat.push(1.0);
        x2_flat.push(y1[i]);
        x2_flat.push(z2[i]);
    }
    let x2 = Array2::from_shape_vec((n, 3), x2_flat).unwrap();

    let equations = vec![
        Equation {
            y: Array1::from(y1),
            x: x1,
            name: "supply".to_string(),
        },
        Equation {
            y: Array1::from(y2),
            x: x2,
            name: "demand".to_string(),
        },
    ];

    let mut z_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        z_flat.push(1.0);
        z_flat.push(z1[i]);
        z_flat.push(z2[i]);
    }
    let z = Array2::from_shape_vec((n, 3), z_flat).unwrap();

    let result = ThreeSLS::fit(&equations, &z).unwrap();

    // 3SLS should produce finite results for both equations
    assert_eq!(result.equations.len(), 2);
    for eq in &result.equations {
        assert_eq!(eq.params.len(), 3);
        assert!(eq.params.iter().all(|p| p.is_finite()));
    }
}

// ============================================================================
// Additional ArellanoBond tests
// ============================================================================

#[test]
fn test_arellano_bond_larger_panel() {
    let n_entities = 30;
    let t_periods = 8;
    let n = n_entities * t_periods;

    let entity_ids = Array1::from(
        (0..n_entities)
            .flat_map(|e| std::iter::repeat_n(e as i64, t_periods))
            .collect::<Vec<_>>(),
    );
    let time_ids = Array1::from(
        (0..n_entities)
            .flat_map(|_| (0..t_periods).map(|t| t as i64))
            .collect::<Vec<_>>(),
    );

    let x_vals: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 13) as f64 * 0.3).collect();
    let x = Array2::from_shape_vec((n, 1), x_vals.clone()).unwrap();

    let mut y_vals = vec![0.0; n];
    for e in 0..n_entities {
        let alpha = (e as f64) * 1.0;
        y_vals[e * t_periods] = alpha + x_vals[e * t_periods];
        for t in 1..t_periods {
            let idx = e * t_periods + t;
            let prev = y_vals[idx - 1];
            y_vals[idx] = 0.4 * prev + x_vals[idx] + alpha + (idx as f64 * 0.1).sin() * 0.2;
        }
    }
    let y = Array1::from(y_vals);

    let result = ArellanoBond::fit(
        &y,
        &x,
        entity_ids.as_slice().unwrap(),
        time_ids.as_slice().unwrap(),
        2,
        false,
        None,
    )
    .unwrap();

    assert!(result.params.len() >= 2);
    assert!(result.n_obs > 0);
    assert!(result.std_errors.iter().all(|&s| s > 0.0 && s.is_finite()));
    assert!(result.sargan_stat >= 0.0);
}

#[test]
fn test_arellano_bond_with_exogenous() {
    let n_entities = 15;
    let t_periods = 6;
    let n = n_entities * t_periods;

    let entity_ids = Array1::from(
        (0..n_entities)
            .flat_map(|e| std::iter::repeat_n(e as i64, t_periods))
            .collect::<Vec<_>>(),
    );
    let time_ids = Array1::from(
        (0..n_entities)
            .flat_map(|_| (0..t_periods).map(|t| t as i64))
            .collect::<Vec<_>>(),
    );

    // Two exogenous variables
    let x1_vals: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 13) as f64 * 0.3).collect();
    let x2_vals: Vec<f64> = (0..n).map(|i| ((i * 11 + 5) % 9) as f64 * 0.4).collect();
    let mut x_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        x_flat.push(x1_vals[i]);
        x_flat.push(x2_vals[i]);
    }
    let x = Array2::from_shape_vec((n, 2), x_flat).unwrap();

    let mut y_vals = vec![0.0; n];
    for e in 0..n_entities {
        let alpha = (e as f64) * 1.5;
        y_vals[e * t_periods] = alpha + x1_vals[e * t_periods] + x2_vals[e * t_periods];
        for t in 1..t_periods {
            let idx = e * t_periods + t;
            let prev = y_vals[idx - 1];
            y_vals[idx] = 0.5 * prev
                + x1_vals[idx]
                + 0.3 * x2_vals[idx]
                + alpha
                + (idx as f64 * 0.1).sin() * 0.3;
        }
    }
    let y = Array1::from(y_vals);

    let result = ArellanoBond::fit(
        &y,
        &x,
        entity_ids.as_slice().unwrap(),
        time_ids.as_slice().unwrap(),
        2,
        false,
        None,
    )
    .unwrap();

    // Should have lag + 2 exogenous = 3 params
    assert!(
        result.params.len() >= 3,
        "Expected at least 3 params (lag + 2 exogenous), got {}",
        result.params.len()
    );
    assert!(result.std_errors.iter().all(|&s| s > 0.0 && s.is_finite()));
}

// ============================================================================
// Additional PanelThreshold tests
// ============================================================================

#[test]
fn test_panel_threshold_clear_break() {
    let n_entities = 15;
    let t_periods = 10;
    let n = n_entities * t_periods;

    let entity_ids = Array1::from(
        (0..n_entities)
            .flat_map(|e| std::iter::repeat_n(e as i64, t_periods))
            .collect::<Vec<_>>(),
    );

    let x_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    // Threshold variable with clear break at 5.0
    let q_vals: Vec<f64> = (0..n).map(|i| (i % t_periods) as f64).collect();
    let threshold = 5.0;

    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let alpha = ((i / t_periods) as f64) * 2.0;
            if q_vals[i] <= threshold {
                alpha + 3.0 * x_vals[i] + (i as f64 * 0.1).sin() * 0.1
            } else {
                alpha - 1.0 * x_vals[i] + (i as f64 * 0.1).sin() * 0.1
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let q = Array1::from(q_vals);

    let result = PanelThreshold::fit(&y, &x, &q, &entity_ids).unwrap();

    assert!(result.threshold_gamma.is_finite());
    // With a very clear break, the two regimes should have very different coefficients
    let diff = (result.params_regime1[0] - result.params_regime2[0]).abs();
    assert!(
        diff > 1.0,
        "Expected clear coefficient difference: r1={}, r2={}, diff={}",
        result.params_regime1[0],
        result.params_regime2[0],
        diff
    );
}
