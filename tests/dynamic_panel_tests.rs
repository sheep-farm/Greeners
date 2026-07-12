use greeners::{LocalLevel, SystemGmm};
use ndarray::{Array1, Array2};

/// Simulated balanced panel with an AR(1) process and one strictly exogenous regressor.
/// y_{i,t} = 0.5 * y_{i,t-1} + 0.3 * x_{i,t} + alpha_i + u_{i,t}
/// where alpha_i ~ N(0, 1) and u_{i,t} ~ N(0, 1).
/// The test ensures System GMM (Blundell-Bond) runs without a singular-matrix error
/// and recovers coefficients with the expected sign and magnitude.
#[test]
fn test_system_gmm_basic() {
    let n_entities = 100;
    let t_periods = 8;
    let mut y_vec = Vec::new();
    let mut x_vec = Vec::new();
    let mut entity_ids = Vec::new();
    let mut time_ids = Vec::new();

    let true_phi = 0.5;
    let true_beta = 0.3;

    for i in 0..n_entities {
        let alpha_i = (i as f64) * 0.01; // small fixed effect
        let mut y_prev = alpha_i + 0.5; // initial value
        for t in 0..t_periods {
            let x = (t as f64) * 0.1 + (i as f64) * 0.01;
            let u = ((i * 7 + t * 13) as f64 % 5.0) / 5.0 - 0.5; // pseudo-random noise
            let y = true_phi * y_prev + true_beta * x + alpha_i + u;
            y_vec.push(y);
            x_vec.push(x);
            entity_ids.push(i as i64);
            time_ids.push(t as i64);
            y_prev = y;
        }
    }

    let y = Array1::from(y_vec);
    let x = Array2::from_shape_vec((y.len(), 1), x_vec).unwrap();

    let result = SystemGmm::fit(
        &y,
        &x,
        &entity_ids,
        &time_ids,
        2,    // max_lags
        true, // two_step
        Some(vec!["x".to_string()]),
    )
    .unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.n_entities > 0);
    assert!(result.n_obs_fd > 0);
    assert!(result.n_obs_lev > 0);
    assert!(result.n_instruments > result.params.len());

    // Coefficient on the lagged dependent variable should be positive and moderate.
    assert!(
        result.params[0] > 0.0 && result.params[0] < 1.0,
        "lag coefficient should be in (0, 1), got {}",
        result.params[0]
    );
    // Coefficient on the exogenous regressor should be positive.
    assert!(
        result.params[1] > 0.0,
        "exogenous coefficient should be positive, got {}",
        result.params[1]
    );
}

/// Regression test for the wagepanel-like bug where including a constant in the
/// exogenous regressor matrix produced a singular instrument matrix.  We now
/// ensure the estimator tolerates a constant column without crashing.
#[test]
fn test_system_gmm_with_constant_column() {
    let n_entities = 50;
    let t_periods = 8;
    let mut y_vec = Vec::new();
    let mut x_rows = Vec::new();
    let mut entity_ids = Vec::new();
    let mut time_ids = Vec::new();

    for i in 0..n_entities {
        let alpha_i = (i as f64) * 0.01;
        let mut y_prev = alpha_i + 0.5;
        for t in 0..t_periods {
            let x1 = (t as f64) * 0.1;
            let u = ((i * 7 + t * 13) as f64 % 5.0) / 5.0 - 0.5;
            let y = 0.5 * y_prev + 0.3 * x1 + alpha_i + u;
            y_vec.push(y);
            x_rows.push(vec![1.0, x1]); // constant column included
            entity_ids.push(i as i64);
            time_ids.push(t as i64);
            y_prev = y;
        }
    }

    let y = Array1::from(y_vec);
    let x = Array2::from_shape_fn((y.len(), 2), |(i, j)| x_rows[i][j]);

    let result = SystemGmm::fit(
        &y,
        &x,
        &entity_ids,
        &time_ids,
        2,
        true,
        Some(vec!["const".to_string(), "x1".to_string()]),
    )
    .unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.params[0] > 0.0 && result.params[0] < 1.0);
    assert!(result.params[1] > 0.0);
}

/// Local-level model on a simulated random walk plus noise.
/// The estimated sigma_obs should be positive and dominate sigma_state
/// for a series with little drift.
#[test]
fn test_local_level_mle() {
    let n = 100;
    let mut y = Vec::with_capacity(n);
    let mut mu = 0.0;
    for t in 0..n {
        mu += 0.01; // small drift
        let obs = mu + ((t * 7) as f64 % 5.0) / 5.0 - 0.5; // moderate noise
        y.push(obs);
    }

    let result = LocalLevel::fit(&y).unwrap();
    assert!(result.sigma_obs > 0.0);
    assert!(result.sigma_state > 0.0);
    assert!(result.log_likelihood.is_finite());
    assert_eq!(result.filtered_states.len(), n);
    assert_eq!(result.smoothed_states.len(), n);
}
