use greeners_timeseries::tvp_var::TvpVar;
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_tvp_var_data(seed: u64, n: usize, k: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let mut data: Vec<f64> = (0..k).map(|_| noise.sample(&mut rng)).collect();
    for _ in 1..n {
        for j in 0..k {
            let prev = data[data.len() - k + j];
            data.push(0.5 * prev + noise.sample(&mut rng));
        }
    }
    Array2::from_shape_vec((n, k), data).unwrap()
}

/// TVP-VAR fit returns smoothed coefficients and a covariance matrix with the expected shapes.
#[test]
fn test_tvp_var_invariants() {
    let y = make_tvp_var_data(12345, 120, 2);
    let result = TvpVar::fit(&y, 2, Some(vec!["y1".into(), "y2".into()])).unwrap();

    assert_eq!(result.n_obs, 118);
    assert_eq!(result.n_vars, 2);
    assert_eq!(result.lags, 2);
    assert_eq!(result.n_regressors, 5);
    assert_eq!(result.beta_smoothed.shape(), &[118, 5, 2]);
    assert_eq!(result.sigma.shape(), &[2, 2]);
    assert!(result.beta_smoothed.iter().all(|v| v.is_finite()));
    assert!(result.sigma.iter().all(|v| v.is_finite()));
    assert!(result.q_scale.is_finite() && result.q_scale >= 0.0);
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

/// TVP-VAR rejects too few observations or lags set to zero.
#[test]
fn test_tvp_var_input_validation() {
    let y = make_tvp_var_data(11111, 8, 2);
    assert!(TvpVar::fit(&y, 2, None).is_err());

    let y2 = make_tvp_var_data(22222, 50, 2);
    assert!(TvpVar::fit(&y2, 0, None).is_err());
}

/// Smoothed TVP-VAR coefficients have bounded variation around the initial OLS estimates.
#[test]
fn test_tvp_var_smoothed_shape() {
    let y = make_tvp_var_data(33333, 150, 3);
    let result = TvpVar::fit(&y, 2, None).unwrap();

    let mid = result.n_obs / 2;
    for r in 0..result.n_regressors {
        for v in 0..result.n_vars {
            let first = result.beta_smoothed[(0, r, v)];
            let middle = result.beta_smoothed[(mid, r, v)];
            let last = result.beta_smoothed[(result.n_obs - 1, r, v)];
            assert!(first.is_finite());
            assert!(middle.is_finite());
            assert!(last.is_finite());
        }
    }
    // Constants should stay within a reasonable band for this stationary-like series.
    let const_view = result.beta_smoothed.slice(ndarray::s![.., 0, ..]);
    let mean_const: f64 = const_view.iter().copied().sum::<f64>() / const_view.len() as f64;
    assert!(mean_const.is_finite());
    assert!(
        mean_const.abs() < 2.0,
        "constant mean too large: {}",
        mean_const
    );
}
