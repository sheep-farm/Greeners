use greeners::{XGBoost, XgboostResult};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_xgb_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut x = Array2::zeros((n, 2));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = rng.gen::<f64>() * 4.0 - 2.0;
        let x2 = rng.gen::<f64>() * 4.0 - 2.0;
        x[(i, 0)] = x1;
        x[(i, 1)] = x2;
        y.push(1.0 + 2.0 * x1 + 3.0 * x2 + noise.sample(&mut rng));
    }
    (Array1::from_vec(y), x)
}

fn assert_xgb_result_finite(result: &XgboostResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, k);
    assert_eq!(result.fitted.len(), n);
    assert!(result.fitted.iter().all(|v| v.is_finite()));
    assert_eq!(result.feature_importance.len(), k);
    assert!(result
        .feature_importance
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.mse.is_finite());
    assert!(result.r_squared.is_finite());
    assert!(result.n_trees > 0);
    assert!(result.learning_rate > 0.0);
    assert!(result.lambda >= 0.0);
    assert!(result.alpha >= 0.0);
    assert!(result.gamma >= 0.0);
}

/// XGBoost returns correct shapes and a reasonable fit.
#[test]
fn test_xgboost_fit_finite_and_reasonable() {
    let n = 50;
    let (y, x) = make_xgb_data(n, 9408);
    let result = XGBoost::fit(
        &y,
        &x,
        50,
        Some(0.1),
        Some(3),
        Some(1.0),
        Some(0.0),
        Some(0.0),
        Some(1.0),
        Some(1.0),
        None,
    )
    .unwrap();
    assert_xgb_result_finite(&result, n, 2);
    assert!(result.mse < 2.0, "mse = {}", result.mse);
    assert!(result.r_squared > 0.5, "r2 = {}", result.r_squared);
}

/// Input validation catches invalid dimensions and zero trees.
#[test]
fn test_xgboost_input_validation() {
    let y = Array1::from_vec(vec![1.0; 4]);
    let x = Array2::from_shape_vec((4, 1), vec![1.0; 4]).unwrap();
    assert!(XGBoost::fit(&y, &x, 10, None, None, None, None, None, None, None, None).is_err());

    let y_ok = Array1::from_vec(vec![1.0; 10]);
    let x_empty = Array2::from_shape_vec((10, 0), vec![]).unwrap();
    assert!(
        XGBoost::fit(&y_ok, &x_empty, 10, None, None, None, None, None, None, None, None).is_err()
    );

    let (y2, x2) = make_xgb_data(10, 9409);
    assert!(XGBoost::fit(&y2, &x2, 0, None, None, None, None, None, None, None, None).is_err());
}
