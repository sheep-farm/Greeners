use greeners_ml::random_forest::RandomForest;
use greeners_ml::random_forest::RandomForestResult;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Normal;

fn make_rf_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
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

fn assert_rf_result_finite(result: &RandomForestResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, k);
    assert_eq!(result.fitted.len(), n);
    assert_eq!(result.oob_predictions.len(), n);
    assert!(result.fitted.iter().all(|v| v.is_finite()));
    assert!(result.oob_predictions.iter().all(|v| v.is_finite()));
    assert_eq!(result.feature_importance.len(), k);
    assert!(result
        .feature_importance
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.mse.is_finite());
    assert!(result.r_squared.is_finite());
    assert!(result.oob_r_squared.is_finite());
    assert!(result.n_trees > 0);
}

/// Random forest returns correct shapes and a reasonable fit.
#[test]
fn test_random_forest_fit_finite_and_reasonable() {
    let n = 50;
    let (y, x) = make_rf_data(n, 9404);
    let result = RandomForest::fit(&y, &x, 50, 5, None).unwrap();
    assert_rf_result_finite(&result, n, 2);
    assert!(result.mse < 2.0, "mse = {}", result.mse);
    assert!(result.r_squared > 0.5, "r2 = {}", result.r_squared);
}

/// Input validation catches invalid dimensions and zero trees.
#[test]
fn test_random_forest_input_validation() {
    let y = Array1::from_vec(vec![1.0; 4]);
    let x = Array2::from_shape_vec((4, 1), vec![1.0; 4]).unwrap();
    assert!(RandomForest::fit(&y, &x, 10, 3, None).is_err());

    let y_ok = Array1::from_vec(vec![1.0; 10]);
    let x_empty = Array2::from_shape_vec((10, 0), vec![]).unwrap();
    assert!(RandomForest::fit(&y_ok, &x_empty, 10, 3, None).is_err());

    let (y2, x2) = make_rf_data(10, 9405);
    assert!(RandomForest::fit(&y2, &x2, 0, 3, None).is_err());
}
