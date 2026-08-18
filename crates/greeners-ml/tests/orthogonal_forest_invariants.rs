use greeners_ml::orthogonal_forest::OrfResult;
use greeners_ml::orthogonal_forest::OrthogonalForest;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Normal;

fn make_orf_data(n: usize, seed: u64) -> (Array1<f64>, Vec<bool>, Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let k = 2;
    let p = 2;
    let mut x = Array2::zeros((n, k));
    let mut w = Array2::zeros((n, p));
    let mut y = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = rng.gen::<f64>() * 4.0 - 2.0;
        let x2 = rng.gen::<f64>() * 4.0 - 2.0;
        x[(i, 0)] = x1;
        x[(i, 1)] = x2;
        // Confounders are a copy of x (w = x)
        w[(i, 0)] = x1;
        w[(i, 1)] = x2;
        let treated = rng.gen_bool(0.5);
        t.push(treated);
        let tau = if treated { 2.0 } else { 0.0 };
        y.push(1.0 + 1.0 * x1 - 0.5 * x2 + tau + noise.sample(&mut rng));
    }
    (Array1::from_vec(y), t, x, w)
}

fn assert_orf_result_finite(result: &OrfResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, k);
    assert_eq!(result.cate.len(), n);
    assert!(result.cate.iter().all(|v| v.is_finite()));
    assert!(result.ate.is_finite());
    assert!(result.ate_se >= 0.0 && result.ate_se.is_finite());
    assert!(result.ate_ci[0].is_finite() && result.ate_ci[1].is_finite());
    assert!(result.ate_ci[0] <= result.ate_ci[1]);
    assert_eq!(result.feature_names.len(), k);
    assert_eq!(result.feature_importance.len(), k);
    assert!(result
        .feature_importance
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.n_trees > 0);
    assert!(result.max_depth > 0);
}

/// OrthogonalRandomForest returns finite CATE estimates and a reasonable ATE.
#[test]
fn test_orthogonal_forest_fit_finite_and_reasonable() {
    let n = 60;
    let (y, t, x, w) = make_orf_data(n, 9419);
    let result = OrthogonalForest::fit(&y, &t, &x, &w, Some(30), Some(5), None).unwrap();
    assert_orf_result_finite(&result, n, 2);
    assert!((result.ate - 2.0).abs() < 2.0, "ate = {}", result.ate);
}

/// Input validation catches mismatched dimensions and insufficient data.
#[test]
fn test_orthogonal_forest_input_validation() {
    let n = 20;
    let (y, t, x, w) = make_orf_data(n, 9420);
    assert!(OrthogonalForest::fit(&y, &t, &x, &w, None, None, None).is_err());

    let n2 = 60;
    let (y2, _t2, x2, w2) = make_orf_data(n2, 9421);
    let t2_all_false = vec![false; n2];
    assert!(OrthogonalForest::fit(&y2, &t2_all_false, &x2, &w2, None, None, None).is_err());

    let mut t_short = vec![false; n2];
    t_short.pop();
    assert!(OrthogonalForest::fit(&y2, &t_short, &x2, &w2, None, None, None).is_err());

    let w_short = Array2::zeros((n2 - 1, 2));
    assert!(OrthogonalForest::fit(&y2, &_t2, &x2, &w_short, None, None, None).is_err());
}
