use greeners::{GrfResult, GRF};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_grf_data(n: usize, seed: u64) -> (Array1<f64>, Vec<bool>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut x = Array2::zeros((n, 2));
    let mut y = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = rng.gen::<f64>() * 4.0 - 2.0;
        let x2 = rng.gen::<f64>() * 4.0 - 2.0;
        x[(i, 0)] = x1;
        x[(i, 1)] = x2;
        let treated = rng.gen_bool(0.5);
        t.push(treated);
        let tau = if treated { 2.0 } else { 0.0 };
        y.push(1.0 + 1.0 * x1 - 0.5 * x2 + tau + noise.sample(&mut rng));
    }
    (Array1::from_vec(y), t, x)
}

fn assert_grf_result_finite(result: &GrfResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, k);
    assert_eq!(result.cate.len(), n);
    assert!(result.cate.iter().all(|v| v.is_finite()));
    assert!(result.ate.is_finite());
    assert!(result.ate_se >= 0.0 && result.ate_se.is_finite());
    assert!(result.ate_ci[0].is_finite() && result.ate_ci[1].is_finite());
    assert!(result.ate_ci[0] <= result.ate_ci[1]);
    assert_eq!(result.propensity.len(), n);
    assert!(result
        .propensity
        .iter()
        .all(|v| v.is_finite() && *v >= 0.01 && *v <= 0.99));
    assert_eq!(result.outcome_reg.len(), n);
    assert!(result.outcome_reg.iter().all(|v| v.is_finite()));
    assert_eq!(result.feature_importance.len(), k);
    assert!(result
        .feature_importance
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.heterogeneity >= 0.0 && result.heterogeneity.is_finite());
    assert!(result.n_trees > 0);
}

/// GRF returns finite CATE estimates and a reasonable ATE.
#[test]
fn test_grf_fit_finite_and_reasonable() {
    let n = 60;
    let (y, t, x) = make_grf_data(n, 9416);
    let result = GRF::fit(&y, &t, &x, Some(50), Some(5), None).unwrap();
    assert_grf_result_finite(&result, n, 2);
    assert!((result.ate - 2.0).abs() < 1.0, "ate = {}", result.ate);
}

/// Input validation catches mismatched dimensions and insufficient support.
#[test]
fn test_grf_input_validation() {
    let n = 15;
    let (y, t, x) = make_grf_data(n, 9417);
    assert!(GRF::fit(&y, &t, &x, None, None, None).is_err());

    let n2 = 60;
    let (y2, _t2, x2) = make_grf_data(n2, 9418);
    let t2_all_false = vec![false; n2];
    assert!(GRF::fit(&y2, &t2_all_false, &x2, None, None, None).is_err());

    let mut t_short = vec![false; n2];
    t_short.pop();
    assert!(GRF::fit(&y2, &t_short, &x2, None, None, None).is_err());
}
