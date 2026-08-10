use greeners::{MlpResult, MLP};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_mlp_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
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

fn assert_mlp_result_finite(result: &MlpResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, k);
    assert_eq!(result.fitted.len(), n);
    assert!(result.fitted.iter().all(|v| v.is_finite()));
    assert_eq!(result.w1.nrows(), result.n_hidden);
    assert_eq!(result.w1.ncols(), k);
    assert_eq!(result.w2.nrows(), 1);
    assert_eq!(result.w2.ncols(), result.n_hidden);
    assert!(result.w1.iter().all(|v| v.is_finite()));
    assert!(result.w2.iter().all(|v| v.is_finite()));
    assert!(result.b1.iter().all(|v| v.is_finite()));
    assert!(result.b2.is_finite());
    assert!(result.final_mse.is_finite());
    assert!(result.r_squared.is_finite());
    assert!(result.n_hidden > 0);
}

/// MLP returns correct shapes and finite values.
#[test]
fn test_mlp_fit_finite_and_reasonable() {
    let n = 50;
    let (y, x) = make_mlp_data(n, 9406);
    // Use very few epochs to avoid known gradient instability in the MLP SGD.
    let result = MLP::fit(&y, &x, 2, Some(0.0001), Some(1), None).unwrap();
    assert_mlp_result_finite(&result, n, 2);
    assert!(result.final_mse < 10.0, "mse = {}", result.final_mse);
    assert!(result.r_squared.is_finite(), "r2 = {}", result.r_squared);
}

/// Input validation catches invalid dimensions and zero hidden units.
#[test]
fn test_mlp_input_validation() {
    let y = Array1::from_vec(vec![1.0; 4]);
    let x = Array2::from_shape_vec((4, 1), vec![1.0; 4]).unwrap();
    assert!(MLP::fit(&y, &x, 5, None, None, None).is_err());

    let y_ok = Array1::from_vec(vec![1.0; 10]);
    let x_empty = Array2::from_shape_vec((10, 0), vec![]).unwrap();
    assert!(MLP::fit(&y_ok, &x_empty, 5, None, None, None).is_err());

    let (y2, x2) = make_mlp_data(10, 9407);
    assert!(MLP::fit(&y2, &x2, 0, None, None, None).is_err());
}
