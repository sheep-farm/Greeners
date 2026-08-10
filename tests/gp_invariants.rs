use greeners::{GaussianProcess, GpResult};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Normal;

fn make_gp_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.3).unwrap();
    let mut x = Array2::zeros((n, 2));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = rng.gen::<f64>() * 4.0 - 2.0;
        let x2 = rng.gen::<f64>() * 4.0 - 2.0;
        x[(i, 0)] = x1;
        x[(i, 1)] = x2;
        y.push(1.0 + 2.0 * x1 - 1.5 * x2 + noise.sample(&mut rng));
    }
    (Array1::from_vec(y), x)
}

fn assert_gp_result_finite(result: &GpResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, k);
    assert_eq!(result.fitted.len(), n);
    assert_eq!(result.fitted_sd.len(), n);
    assert!(result.fitted.iter().all(|v| v.is_finite()));
    assert!(result.fitted_sd.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.length_scale.is_finite());
    assert!(result.signal_variance.is_finite());
    assert!(result.noise_variance.is_finite());
    assert!(result.mse.is_finite());
    assert!(result.r_squared.is_finite());
}

/// GP fit returns correct shapes, finite values, and a reasonable fit.
#[test]
fn test_gp_fit_finite_and_reasonable() {
    let n = 40;
    let (y, x) = make_gp_data(n, 9401);
    let result = GaussianProcess::fit(&y, &x, None).unwrap();
    assert_gp_result_finite(&result, n, 2);
    assert!(result.mse < 2.0, "mse = {}", result.mse);
    assert!(result.r_squared > 0.6, "r2 = {}", result.r_squared);
}

/// Input validation: too few observations, no features, or zero-variance y.
#[test]
fn test_gp_input_validation() {
    let y_short = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let x_2 = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
    assert!(GaussianProcess::fit(&y_short, &x_2, None).is_err());

    let y_ok = Array1::from_vec(vec![1.0; 10]);
    let x_empty = Array2::from_shape_vec((10, 0), vec![]).unwrap();
    assert!(GaussianProcess::fit(&y_ok, &x_empty, None).is_err());

    let y_const = Array1::from_vec(vec![5.0; 10]);
    let x_some = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
    assert!(GaussianProcess::fit(&y_const, &x_some, None).is_err());
}
