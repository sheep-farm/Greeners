use greeners::{predict_exp, predict_power, GreenersError, NLS};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// NLS converges for a constant model from an exact start.
#[test]
fn test_nls_constant_exact() {
    let n = 20;
    let y = Array1::from_vec(vec![5.0; n]);
    let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64).collect()).unwrap();
    let predict = |params: &[f64], _x: &[f64]| params[0];
    let result = NLS::fit(&y, &x, &predict, &[5.0], 100, 1e-8).unwrap();
    assert!(result.converged);
    assert_eq!(result.n_iter, 1);
    approx_zero((result.params[0] - 5.0).abs(), 1e-6);
    assert!(result.rss < 1e-10);
}

/// NLS recovers a slope parameter from a simple proportional model.
#[test]
fn test_nls_slope_recovery() {
    let n = 50;
    let mut rng = StdRng::seed_from_u64(4001);
    let noise = Normal::new(0.0, 0.2).unwrap();

    let true_b = 2.5;

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i + 1) as f64 / 10.0;
        x_vec.push(x);
        y_vec.push(true_b * x + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let predict = |params: &[f64], x_row: &[f64]| params[0] * x_row[0];

    let result = NLS::fit(&y, &x, &predict, &[2.0], 100, 1e-6).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - true_b).abs(), 0.05);
    assert!(result.rss >= 0.0);
}

/// NLS recovers the parameters of an exponential growth model.
#[test]
fn test_nls_exponential_recovery() {
    let n = 80;
    let mut rng = StdRng::seed_from_u64(4002);
    let noise = Normal::new(0.0, 0.2).unwrap();

    let true_a = 2.0;
    let true_b = 0.1;

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i + 1) as f64 / 10.0;
        x_vec.push(x);
        y_vec.push((true_a * (true_b * x).exp()) + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = NLS::fit(&y, &x, &predict_exp, &[1.5, 0.05], 300, 1e-5).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - true_a).abs(), 0.2);
    approx_zero((result.params[1] - true_b).abs(), 0.05);
    assert!(result.rss >= 0.0);
    assert!(result.n_iter <= 300);
}

/// NLS recovers a power-law model.
#[test]
fn test_nls_power_recovery() {
    let n = 80;
    let mut rng = StdRng::seed_from_u64(4003);
    let noise = Normal::new(0.0, 0.2).unwrap();

    let true_a = 2.0;
    let true_b = 0.5;

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i + 1) as f64 / 10.0;
        x_vec.push(x);
        y_vec.push((true_a * x.powf(true_b)) + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = NLS::fit(&y, &x, &predict_power, &[1.5, 0.4], 300, 1e-5).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - true_a).abs(), 0.2);
    approx_zero((result.params[1] - true_b).abs(), 0.05);
    assert!(result.rss >= 0.0);
}

/// Input validation.
#[test]
fn test_nls_input_validation() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
    let predict = |_p: &[f64], _x: &[f64]| 1.0;
    let result = NLS::fit(&y, &x, &predict, &[1.0], 10, 1e-6);
    assert!(matches!(result, Err(GreenersError::ShapeMismatch(_))));
}
