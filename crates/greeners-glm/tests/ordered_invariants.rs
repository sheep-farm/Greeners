use greeners_glm::ordered::OrderedLogit;
use greeners_glm::ordered::OrderedProbit;
use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, Uniform};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

fn logistic_sample(rng: &mut StdRng) -> f64 {
    let u: f64 = rng.gen();
    (u / (1.0 - u)).ln()
}

/// Ordered Logit recovers the slope and the thresholds are monotone.
#[test]
fn test_ordered_logit_recovery() {
    let n = 600;
    let mut rng = StdRng::seed_from_u64(9401);
    let unif = Uniform::new(-2.0_f64, 2.0_f64);

    let beta = 1.5;
    let thresholds = [-1.0, 1.0];

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for _ in 0..n {
        let x = unif.sample(&mut rng);
        let z = beta * x + logistic_sample(&mut rng);
        let y = if z < thresholds[0] {
            0.0
        } else if z < thresholds[1] {
            1.0
        } else {
            2.0
        };
        x_vec.push(x);
        y_vec.push(y);
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = OrderedLogit::fit(&y, &x).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - beta).abs(), 0.2);
    assert_eq!(result.thresholds.len(), 2);
    assert!(result.thresholds[0] < result.thresholds[1]);
    assert!(result.log_likelihood.is_finite());
    assert!(result.n_obs == n);
    assert!(result.n_categories == 3);
}

/// Ordered Probit recovers the slope and the thresholds are monotone.
#[test]
fn test_ordered_probit_recovery() {
    let n = 600;
    let mut rng = StdRng::seed_from_u64(9402);
    let unif = Uniform::new(-2.0_f64, 2.0_f64);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let beta = 1.5;
    let thresholds = [-1.0, 1.0];

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for _ in 0..n {
        let x = unif.sample(&mut rng);
        let z = beta * x + noise.sample(&mut rng);
        let y = if z < thresholds[0] {
            0.0
        } else if z < thresholds[1] {
            1.0
        } else {
            2.0
        };
        x_vec.push(x);
        y_vec.push(y);
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = OrderedProbit::fit(&y, &x).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - beta).abs(), 0.2);
    assert_eq!(result.thresholds.len(), 2);
    assert!(result.thresholds[0] < result.thresholds[1]);
    assert!(result.log_likelihood.is_finite());
    assert!(result.n_obs == n);
    assert!(result.n_categories == 3);
}

/// Input validation.
#[test]
fn test_ordered_input_validation() {
    let y = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0]);
    let x = Array2::from_shape_vec((4, 1), vec![1.0; 4]).unwrap();
    assert!(OrderedLogit::fit(&y, &x).is_err()); // < 3 categories

    let y_nan = Array1::from_vec(vec![0.0, 1.0, f64::NAN]);
    assert!(OrderedLogit::fit(&y_nan, &x).is_err());
}
