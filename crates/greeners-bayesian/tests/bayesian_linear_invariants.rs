use greeners_bayesian::bayesian_linear::BayesianLinear;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let x_vec: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let noise: f64 = StandardNormal.sample(&mut rng);
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 0.5 * noise;
    }
    (y, x)
}

/// BayesianLinear recovers the true coefficients (up to sampling noise).
#[test]
fn test_bayesian_linear_recovery() {
    let (y, x) = make_data(200, 7001);
    let r = BayesianLinear::fit(&y, &x, None).unwrap();
    assert_eq!(r.beta.len(), 2);
    assert!(r.beta.iter().all(|v| v.is_finite()));
    assert!((r.beta[0] - 1.0).abs() < 0.1);
    assert!((r.beta[1] - 2.0).abs() < 0.1);
    assert!(r.sigma2 > 0.0);
    assert_eq!(r.beta_ci.shape(), &[2, 2]);
    assert!(r.fitted.len() == y.len());
}

/// Fitted values are finite and have the same length as y.
#[test]
fn test_bayesian_linear_fitted_length() {
    let (y, x) = make_data(80, 7002);
    let r = BayesianLinear::fit(&y, &x, None).unwrap();
    assert_eq!(r.fitted.len(), y.len());
    assert!(r.fitted.iter().all(|v| v.is_finite()));
    assert!(r.r_squared >= 0.0 && r.r_squared <= 1.0);
}

/// Input validation rejects mismatched shapes and too few obs.
#[test]
fn test_bayesian_linear_input_validation() {
    let (y, x) = make_data(10, 7003);
    assert!(BayesianLinear::fit(&y, &x, None).is_ok());

    let y_short = Array1::from_vec(vec![1.0; 5]);
    assert!(BayesianLinear::fit(&y_short, &x, None).is_err());

    let x_tall = Array2::from_shape_vec((5, 4), vec![0.0; 20]).unwrap();
    assert!(BayesianLinear::fit(&y_short, &x_tall, None).is_err());
}
