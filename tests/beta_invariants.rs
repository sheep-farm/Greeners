use greeners::{BetaLink, BetaModel};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Beta;
use statrs::distribution::ContinuousCDF;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Beta regression with a logit link recovers the true mean parameters.
#[test]
fn test_beta_regression_logit_recovery() {
    let n = 500;
    let mut rng = StdRng::seed_from_u64(9501);

    let beta0 = 0.5;
    let beta1 = 1.0;
    let phi = 5.0;

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 / 100.0) - 2.5;
        x_vec.push(1.0);
        x_vec.push(x);
        let eta = beta0 + beta1 * x;
        let mu = 1.0 / (1.0 + (-eta).exp());
        let a = mu * phi;
        let b = (1.0 - mu) * phi;
        let d = Beta::new(a, b).unwrap();
        y_vec.push(d.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = BetaModel::fit(&y, &x, &BetaLink::Logit).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - beta0).abs(), 0.2);
    approx_zero((result.params[1] - beta1).abs(), 0.15);
    assert!(result.precision_param > 0.0);
    assert!(result.log_likelihood.is_finite());
    assert!(result.pseudo_r2.is_finite());
    assert!(result.n_obs == n);
}

/// Beta regression with a probit link recovers the slope sign.
#[test]
fn test_beta_regression_probit_recovery() {
    let n = 500;
    let mut rng = StdRng::seed_from_u64(9502);

    let beta0 = 0.0;
    let beta1 = 1.0;
    let phi = 4.0;

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.gen::<f64>() * 2.0 - 1.0;
        x_vec.push(1.0);
        x_vec.push(x);
        let eta = beta0 + beta1 * x;
        let mu = statrs::distribution::Normal::new(0.0, 1.0)
            .unwrap()
            .cdf(eta);
        let a = mu * phi;
        let b = (1.0 - mu) * phi;
        let d = Beta::new(a, b).unwrap();
        y_vec.push(d.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = BetaModel::fit(&y, &x, &BetaLink::Probit).unwrap();
    assert!(result.converged);
    assert!((result.params[1] - beta1).abs() < 0.2);
    assert!(result.precision_param > 0.0);
    assert!(result.log_likelihood.is_finite());
}

/// Input validation.
#[test]
fn test_beta_input_validation() {
    let y = Array1::from_vec(vec![0.0, 0.5, 1.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0; 3]).unwrap();
    assert!(BetaModel::fit(&y, &x, &BetaLink::Logit).is_err());

    let y2 = Array1::from_vec(vec![0.5, 0.6, 0.7]);
    let x2 = Array2::from_shape_vec((2, 1), vec![1.0; 2]).unwrap();
    assert!(BetaModel::fit(&y2, &x2, &BetaLink::Logit).is_err());
}
