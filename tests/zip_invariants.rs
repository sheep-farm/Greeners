use greeners::{ZINB, ZIP};
use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, Uniform};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Poisson;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// ZIP recovers the count and inflation parameters approximately.
#[test]
fn test_zip_recovery() {
    let n = 500;
    let mut rng = StdRng::seed_from_u64(333);
    let unif = Uniform::new(0.0_f64, 1.0_f64);

    let beta0 = 0.5;
    let beta1 = 0.3;
    let gamma0: f64 = -1.0; // logit(0.27) ~ -1
    let p: f64 = 1.0 / (1.0 + (-gamma0).exp());

    let mut x_count_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut x_infl_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 100.0;
        x_count_vec.push(1.0);
        x_count_vec.push(x);
        x_infl_vec.push(1.0);

        let mu = (beta0 + beta1 * x).exp();
        let y = if unif.sample(&mut rng) < p {
            0.0
        } else {
            let dist = Poisson::new(mu).unwrap();
            dist.sample(&mut rng) as f64
        };
        y_vec.push(y);
    }
    let x_count = Array2::from_shape_vec((n, 2), x_count_vec).unwrap();
    let x_inflate = Array2::from_shape_vec((n, 1), x_infl_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = ZIP::fit(&y, &x_count, Some(&x_inflate)).unwrap();
    assert!(result.converged);
    assert!(result.alpha.is_none());

    // Count parameters.
    approx_zero((result.count_params[0] - beta0).abs(), 0.3);
    approx_zero((result.count_params[1] - beta1).abs(), 0.2);

    // Inflate intercept.
    approx_zero((result.inflate_params[0] - gamma0).abs(), 0.3);

    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

/// ZINB converges and produces a positive dispersion on data generated from
/// a ZIP process.
#[test]
fn test_zinb_convergence() {
    let n = 400;
    let mut rng = StdRng::seed_from_u64(444);
    let unif = Uniform::new(0.0_f64, 1.0_f64);

    let beta0 = 0.5;
    let beta1 = 0.2;
    let gamma0: f64 = -0.8;
    let p: f64 = 1.0 / (1.0 + (-gamma0).exp());

    let mut x_count_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut x_infl_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 80.0;
        x_count_vec.push(1.0);
        x_count_vec.push(x);
        x_infl_vec.push(1.0);

        let mu = (beta0 + beta1 * x).exp();
        let y = if unif.sample(&mut rng) < p {
            0.0
        } else {
            let dist = Poisson::new(mu).unwrap();
            dist.sample(&mut rng) as f64
        };
        y_vec.push(y);
    }
    let x_count = Array2::from_shape_vec((n, 2), x_count_vec).unwrap();
    let x_inflate = Array2::from_shape_vec((n, 1), x_infl_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = ZINB::fit(&y, &x_count, Some(&x_inflate)).unwrap();
    assert!(result.converged);
    assert!(result.alpha.unwrap() > 0.0);
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}
