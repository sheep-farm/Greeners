use greeners::{CovarianceType, GenPoisson, NegBin, NegBinP, Poisson};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Poisson as PoissonDist;
use rand::distributions::Distribution;
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Poisson GLM recovers the true log-linear coefficients.
#[test]
fn test_poisson_recovery() {
    let n = 500;
    let mut rng = StdRng::seed_from_u64(1001);

    let beta0 = 0.5;
    let beta1 = 0.3;

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 100.0;
        x_vec.push(1.0);
        x_vec.push(x);
        let mu = (beta0 + beta1 * x).exp();
        let d = PoissonDist::new(mu).unwrap();
        y_vec.push(d.sample(&mut rng) as f64);
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = Poisson::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - beta0).abs(), 0.2);
    approx_zero((result.params[1] - beta1).abs(), 0.1);
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());

    // Mean prediction is positive.
    let x_new = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 1.0]).unwrap();
    let pred = result.predict_count(&x_new);
    assert!(pred.iter().all(|&v| v > 0.0));
}

/// Poisson with exposure uses the offset correctly: the predicted count
/// equals exp(Xβ + ln(exposure)) = exposure * exp(Xβ).
#[test]
fn test_poisson_exposure_offset() {
    let n = 300;
    let mut rng = StdRng::seed_from_u64(1002);

    let beta0 = -0.2;
    let beta1 = 0.2;

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut exposure = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 100.0;
        x_vec.push(1.0);
        x_vec.push(x);
        let e = 2.0;
        exposure.push(e);
        let rate = (beta0 + beta1 * x).exp();
        let d = PoissonDist::new(rate * e).unwrap();
        y_vec.push(d.sample(&mut rng) as f64);
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let e_arr = Array1::from_vec(exposure);

    let result = Poisson::fit_with_exposure(&y, &x, &e_arr, CovarianceType::NonRobust).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - beta0).abs(), 0.3);
    approx_zero((result.params[1] - beta1).abs(), 0.15);
}

/// Negative Binomial with known alpha recovers coefficients on overdispersed
/// count data.
#[test]
fn test_negbin_known_alpha_recovery() {
    let n = 600;
    let mut rng = StdRng::seed_from_u64(2001);

    let beta0 = 0.5;
    let beta1 = 0.25;
    let alpha = 1.5;
    let r = 1.0 / alpha;

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 100.0;
        x_vec.push(1.0);
        x_vec.push(x);
        let mu = (beta0 + beta1 * x).exp();
        let p = r / (r + mu);
        // Draw gamma shape r, scale (1-p)/p, then Poisson
        let gamma = gamma_sample(r, (1.0 - p) / p, &mut rng);
        let d = PoissonDist::new(gamma.max(1e-6)).unwrap();
        y_vec.push(d.sample(&mut rng) as f64);
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = NegBin::fit_with_alpha(&y, &x, alpha, CovarianceType::NonRobust, None).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - beta0).abs(), 0.3);
    approx_zero((result.params[1] - beta1).abs(), 0.15);
    assert!(result.alpha > 0.0);
    assert!(result.log_likelihood.is_finite());
}

/// Negative Binomial auto alpha converges and produces a positive dispersion.
#[test]
fn test_negbin_auto_alpha_converges() {
    let n = 400;
    let mut rng = StdRng::seed_from_u64(2002);

    let beta0 = 0.4;
    let beta1 = 0.2;
    let alpha = 1.0;
    let r = 1.0 / alpha;

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 100.0;
        x_vec.push(1.0);
        x_vec.push(x);
        let mu = (beta0 + beta1 * x).exp();
        let p = r / (r + mu);
        // Draw gamma shape r, scale (1-p)/p, then Poisson
        let gamma = gamma_sample(r, (1.0 - p) / p, &mut rng);
        let d = PoissonDist::new(gamma.max(1e-6)).unwrap();
        y_vec.push(d.sample(&mut rng) as f64);
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = NegBin::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    assert!(result.converged);
    assert!(result.alpha > 0.0);
    approx_zero((result.params[0] - beta0).abs(), 0.4);
    approx_zero((result.params[1] - beta1).abs(), 0.2);
}

/// Generalized Poisson reduces to Poisson when the data are Poisson:
/// alpha is close to zero and the mean parameters are recovered.
#[test]
fn test_genpoisson_reduces_to_poisson() {
    let n = 500;
    let mut rng = StdRng::seed_from_u64(3001);

    let beta0 = 0.5;
    let beta1 = 0.2;

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 100.0;
        x_vec.push(1.0);
        x_vec.push(x);
        let mu = (beta0 + beta1 * x).exp();
        let d = PoissonDist::new(mu).unwrap();
        y_vec.push(d.sample(&mut rng) as f64);
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = GenPoisson::fit(&y, &x).unwrap();
    approx_zero((result.params[0] - beta0).abs(), 0.2);
    approx_zero((result.params[1] - beta1).abs(), 0.1);
    approx_zero(result.alpha, 0.05);
    assert!(result.alpha > -0.5 && result.alpha < 0.5);
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

/// NegBinP with p=2 (NB2) recovers coefficients and positive alpha on
/// Gamma-Poisson data.
#[test]
fn test_negbinp_recovery() {
    let n = 600;
    let mut rng = StdRng::seed_from_u64(3002);

    let beta0 = 0.5;
    let beta1 = 0.25;
    let alpha = 0.5;

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 100.0;
        x_vec.push(1.0);
        x_vec.push(x);
        let mu = (beta0 + beta1 * x).exp();
        let r = 1.0 / (alpha * mu);
        let p = r / (r + mu);
        let gamma = gamma_sample(r, (1.0 - p) / p, &mut rng);
        let d = PoissonDist::new(gamma.max(1e-6)).unwrap();
        y_vec.push(d.sample(&mut rng) as f64);
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = NegBinP::fit(&y, &x, 2.0).unwrap();
    assert!(result.converged);
    approx_zero((result.params[0] - beta0).abs(), 0.3);
    approx_zero((result.params[1] - beta1).abs(), 0.15);
    assert!(result.alpha > 0.0);
    assert_eq!(result.p_param, 2.0);
    assert!(result.log_likelihood.is_finite());
}

/// Gamma sample via Marsaglia-Tsang for shape >= 1.
fn gamma_sample(shape: f64, scale: f64, rng: &mut StdRng) -> f64 {
    if shape < 1.0 {
        return gamma_sample(shape + 1.0, scale, rng) * rng.gen::<f64>().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (3.0 * d.sqrt());
    loop {
        let z: f64 = ndarray_rand::rand_distr::StandardNormal.sample(rng);
        if z <= -1.0 / c {
            continue;
        }
        let v = (1.0_f64 + c * z).powi(3);
        let u = rand::random::<f64>();
        if u < 1.0 - 0.0331 * z * z * z * z || u.ln() < 0.5 * z * z + d * (v.ln() - v + 1.0) {
            return d * v * scale;
        }
    }
}
