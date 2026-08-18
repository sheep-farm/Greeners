use greeners_ols::gls::FGLS;
use greeners_ols::glsar::GLSAR;
use greeners_ols::ols::OLS;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// WLS with unit weights is numerically equivalent to OLS.
#[test]
fn test_wls_unit_weights_equals_ols() {
    let n = 30;
    let x = Array2::from_shape_vec(
        (n, 2),
        (0..n).flat_map(|i| vec![1.0, i as f64 / 10.0]).collect(),
    )
    .unwrap();
    let y = x.column(0).to_owned() * 1.0 + x.column(1).to_owned() * 2.0;
    let weights = Array1::from_vec(vec![1.0; n]);

    let wls = FGLS::wls(&y, &x, &weights).unwrap();
    let ols = OLS::fit(&y, &x, greeners_core::types::CovarianceType::NonRobust).unwrap();

    for j in 0..ols.params.len() {
        approx_zero((wls.params[j] - ols.params[j]).abs(), 1e-6);
    }
}

/// WLS with known inverse-variance weights recovers true coefficients on
/// heteroskedastic data.
#[test]
fn test_wls_known_weights_recovery() {
    let n = 100;
    let mut rng = StdRng::seed_from_u64(321);
    let norm = Normal::new(0.0, 1.0).unwrap();

    // Generate regressors and heteroskedastic errors with known weight.
    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut w_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = 1.0;
        let x2 = i as f64 / 50.0;
        x_vec.push(x1);
        x_vec.push(x2);
        let w = 0.5 + 0.05 * (i as f64);
        w_vec.push(w);
        let err = norm.sample(&mut rng) / w.sqrt();
        y_vec.push(1.0 + 2.0 * x2 + err);
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let weights = Array1::from_vec(w_vec);

    let wls = FGLS::wls(&y, &x, &weights).unwrap();
    approx_zero((wls.params[0] - 1.0).abs(), 0.2);
    approx_zero((wls.params[1] - 2.0).abs(), 0.15);
}

/// WLS rejects mismatched weights and non-finite data.
#[test]
fn test_wls_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0; 3]).unwrap();
    let w_bad_len = Array1::from(vec![1.0, 2.0]);
    assert!(FGLS::wls(&y, &x, &w_bad_len).is_err());

    let w_nan = Array1::from(vec![1.0, f64::NAN, 1.0]);
    assert!(FGLS::wls(&y, &x, &w_nan).is_err());
}

/// GLSAR(1) on AR(1) errors: rho is close to the true autoregressive
/// parameter and the slope close to the true beta.
#[test]
fn test_glsar_ar1_recovery() {
    let n = 120;
    let true_rho = 0.6;
    let true_beta = 2.0;

    let mut rng = StdRng::seed_from_u64(555);
    let norm = Normal::new(0.0, 1.0).unwrap();

    let mut u = 0.0;
    let mut y_vec = Vec::with_capacity(n);
    let mut x_vec = Vec::with_capacity(n);
    for i in 0..n {
        u = true_rho * u + norm.sample(&mut rng);
        let x2 = i as f64 / 20.0;
        x_vec.push(x2);
        y_vec.push(true_beta * x2 + u);
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = GLSAR::fit(&y, &x, 1, 50).unwrap();
    approx_zero((result.rho[0] - true_rho).abs(), 0.2);
    approx_zero((result.params[0] - true_beta).abs(), 0.15);
    assert!(result.converged);
}

/// GLSAR rejects invalid AR order or insufficient observations.
#[test]
fn test_glsar_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    assert!(GLSAR::fit(&y, &x, 0, 10).is_err());
    assert!(GLSAR::fit(&y, &x, 3, 10).is_err()); // n <= ar_order + k
}
