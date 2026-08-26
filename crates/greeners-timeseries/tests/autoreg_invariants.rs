use greeners_timeseries::autoreg::AutoReg;
use greeners_timeseries::autoreg::ARDL;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// AutoReg recovers the AR(2) coefficients.
#[test]
fn test_autoreg_ar2_recovery() {
    let n = 300;
    let mut rng = StdRng::seed_from_u64(9301);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let c = 1.0;
    let phi1 = 0.5;
    let phi2 = -0.3;

    let mut y_vec = Vec::with_capacity(n);
    let mut y1 = 0.0;
    let mut y2 = 0.0;
    for _ in 0..n {
        let y = c + phi1 * y1 + phi2 * y2 + noise.sample(&mut rng);
        y_vec.push(y);
        y2 = y1;
        y1 = y;
    }
    let y = Array1::from_vec(y_vec);

    let result = AutoReg::fit(&y, 2, None, "c").unwrap();
    assert_eq!(result.lags, 2);
    assert_eq!(result.trend, "c");
    approx_zero((result.params[0] - c).abs(), 0.2);
    approx_zero((result.params[1] - phi1).abs(), 0.1);
    approx_zero((result.params[2] - phi2).abs(), 0.1);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert_eq!(result.n_obs, n - 2);
}

/// AutoReg with an exogenous regressor recovers both the AR coefficient and
/// the exogenous slope.
#[test]
fn test_autoreg_exog_recovery() {
    let n = 300;
    let mut rng = StdRng::seed_from_u64(9302);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let c = 0.5;
    let phi = 0.6;
    let beta = 1.5;

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut prev_y = 0.0;
    for _ in 0..n {
        let x = noise.sample(&mut rng);
        let y = c + phi * prev_y + beta * x + noise.sample(&mut rng);
        x_vec.push(x);
        y_vec.push(y);
        prev_y = y;
    }
    let y = Array1::from_vec(y_vec);
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();

    let result = AutoReg::fit(&y, 1, Some(&x), "c").unwrap();
    assert_eq!(result.params.len(), 3); // const, y.L1, x1
    approx_zero((result.params[0] - c).abs(), 0.2);
    approx_zero((result.params[1] - phi).abs(), 0.1);
    approx_zero((result.params[2] - beta).abs(), 0.1);
    assert!(result.residuals.iter().all(|&v| v.is_finite()));
}

/// ARDL recovers the contemporaneous and lagged effects of an exogenous
/// regressor.
#[test]
fn test_ardl_recovery() {
    let n = 400;
    let mut rng = StdRng::seed_from_u64(9303);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let c = 0.5;
    let rho = 0.3; // AR(1) component on y
    let beta0 = 1.0; // contemporaneous x effect
    let beta1 = 0.5; // lag-1 x effect

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut prev_y = 0.0;
    let mut prev_x = 0.0;
    for _ in 0..n {
        let x = noise.sample(&mut rng);
        let y = c + rho * prev_y + beta0 * x + beta1 * prev_x + noise.sample(&mut rng);
        x_vec.push(x);
        y_vec.push(y);
        prev_y = y;
        prev_x = x;
    }
    let y = Array1::from_vec(y_vec);
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();

    let result = ARDL::fit(&y, &x, 1, 1).unwrap();
    assert_eq!(result.y_lags, 1);
    assert_eq!(result.x_lags, 1);
    // Columns: const, y.L1, x1, x1.L1
    approx_zero((result.params[0] - c).abs(), 0.2);
    approx_zero((result.params[1] - rho).abs(), 0.1);
    approx_zero((result.params[2] - beta0).abs(), 0.15);
    approx_zero((result.params[3] - beta1).abs(), 0.15);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
}

/// Input validation.
#[test]
fn test_autoreg_ardl_input_validation() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(AutoReg::fit(&y, 5, None, "c").is_err());
    assert!(ARDL::fit(
        &y,
        &Array2::from_shape_vec((3, 1), vec![1.0; 3]).unwrap(),
        2,
        2
    )
    .is_err());

    let x_mismatch = Array2::from_shape_vec((2, 1), vec![1.0; 2]).unwrap();
    assert!(ARDL::fit(&y, &x_mismatch, 1, 1).is_err());
}
