use greeners_core::linalg::LinalgInverse as _;
use greeners_core::types::CovarianceType;
use greeners_ols::iv::IV;
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Small exactly identified IV design:
/// y = 1 + 5*x,  x = 2 + 3*z
/// Z = [const, z],  X = [const, x]
/// With n=4 and z=[1,2,3,4] the 2SLS coefficients are exactly [1, 5].
fn exact_iv_design() -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let z = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let x = z.mapv(|v| 2.0 + 3.0 * v); // x = 2 + 3z
    let y = x.mapv(|v| 1.0 + 5.0 * v); // y = 1 + 5x

    let n = z.len();
    let x_mat = Array2::from_shape_vec((n, 2), {
        let mut v = Vec::with_capacity(2 * n);
        for i in 0..n {
            v.push(1.0);
            v.push(x[i]);
        }
        v
    })
    .unwrap();

    let z_mat = Array2::from_shape_vec((n, 2), {
        let mut v = Vec::with_capacity(2 * n);
        for i in 0..n {
            v.push(1.0);
            v.push(z[i]);
        }
        v
    })
    .unwrap();

    (y, x_mat, z_mat)
}

/// First stage: X_hat = Z (Z'Z)^-1 Z' X.
/// X_hat is the orthogonal projection of each column of X onto the column
/// space of Z.
#[test]
fn test_iv_first_stage_projection() {
    let (y, x, z) = exact_iv_design();

    let zt_z = z.t().dot(&z);
    let zt_z_inv = zt_z.inv().unwrap();
    let zt_x = z.t().dot(&x);
    let first_stage_coeffs = zt_z_inv.dot(&zt_x);
    let x_hat = z.dot(&first_stage_coeffs);

    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust).unwrap();
    let predicted = x.dot(&result.params);
    let residuals = &y - &predicted;

    // The 2SLS estimator satisfies X_hat' e = 0 by construction.
    let xht_e = x_hat.t().dot(&residuals);
    for &v in xht_e.iter() {
        approx_zero(v, 1e-8);
    }
}

/// 2SLS coefficients solve (X_hat' X_hat) β = X_hat' y.
/// This is the second-stage normal equation.
#[test]
fn test_iv_beta_satisfies_second_stage_normal_equations() {
    let (y, x, z) = exact_iv_design();

    // Reconstruct X_hat manually
    let zt_z = z.t().dot(&z);
    let zt_z_inv = zt_z.inv().unwrap();
    let zt_x = z.t().dot(&x);
    let first_stage_coeffs = zt_z_inv.dot(&zt_x);
    let x_hat = z.dot(&first_stage_coeffs);

    // Compute beta explicitly from X_hat
    let xht_xh = x_hat.t().dot(&x_hat);
    let xht_xh_inv = xht_xh.inv().unwrap();
    let xht_y = x_hat.t().dot(&y);
    let expected_beta = xht_xh_inv.dot(&xht_y);

    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust).unwrap();
    for i in 0..result.params.len() {
        approx_zero((result.params[i] - expected_beta[i]).abs(), 1e-10);
    }
}

/// First-stage residuals v = X - X_hat must be orthogonal to Z.
#[test]
fn test_iv_first_stage_residuals_orthogonal_to_instruments() {
    let (_y, x, z) = exact_iv_design();

    let zt_z = z.t().dot(&z);
    let zt_z_inv = zt_z.inv().unwrap();
    let zt_x = z.t().dot(&x);
    let first_stage_coeffs = zt_z_inv.dot(&zt_x);
    let x_hat = z.dot(&first_stage_coeffs);
    let v = &x - &x_hat;

    let ztv = z.t().dot(&v);
    for &val in ztv.iter() {
        approx_zero(val, 1e-10);
    }
}

/// IV/2SLS is linear in y: scaling y by c scales β by c.
#[test]
fn test_iv_invariance_scale_y() {
    let (y, x, z) = exact_iv_design();
    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust).unwrap();

    let c = 2.5;
    let y_scaled = y.mapv(|v| v * c);
    let result_scaled = IV::fit(&y_scaled, &x, &z, CovarianceType::NonRobust).unwrap();

    for i in 0..result.params.len() {
        approx_zero(
            (result_scaled.params[i] - c * result.params[i]).abs(),
            1e-10,
        );
    }
}

/// Exact arithmetic for the hand-crafted DGP.
#[test]
fn test_iv_exact_small_integer_design() {
    let (y, x, z) = exact_iv_design();
    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(
        (result.params[0] - 1.0).abs() < 1e-12,
        "intercept: {}",
        result.params[0]
    );
    assert!(
        (result.params[1] - 5.0).abs() < 1e-12,
        "slope: {}",
        result.params[1]
    );
}

/// The order condition (L >= K) must be enforced.
#[test]
fn test_iv_order_condition_fails() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    // X has 2 columns (intercept + endogenous)
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
    // Z has only 1 column (constant) -> underidentified
    let z = Array2::from_shape_vec((4, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap();

    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust);
    assert!(result.is_err());
}
