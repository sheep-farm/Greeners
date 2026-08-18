use greeners::linalg::LinalgInverse as _;
use greeners::{CovarianceType, DataFrame, Formula, OLS};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

fn build_df(y: &[f64], x1: &[f64]) -> (DataFrame, Formula) {
    let mut data = IndexMap::new();
    data.insert("y".to_string(), Array1::from(y.to_vec()));
    data.insert("x1".to_string(), Array1::from(x1.to_vec()));
    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1").unwrap();
    (df, formula)
}

fn build_noisy_data() -> (DataFrame, Formula, Array1<f64>, Array2<f64>) {
    // y = 1 + 2*x + small noise
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![3.1, 4.8, 7.4, 9.1, 10.9, 13.2, 15.1, 16.8];

    let (df, formula) = build_df(&y, &x1);
    let (y_arr, x_arr) = df.to_design_matrix(&formula).unwrap();
    (df, formula, y_arr, x_arr)
}

/// The OLS normal equations: X' (y - X β̂) = 0.
/// This is the first-order condition of the least-squares problem.
#[test]
fn test_ols_normal_equations() {
    let (_df, _formula, y, x) = build_noisy_data();
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let residuals = result.residuals(&y, &x);
    let xt_e = x.t().dot(&residuals);
    for &v in xt_e.iter() {
        approx_zero(v, 1e-10);
    }
}

/// When X includes a constant column, the residuals sum to zero.
/// Proof: the first row of X' e = 0 is 1' e = Σ e_i = 0.
#[test]
fn test_ols_residuals_sum_to_zero() {
    let (_df, _formula, y, x) = build_noisy_data();
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let residuals = result.residuals(&y, &x);
    approx_zero(residuals.sum(), 1e-10);
}

/// The projection matrix P = X (X'X)^-1 X' is idempotent and symmetric.
/// Therefore y = P y + (I - P) y, with P y ⟂ (I - P) y.
#[test]
fn test_ols_projection_properties() {
    let (_df, _formula, y, x) = build_noisy_data();
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    let fitted = result.fitted_values(&x);
    let residuals = result.residuals(&y, &x);

    // y = fitted + residuals
    for i in 0..y.len() {
        approx_zero((y[i] - fitted[i] - residuals[i]).abs(), 1e-10);
    }

    // fitted is in the column space of X, residuals are orthogonal to it
    let fitted_dot_residuals = (&fitted * &residuals).sum();
    approx_zero(fitted_dot_residuals, 1e-10);

    // Idempotency of P
    let xt_x = x.t().dot(&x);
    let xt_x_inv = xt_x.inv().unwrap();
    let p = x.dot(&xt_x_inv).dot(&x.t());
    let p2 = p.dot(&p);
    for (pi, p2i) in p.iter().zip(p2.iter()) {
        approx_zero((pi - p2i).abs(), 1e-10);
    }

    // P y = fitted
    let py = p.dot(&y);
    for i in 0..y.len() {
        approx_zero((py[i] - fitted[i]).abs(), 1e-10);
    }
}

/// OLS is linear in y: scaling y by c scales all coefficients by c.
#[test]
fn test_ols_invariance_scale_y() {
    let (_df, _formula, y, x) = build_noisy_data();
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    let c = 2.5;
    let y_scaled = y.mapv(|v| v * c);
    let result_scaled = OLS::fit(&y_scaled, &x, CovarianceType::NonRobust).unwrap();

    for i in 0..result.params.len() {
        approx_zero(
            (result_scaled.params[i] - c * result.params[i]).abs(),
            1e-10,
        );
    }
}

/// Translation of a regressor by a constant a changes the intercept by -a * slope,
/// but leaves the slope itself unchanged.
#[test]
fn test_ols_invariance_translation_x() {
    let (_df, _formula, y, x) = build_noisy_data();
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    // Original formula: y ~ x1, columns are [const, x1]
    let a = 3.0;
    let x1_col = x.column(1).to_owned();
    let x1_shifted = x1_col.mapv(|v| v + a);
    let mut x_shifted = x.clone();
    x_shifted.column_mut(1).assign(&x1_shifted);

    let result_shifted = OLS::fit(&y, &x_shifted, CovarianceType::NonRobust).unwrap();

    // slope unchanged
    approx_zero((result_shifted.params[1] - result.params[1]).abs(), 1e-10);

    // intercept changes by -a * slope
    let expected_intercept = result.params[0] - a * result.params[1];
    approx_zero((result_shifted.params[0] - expected_intercept).abs(), 1e-10);
}

/// Exact arithmetic for a small, integer design where the true coefficients are rational.
/// This is the closest we can get to a "proof by exact computation" with f64.
#[test]
fn test_ols_exact_small_integer_design() {
    // X (with intercept) is 3x2, y is 3x1. The true beta is [2, 3].
    let y = Array1::from(vec![5.0, 8.0, 11.0]);
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(
        (result.params[0] - 2.0).abs() < 1e-14,
        "intercept: {}",
        result.params[0]
    );
    assert!(
        (result.params[1] - 3.0).abs() < 1e-14,
        "slope: {}",
        result.params[1]
    );

    // SSR = 0 (perfect fit)
    assert!((result.r_squared - 1.0).abs() < 1e-14);
    let residuals = result.residuals(&y, &x);
    for &r in residuals.iter() {
        assert!(r.abs() < 1e-14, "residual {}", r);
    }
}
