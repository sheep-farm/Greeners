use greeners_ols::quantile::QuantileReg;
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Perfectly collinear data: y = 1 + 2x. Any quantile of a degenerate
/// distribution is zero, so the quantile regression line is the same line.
#[test]
fn test_quantile_perfect_fit_independent_of_tau() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let y = x.column(0).to_owned() * 1.0 + x.column(1).to_owned() * 2.0;

    for &tau in &[0.25, 0.5, 0.75] {
        let result = QuantileReg::fit(&y, &x, tau, 0).unwrap();
        approx_zero((result.params[0] - 1.0).abs(), 1e-8);
        approx_zero((result.params[1] - 2.0).abs(), 1e-8);
    }
}

/// Quantile regression is equivariant to scaling of y.
#[test]
fn test_quantile_scale_invariance() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let y = Array1::from(vec![0.5, 1.8, 3.1, 4.0, 5.2, 6.9]);

    let base = QuantileReg::fit(&y, &x, 0.5, 0).unwrap();
    let scaled = QuantileReg::fit(&y.mapv(|v| 2.0 * v), &x, 0.5, 0).unwrap();

    for i in 0..base.params.len() {
        approx_zero((scaled.params[i] - 2.0 * base.params[i]).abs(), 1e-5);
    }
}

/// Quantile regression is equivariant to translations of y: only the
/// intercept shifts.
#[test]
fn test_quantile_translation_invariance() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let y = Array1::from(vec![0.5, 1.8, 3.1, 4.0, 5.2, 6.9]);

    let base = QuantileReg::fit(&y, &x, 0.5, 0).unwrap();
    let shifted = QuantileReg::fit(&y.mapv(|v| v + 10.0), &x, 0.5, 0).unwrap();

    approx_zero((shifted.params[0] - (base.params[0] + 10.0)).abs(), 1e-8);
    for i in 1..base.params.len() {
        approx_zero((shifted.params[i] - base.params[i]).abs(), 1e-8);
    }
}

/// tau must be in (0, 1).
#[test]
fn test_quantile_invalid_tau_fails() {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0]);

    assert!(QuantileReg::fit(&y, &x, 0.0, 0).is_err());
    assert!(QuantileReg::fit(&y, &x, 1.0, 0).is_err());
    assert!(QuantileReg::fit(&y, &x, -0.1, 0).is_err());
}
