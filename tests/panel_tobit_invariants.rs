use greeners::{CovarianceType, PanelTobit, OLS};
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// When no observation is censored, Panel Tobit EM reduces to OLS.
#[test]
fn test_panel_tobit_no_censoring_equals_ols() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    )
    .unwrap();
    let y = &x.column(0) * 1.0 + &x.column(1) * 2.0;

    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    // Two panels, three observations each.
    let panel_ids = vec![1i64, 1, 1, 2, 2, 2];
    let censor_left = -1000.0; // far below all y values

    let tobit = PanelTobit::fit(&y, &x, &panel_ids, censor_left, None).unwrap();

    for j in 0..ols.params.len() {
        approx_zero((tobit.beta[j] - ols.params[j]).abs(), 1e-8);
    }
}

/// Censoring point above all observations should be rejected (no uncensored
/// observations to identify beta).
#[test]
fn test_panel_tobit_all_censored_fails() {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let panel_ids = vec![1i64, 1, 1];

    assert!(PanelTobit::fit(&y, &x, &panel_ids, 10.0, None).is_err());
}
