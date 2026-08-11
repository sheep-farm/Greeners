use greeners::{CorrStructure, CovarianceType, Family, Link, GEE, OLS};
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// GEE with Gaussian family, identity link and independent working correlation
/// is equivalent to pooled OLS.
#[test]
fn test_gee_independence_identity_equals_ols() {
    // Two groups with repeated observations.
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0,
            4.0, 1.0, 5.0,
        ],
    )
    .unwrap();
    let y = x.column(0).to_owned() * 1.0 + x.column(1).to_owned() * 2.0;
    let groups = Array1::from(vec![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let gee = GEE::fit(
        &y,
        &x,
        &groups,
        &Family::Gaussian,
        &Link::Identity,
        &CorrStructure::Independence,
    )
    .unwrap();

    for j in 0..ols.params.len() {
        approx_zero((gee.params[j] - ols.params[j]).abs(), 1e-6);
    }

    // For the Independence structure the working correlation should be
    // close to the identity matrix (diagonal 1, off-diagonal 0).
    for i in 0..gee.working_correlation.nrows() {
        for j in 0..gee.working_correlation.ncols() {
            let expected = if i == j { 1.0 } else { 0.0 };
            approx_zero((gee.working_correlation[[i, j]] - expected).abs(), 1e-6);
        }
    }
}

/// Input validation: mismatched dimensions fail.
#[test]
fn test_gee_input_validation() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0; 5]).unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let groups = Array1::from(vec![0usize, 0, 1, 1, 1]);

    assert!(GEE::fit(
        &y,
        &x,
        &groups,
        &Family::Gaussian,
        &Link::Identity,
        &CorrStructure::Independence
    )
    .is_ok());

    let bad_groups = Array1::from(vec![0usize, 0, 1, 1]);
    assert!(GEE::fit(
        &y,
        &x,
        &bad_groups,
        &Family::Gaussian,
        &Link::Identity,
        &CorrStructure::Independence
    )
    .is_err());
}
