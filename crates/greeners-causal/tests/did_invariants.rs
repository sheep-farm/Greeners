use greeners_causal::did::DiffInDiff;
use greeners_core::types::CovarianceType;
use ndarray::Array1;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Canonical 2x2 DiD: the ATT equals the difference of differences.
#[test]
fn test_did_att_equals_difference_in_differences() {
    // Two control units and two treated units, pre and post.
    let y = Array1::from(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 6.0, 6.0]);
    let treated = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let post = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);

    let result = DiffInDiff::fit(&y, &treated, &post, CovarianceType::NonRobust).unwrap();

    // (treated_post - treated_pre) - (control_post - control_pre)
    let att = (6.0 - 3.0) - (2.0 - 1.0);
    approx_zero(result.att - att, 1e-10);

    // Group means are correctly computed.
    approx_zero(result.control_pre_mean - 1.0, 1e-10);
    approx_zero(result.control_post_mean - 2.0, 1e-10);
    approx_zero(result.treated_pre_mean - 3.0, 1e-10);
    approx_zero(result.treated_post_mean - 6.0, 1e-10);

    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
}

/// Mismatched input lengths are rejected.
#[test]
fn test_did_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let treated = Array1::from(vec![0.0, 0.0, 1.0, 1.0]);
    let post = Array1::from(vec![0.0, 1.0, 0.0]);

    assert!(DiffInDiff::fit(&y, &treated, &post, CovarianceType::NonRobust).is_err());
}
