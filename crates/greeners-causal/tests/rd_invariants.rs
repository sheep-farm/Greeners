use greeners_causal::rd::RdKernel;
use greeners_causal::rd::RD;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_rd_data(
    seed: u64,
    n: usize,
    cutoff: f64,
    jump: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.3).unwrap();
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);

    for i in 0..n {
        let xi = -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0);
        let treat = if xi >= cutoff { 1.0 } else { 0.0 };
        x.push(xi);
        d.push(treat);
        y.push(1.0 + 0.5 * xi + jump * treat + noise.sample(&mut rng));
    }

    (
        Array1::from_vec(y),
        Array1::from_vec(x),
        Array1::from_vec(d),
    )
}

/// Sharp RD recovers a finite treatment effect near the true jump.
#[test]
fn test_rd_sharp_invariants() {
    let (y, x, _) = make_rd_data(12345, 100, 0.0, 2.5);
    let result = RD::fit(
        &y,
        &x,
        0.0,
        None,
        1,
        RdKernel::Triangular,
        Some(("y".into(), "x".into())),
    )
    .unwrap();

    assert!(result.n_total <= 100);
    assert_eq!(result.n_left + result.n_right, result.n_total);
    assert!(result.tau.is_finite());
    assert!(result.se.is_finite());
    assert!(result.bandwidth > 0.0 && result.bandwidth.is_finite());
    assert!(
        (result.tau - 2.5).abs() < 1.0,
        "tau far from true jump: {}",
        result.tau
    );
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert_eq!(result.kernel, RdKernel::Triangular);
    assert!(!result.is_fuzzy);
    assert!(result.outcome_name.as_ref().unwrap() == "y");
}

/// Fuzzy RD returns a finite LATE and first-stage diagnostics.
#[test]
fn test_rd_fuzzy_invariants() {
    let (y, x, d) = make_rd_data(23456, 100, 0.0, 2.5);
    let result = RD::fit_fuzzy(
        &y,
        &d,
        &x,
        0.0,
        None,
        1,
        RdKernel::Triangular,
        Some(("y".into(), "x".into(), "d".into())),
    )
    .unwrap();

    assert!(result.tau.is_finite());
    assert!(result.se.is_finite());
    assert!(result.bandwidth > 0.0);
    assert!(result.is_fuzzy);
    assert!(result.first_stage_tau.is_some());
    assert!(result.first_stage_se.is_some());
    let fs_tau = result.first_stage_tau.unwrap();
    assert!(
        (fs_tau - 1.0).abs() < 0.2,
        "first-stage jump too far: {}",
        fs_tau
    );
}

/// Input validation rejects length mismatches, NaNs, and insufficient local samples.
#[test]
fn test_rd_input_validation() {
    let (y, x, _) = make_rd_data(11111, 100, 0.0, 2.5);

    let x_short = x.slice(ndarray::s![0..50]).to_owned();
    assert!(RD::fit(&y, &x_short, 0.0, None, 1, RdKernel::Triangular, None).is_err());

    let mut y_nan = y.clone();
    y_nan[10] = f64::NAN;
    assert!(RD::fit(&y_nan, &x, 0.0, None, 1, RdKernel::Triangular, None).is_err());

    // Too few observations on one side for a local linear fit.
    let y_small = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let x_small = Array1::from_vec(vec![-0.9, -0.8, 0.8, 0.9]);
    assert!(RD::fit(
        &y_small,
        &x_small,
        0.0,
        Some(0.1),
        1,
        RdKernel::Uniform,
        None
    )
    .is_err());
}
