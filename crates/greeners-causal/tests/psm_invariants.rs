use greeners_causal::psm::PSM;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn psm_runs_and_recovers_att() {
    let n = 120;
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n * 2);

    for _ in 0..n {
        let x1 = noise.sample(&mut rng);
        let x2 = noise.sample(&mut rng);
        x.push(x1);
        x.push(x2);

        let treated = (0.5 + 0.8 * x1 + 0.3 * x2 + noise.sample(&mut rng)) > 0.0;
        let d_i = if treated { 1.0 } else { 0.0 };
        let y_val = 2.0 * x1 + 1.5 * d_i + noise.sample(&mut rng);

        y.push(y_val);
        d.push(d_i);
    }

    let y = Array1::from_vec(y);
    let d = Array1::from_vec(d);
    let x = Array2::from_shape_vec((n, 2), x).unwrap();

    let result = PSM::fit(&y, &d, &x, 2, None, false, 30, None).unwrap();

    assert!(result.att.is_finite());
    assert!(result.att > 0.0, "ATT should be positive: {}", result.att);
    assert!(result.se >= 0.0 && result.se.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(result.n_treated > 0);
    assert!(result.n_control > 0);
    assert!(result.n_matched_treated > 0);
    assert_eq!(result.propensity_scores.len(), n);
    assert!(!result.matched_pairs.is_empty());
}

#[test]
fn psm_input_validation() {
    let n = 20;
    let y = Array1::from_vec(vec![0.0; n]);
    let d = Array1::from_vec(vec![0.0; n - 1]);
    let x = Array2::zeros((n, 1));

    // Length mismatch
    assert!(PSM::fit(&y, &d, &x, 1, None, false, 10, None).is_err());

    // k == 0
    let d2 = Array1::from_vec(vec![0.0; n]);
    assert!(PSM::fit(&y, &d2, &x, 0, None, false, 10, None).is_err());
}
