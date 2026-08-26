use greeners_causal::synth_did::SyntheticDiD;
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn synthetic_did_runs_and_recovers_att() {
    let n = 5;
    let t = 8;
    let treatment_period = 4;
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y_vec = Vec::with_capacity(n * t);
    let mut treated = vec![false; n];
    treated[0] = true;

    for i in 0..n {
        let unit_fe = noise.sample(&mut rng) * 2.0;
        for tt in 0..t {
            let post = if tt >= treatment_period { 1.0 } else { 0.0 };
            let treated_effect = if treated[i] { 2.0 * post } else { 0.0 };
            let y_val = 1.0 + 0.1 * (tt as f64) + unit_fe + treated_effect + noise.sample(&mut rng);
            y_vec.push(y_val);
        }
    }

    let y = Array2::from_shape_vec((n, t), y_vec).unwrap();

    let result = SyntheticDiD::fit(&y, &treated, treatment_period).unwrap();

    assert!(result.att.is_finite());
    assert!(
        (result.att - 2.0).abs() < 0.8,
        "ATT out of range: {}",
        result.att
    );
    assert!(result.se >= 0.0 && result.se.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert_eq!(result.n_treated, 1);
    assert_eq!(result.n_control, 4);
    assert_eq!(result.n_pre, 4);
    assert_eq!(result.n_post, 4);
    assert_eq!(result.n_periods, t);
    assert_eq!(result.synthetic_control.len(), t);
    assert_eq!(result.treated_avg.len(), t);
    assert!(
        (result.unit_weights.sum() - 1.0).abs() < 1e-6 || result.unit_weights.sum().abs() < 1e-9
    );
}

#[test]
fn synthetic_did_input_validation() {
    let y = Array2::zeros((4, 6));
    let treated = vec![false; 4];

    // No treated
    assert!(SyntheticDiD::fit(&y, &treated, 3).is_err());

    // treatment_period out of range
    let treated2 = vec![true, false, false, false];
    assert!(SyntheticDiD::fit(&y, &treated2, 0).is_err());
    assert!(SyntheticDiD::fit(&y, &treated2, 6).is_err());

    // treated length mismatch
    let treated3 = vec![true, false, false];
    assert!(SyntheticDiD::fit(&y, &treated3, 3).is_err());
}
