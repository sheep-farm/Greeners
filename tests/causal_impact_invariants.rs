use greeners::CausalImpact;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::{Distribution, StandardNormal, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_causal_impact_data(n: usize, k: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let c_vec: Vec<f64> = (0..n * k).map(|_| dist.sample(&mut rng)).collect();
    let controls = Array2::from_shape_vec((n, k), c_vec).unwrap();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let noise: f64 = StandardNormal.sample(&mut rng);
        let base: f64 = controls.row(i).sum();
        y[i] = 0.5 + 0.4 * base + noise;
    }
    (y, controls)
}

/// CausalImpact returns expected shapes and finite post-treatment effects.
#[test]
fn test_causal_impact_shape_and_finite() {
    let n = 50;
    let (y, controls) = make_causal_impact_data(n, 2, 19001);
    let r = CausalImpact::fit(&y, &controls, 30, None).unwrap();
    assert_eq!(r.y.len(), n);
    assert_eq!(r.counterfactual.len(), n);
    assert_eq!(r.pointwise_effect.len(), n);
    assert_eq!(r.cumulative_effect.len(), n);
    assert_eq!(r.n_pre, 30);
    assert_eq!(r.n_post, n - 30);
    assert!(r.avg_effect.is_finite());
    assert!(r.total_effect.is_finite());
    assert!(r.pre_r_squared >= 0.0 && r.pre_r_squared <= 1.0);
    assert!(r.p_effect_positive >= 0.0 && r.p_effect_positive <= 1.0);
}

/// Pre-period counterfactual equals observed y (in-sample) for a noiseless design.
#[test]
fn test_causal_impact_pre_fit() {
    let n = 40;
    let mut y = Array1::zeros(n);
    let mut controls = Array2::zeros((n, 1));
    for i in 0..n {
        controls[(i, 0)] = i as f64;
        y[i] = 0.5 + 0.4 * controls[(i, 0)];
    }
    let r = CausalImpact::fit(&y, &controls, 25, None).unwrap();
    for i in 0..25 {
        assert!((y[i] - r.counterfactual[i]).abs() < 1e-6);
    }
}

/// Input validation rejects out-of-bounds treatment periods and mismatched rows.
#[test]
fn test_causal_impact_input_validation() {
    let (y, controls) = make_causal_impact_data(50, 2, 19002);
    assert!(CausalImpact::fit(&y, &controls, 48, None).is_err());
    assert!(CausalImpact::fit(&y, &controls, 1, None).is_err());
    let controls_bad = Array2::from_shape_vec((45, 2), vec![0.0; 90]).unwrap();
    assert!(CausalImpact::fit(&y, &controls_bad, 30, None).is_err());
}
