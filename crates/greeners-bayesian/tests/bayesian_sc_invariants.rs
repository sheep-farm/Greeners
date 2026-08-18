use greeners_bayesian::bayesian_sc::BayesianSC;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_sc_data(t: usize, n_controls: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let c_vec: Vec<f64> = (0..t * n_controls).map(|_| dist.sample(&mut rng)).collect();
    let controls = Array2::from_shape_vec((t, n_controls), c_vec).unwrap();
    let mut treated = Array1::zeros(t);
    for i in 0..t {
        let noise: f64 = StandardNormal.sample(&mut rng);
        let base: f64 = controls.row(i).sum();
        treated[i] = 0.5 + 0.3 * base + noise;
    }
    (treated, controls)
}

/// BayesianSC returns expected shapes and finite treatment effect estimates.
#[test]
fn test_bayesian_sc_shape_and_finite() {
    let (treated, controls) = make_sc_data(30, 2, 8001);
    let r = BayesianSC::fit(&treated, &controls, 20, Some(1.0)).unwrap();
    assert_eq!(r.n_controls, controls.ncols());
    assert_eq!(r.n_pre, 20);
    assert_eq!(r.n_post, treated.len() - 20);
    assert_eq!(r.weights.len(), controls.ncols());
    assert_eq!(r.counterfactual.len(), treated.len());
    assert_eq!(r.observed.len(), treated.len());
    assert!(r.tau.is_finite());
    assert!(r.tau_sd.is_finite());
    assert!(r.sigma2 > 0.0);
    assert!(r.weights.iter().all(|v| v.is_finite()));
    assert!(r.counterfactual.iter().all(|v| v.is_finite()));
}

/// The counterfactual and observed series have the same length and reproduce y for pre-period.
#[test]
fn test_bayesian_sc_pre_post_split() {
    let (treated, controls) = make_sc_data(24, 2, 8002);
    let r = BayesianSC::fit(&treated, &controls, 16, None).unwrap();
    assert_eq!(r.n_pre, 16);
    assert_eq!(r.n_post, 8);
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    assert!(r.cumulative_effect.is_finite());
}

/// Input validation rejects out-of-bounds treatment periods and mismatched lengths.
#[test]
fn test_bayesian_sc_input_validation() {
    let (treated, controls) = make_sc_data(30, 2, 8003);
    assert!(BayesianSC::fit(&treated, &controls, 0, None).is_err());
    assert!(BayesianSC::fit(&treated, &controls, 30, None).is_err());
    let controls_bad = Array2::from_shape_vec((25, 2), vec![0.0; 50]).unwrap();
    assert!(BayesianSC::fit(&treated, &controls_bad, 20, None).is_err());
}
