use greeners_causal::dr_learner::DRLearner;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_dr_data(n: usize, seed: u64) -> (Array1<f64>, Vec<bool>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let x_vec: Vec<f64> = (0..n * 2).map(|_| dist.sample(&mut rng)).collect();
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let t: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let conf = 0.4 * x[(i, 0)] + 0.6 * x[(i, 1)];
        let effect = if t[i] { 0.5 } else { 0.0 };
        let e: f64 = StandardNormal.sample(&mut rng);
        y[i] = conf + effect + 0.3 * e;
    }
    (y, t, x)
}

/// DRLearner returns finite CATE predictions and ATE.
#[test]
fn test_dr_learner_shape_and_finite() {
    let (y, t, x) = make_dr_data(50, 18001);
    let r = DRLearner::fit(&y, &t, &x, None, None).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.n_features, x.ncols());
    assert_eq!(r.cate.len(), y.len());
    assert!(r.cate.iter().all(|v| v.is_finite()));
    assert!(r.ate.is_finite());
    assert!(r.ate_se.is_finite());
    assert!(r.propensity.iter().all(|&p| (0.0..=1.0).contains(&p)));
}

/// CATE regression coefficients are finite and have expected length.
#[test]
fn test_dr_learner_coefs() {
    let (y, t, x) = make_dr_data(60, 18002);
    let r = DRLearner::fit(&y, &t, &x, Some(2), None).unwrap();
    assert_eq!(r.cate_coefficients.len(), x.ncols() + 1);
    assert!(r.cate_coefficients.iter().all(|v| v.is_finite()));
}

/// Input validation rejects too few obs or imbalanced treatment.
#[test]
fn test_dr_learner_input_validation() {
    let (y, t, x) = make_dr_data(50, 18003);
    let t_all = vec![true; 50];
    assert!(DRLearner::fit(&y, &t_all, &x, None, None).is_err());
    let y_short = Array1::from_vec(vec![0.0; 5]);
    assert!(DRLearner::fit(&y_short, &t, &x, None, None).is_err());
}
