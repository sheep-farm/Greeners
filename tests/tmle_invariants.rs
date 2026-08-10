use greeners::TMLE;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::{Distribution, StandardNormal, Uniform};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_tmle_data(n: usize, seed: u64) -> (Array1<f64>, Vec<bool>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let x_vec: Vec<f64> = (0..n * 2).map(|_| dist.sample(&mut rng)).collect();
    let w = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let t: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let base = 1.0 + 0.5 * w[(i, 0)] + 0.3 * w[(i, 1)];
        let effect = if t[i] { 0.4 } else { 0.0 };
        let noise: f64 = StandardNormal.sample(&mut rng);
        y[i] = base + effect + noise;
    }
    (y, t, w)
}

/// TMLE returns finite ATE and standard error.
#[test]
fn test_tmle_shape_and_finite() {
    let (y, t, w) = make_tmle_data(50, 15001);
    let r = TMLE::fit(&y, &t, &w).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.n_confounders, w.ncols());
    assert!(r.ate.is_finite());
    assert!(r.se.is_finite());
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    assert_eq!(r.propensity.len(), y.len());
    assert_eq!(r.targeted_q.len(), y.len());
    assert!(r.propensity.iter().all(|v| *v > 0.0 && *v < 1.0));
}

/// Propensity scores and EIF are bounded and have correct length.
#[test]
fn test_tmle_bounded_propensity() {
    let (y, t, w) = make_tmle_data(80, 15002);
    let r = TMLE::fit(&y, &t, &w).unwrap();
    assert!(r.propensity.iter().all(|&p| (0.0..=1.0).contains(&p)));
    assert_eq!(r.eif.len(), y.len());
    assert!(r.eif.iter().all(|v| v.is_finite()));
}

/// Input validation rejects mismatched lengths and too few treated/control.
#[test]
fn test_tmle_input_validation() {
    let (y, mut t, w) = make_tmle_data(50, 15003);
    t.truncate(5);
    assert!(TMLE::fit(&y, &t, &w).is_err());

    let t_all_true = vec![true; 50];
    assert!(TMLE::fit(&y, &t_all_true, &w).is_err());
}
