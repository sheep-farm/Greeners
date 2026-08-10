use greeners::BayesianSFA;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::{Distribution, StandardNormal, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_sfa_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(0.0, 1.0);
    let mut x = Array2::zeros((n, 2));
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = dist.sample(&mut rng);
    }
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let noise: f64 = StandardNormal.sample(&mut rng);
        // production: y = 1 + 2*x - inefficiency(noise)
        y[i] = 1.0 + 2.0 * x[(i, 1)] - noise.abs();
    }
    (y, x)
}

/// BayesianSFA returns finite coefficients and efficiency estimates.
#[test]
fn test_bayesian_sfa_shape_and_finite() {
    let (y, x) = make_sfa_data(40, 9001);
    let r = BayesianSFA::fit_production(&y, &x, None, 20, 50).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.beta.len(), x.ncols());
    assert!(r.beta.iter().all(|v| v.is_finite()));
    assert!(r.beta_sd.iter().all(|v| v.is_finite()));
    assert!(r.sigma_v > 0.0);
    assert!(r.sigma_u >= 0.0);
    assert!(r.mean_efficiency > 0.0 && r.mean_efficiency <= 1.0);
    assert_eq!(r.model_type, "production");
}

/// Cost frontier returns the expected model type.
#[test]
fn test_bayesian_sfa_cost_type() {
    let (y, x) = make_sfa_data(40, 9002);
    let r = BayesianSFA::fit_cost(&y, &x, None, 20, 50).unwrap();
    assert_eq!(r.model_type, "cost");
    assert!(r.beta.len() == x.ncols());
    assert!(r.n_draws > 0);
}

/// Input validation rejects mismatched row counts.
#[test]
fn test_bayesian_sfa_input_validation() {
    let (y, _x) = make_sfa_data(20, 9003);
    let x_bad = Array2::from_shape_vec((25, 2), vec![0.0; 50]).unwrap();
    assert!(BayesianSFA::fit_production(&y, &x_bad, None, 10, 30).is_err());
}
