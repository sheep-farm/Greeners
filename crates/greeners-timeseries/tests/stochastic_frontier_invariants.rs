use greeners_timeseries::stochastic_frontier::StochasticFrontier;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

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
        y[i] = 1.0 + 2.0 * x[(i, 1)] - noise.abs();
    }
    (y, x)
}

/// StochasticFrontier returns finite parameters and expected shapes.
#[test]
fn test_sfa_shape_and_finite() {
    let (y, x) = make_sfa_data(50, 10001);
    let r = StochasticFrontier::fit_production(&y, &x, None).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.beta.len(), x.ncols());
    assert_eq!(r.efficiency.len(), y.len());
    assert!(r.beta.iter().all(|v| v.is_finite()));
    assert!(r.efficiency.iter().all(|v| v.is_finite()));
    assert!(r.efficiency.iter().all(|v| *v > 0.0 && *v <= 1.0));
    assert!(r.sigma_v > 0.0);
    assert!(r.sigma_u >= 0.0);
    assert!(r.log_likelihood.is_finite());
    assert_eq!(r.model_type, "production");
}

/// Cost frontier returns the expected model type and finite mean efficiency.
#[test]
fn test_sfa_cost_type() {
    let (y, x) = make_sfa_data(50, 10002);
    let r = StochasticFrontier::fit_cost(&y, &x, None).unwrap();
    assert_eq!(r.model_type, "cost");
    assert!(r.mean_efficiency > 0.0 && r.mean_efficiency <= 1.0);
}

/// Input validation rejects mismatched shapes.
#[test]
fn test_sfa_input_validation() {
    let (y, _x) = make_sfa_data(20, 10003);
    let x_bad = Array2::from_shape_vec((25, 2), vec![0.0; 50]).unwrap();
    assert!(StochasticFrontier::fit_production(&y, &x_bad, None).is_err());
}
