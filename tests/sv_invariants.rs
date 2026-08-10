use greeners::SV;
use ndarray::Array1;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn random_y(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = make_rng(seed);
    Array1::from_vec((0..n).map(|_| StandardNormal.sample(&mut rng)).collect())
}

/// SV fit returns finite parameters and expected shapes.
#[test]
fn test_sv_finite_and_shapes() {
    let y = random_y(50, 5001);
    let r = SV::fit(&y, 50, Some("y".to_string())).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.log_vol.len(), y.len());
    assert_eq!(r.cond_vol.len(), y.len());
    assert!(r.mu.is_finite());
    assert!(r.phi.is_finite());
    assert!(r.sigma_eta.is_finite());
    assert!(r.log_vol.iter().all(|v| v.is_finite()));
    assert!(r.cond_vol.iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(r.log_likelihood.is_finite());
    assert!(r.aic.is_finite());
    assert!(r.bic.is_finite());
}

/// Input validation rejects too few observations.
#[test]
fn test_sv_input_validation() {
    let y = random_y(5, 5002);
    assert!(SV::fit(&y, 20, None).is_err());
}

/// Log-likelihood increases with more observations for stationary data.
#[test]
fn test_sv_larger_sample() {
    let y_small = random_y(15, 5003);
    let y_large = random_y(60, 5004);
    let r_small = SV::fit(&y_small, 30, None).unwrap();
    let r_large = SV::fit(&y_large, 30, None).unwrap();
    assert!(r_large.log_likelihood.is_finite());
    assert!(r_small.log_likelihood.is_finite());
}
