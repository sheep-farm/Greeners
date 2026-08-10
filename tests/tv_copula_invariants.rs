use greeners::{TvCopula, TvCopulaType};
use ndarray::Array2;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn random_x(n: usize, k: usize, seed: u64) -> Array2<f64> {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    Array2::from_shape_vec(
        (n, k),
        (0..n * k)
            .map(|_| {
                let v: f64 = dist.sample(&mut rng);
                v
            })
            .collect(),
    )
    .unwrap()
}

/// TvCopula fits and returns the expected shapes for all copula types.
/// Outputs may be NaN for some types on random data, so only shape invariants are enforced.
#[test]
fn test_tvcopula_returns_ok_and_shapes() {
    let x = random_x(50, 2, 3001);
    for &ctype in &[
        TvCopulaType::Gaussian,
        TvCopulaType::Clayton,
        TvCopulaType::Gumbel,
    ] {
        let r = TvCopula::fit(&x, ctype, None).unwrap();
        assert_eq!(r.n_obs, x.nrows());
        assert_eq!(r.n_vars, x.ncols());
        assert_eq!(r.theta_path.len(), x.nrows());
        assert_eq!(r.kendall_tau_path.len(), x.nrows());
        assert_eq!(r.dynamics_params.len(), 3);
    }
}

/// A deterministic, well-separated design returns a finite result.
#[test]
fn test_tvcopula_deterministic_finite() {
    // Two variables that are perfectly concordant then perfectly discordant.
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, -1.0, 1.0, -2.0, 2.0, -3.0,
            3.0, -4.0, 4.0, -5.0, 5.0, -6.0, 6.0,
        ],
    )
    .unwrap();
    let r = TvCopula::fit(&x, TvCopulaType::Clayton, None).unwrap();
    assert!(r.log_likelihood.is_finite());
    assert!(r.aic.is_finite());
    assert!(r.bic.is_finite());
    assert_eq!(r.theta_path.len(), x.nrows());
}

/// Input validation rejects too few obs or a single variable.
#[test]
fn test_tvcopula_input_validation() {
    let x_small = random_x(8, 2, 3003);
    assert!(TvCopula::fit(&x_small, TvCopulaType::Gaussian, None).is_err());
    let x_one = random_x(50, 1, 3003);
    assert!(TvCopula::fit(&x_one, TvCopulaType::Gaussian, None).is_err());
}
