use greeners::{Copula, CopulaType};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn random_x(n: usize, k: usize, seed: u64) -> Array2<f64> {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    Array2::from_shape_vec((n, k), (0..n * k).map(|_| dist.sample(&mut rng)).collect()).unwrap()
}

fn assert_array2_finite(a: &Array2<f64>) {
    assert!(
        a.iter().all(|v| v.is_finite()),
        "non-finite value in Array2"
    );
}

/// Fit a Gaussian copula to bivariate normal data and check that the
/// estimated correlation is close to the true latent correlation.
#[test]
fn test_copula_gaussian_recovery() {
    let mut rng = make_rng(2025);
    let n = 120;
    let rho_true = 0.6;
    let mut x_vec = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let z1: f64 = StandardNormal.sample(&mut rng);
        let z2: f64 = StandardNormal.sample(&mut rng);
        let x1 = z1;
        let x2 = rho_true * z1 + (1.0f64 - rho_true * rho_true).sqrt() * z2;
        x_vec.push(x1);
        x_vec.push(x2);
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();

    let r = Copula::fit(&x, CopulaType::Gaussian, None).unwrap();
    assert_eq!(r.n_obs, n);
    assert_eq!(r.n_vars, 2);
    assert_array2_finite(&r.corr_matrix);
    assert_array2_finite(&r.kendall_tau);
    assert_array2_finite(&r.spearman_rho);
    assert!(r.log_likelihood.is_finite());
    assert!(r.aic.is_finite());
    assert!(r.bic.is_finite());

    let rho_hat = r.corr_matrix[[0, 1]];
    assert!(
        (rho_hat - rho_true).abs() < 0.15,
        "rho_hat={} too far from {}",
        rho_hat,
        rho_true
    );
}

/// All implemented copula types return finite results and expected shapes.
#[test]
fn test_copula_all_types_finite() {
    let x = random_x(60, 3, 2026);
    for &ctype in &[
        CopulaType::Gaussian,
        CopulaType::Clayton,
        CopulaType::Gumbel,
        CopulaType::Frank,
    ] {
        let r = Copula::fit(&x, ctype, None).unwrap();
        assert_eq!(r.n_obs, x.nrows());
        assert_eq!(r.n_vars, x.ncols());
        assert_eq!(r.corr_matrix.shape(), &[3, 3]);
        assert_eq!(r.kendall_tau.shape(), &[3, 3]);
        assert_eq!(r.spearman_rho.shape(), &[3, 3]);
        assert_array2_finite(&r.corr_matrix);
        assert_array2_finite(&r.kendall_tau);
        assert_array2_finite(&r.spearman_rho);
        assert!(r.log_likelihood.is_finite());
        assert!(r.aic.is_finite());
        assert!(r.bic.is_finite());
    }
}

/// Input validation rejects too few observations or a single variable.
#[test]
fn test_copula_input_validation() {
    let x_small = random_x(4, 2, 2027);
    assert!(Copula::fit(&x_small, CopulaType::Gaussian, None).is_err());
    let x_one = random_x(50, 1, 2027);
    assert!(Copula::fit(&x_one, CopulaType::Gaussian, None).is_err());
}
