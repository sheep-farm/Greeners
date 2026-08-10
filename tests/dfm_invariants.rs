use greeners::DFM;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn generate_dfm_data(t: usize, n_series: usize, n_factors: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    // Generate n_factors independent AR(1) factors.
    let mut factors = Array2::<f64>::zeros((t, n_factors));
    for i in 1..t {
        for f in 0..n_factors {
            factors[[i, f]] = 0.6 * factors[[i - 1, f]] + noise.sample(&mut rng);
        }
    }

    // Observed series are linear combinations of factors.
    let mut x = Array2::zeros((t, n_series));
    for i in 0..t {
        for s in 0..n_series {
            let loading = (s as f64 + 1.0) / (n_series as f64 + 1.0);
            x[[i, s]] = loading * factors[[i, s % n_factors]] + noise.sample(&mut rng);
        }
    }

    x
}

#[test]
fn test_dfm_runs_and_produces_finite_output() {
    let x = generate_dfm_data(60, 5, 2, 17001);

    let result = DFM::fit(&x, 2, 20, None).unwrap();

    assert_eq!(result.n_obs, 60);
    assert_eq!(result.n_series, 5);
    assert_eq!(result.n_factors, 2);
    assert_eq!(result.factors.shape(), &[60, 2]);
    assert_eq!(result.loadings.shape(), &[5, 2]);
    assert_eq!(result.factor_ar.shape(), &[2, 2]);
    assert_eq!(result.factor_cov.shape(), &[2, 2]);
    assert_eq!(result.obs_variances.len(), 5);
    assert!(result.factors.iter().all(|&v| v.is_finite()));
    assert!(result.loadings.iter().all(|&v| v.is_finite()));
    assert!(result.factor_ar.iter().all(|&v| v.is_finite()));
    assert!(result.factor_cov.iter().all(|&v| v.is_finite()));
    assert!(result
        .obs_variances
        .iter()
        .all(|&v| v.is_finite() && v > 0.0));
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_dfm_factor_ar_is_stable() {
    let x = generate_dfm_data(80, 5, 2, 17002);

    let result = DFM::fit(
        &x,
        2,
        20,
        Some(vec![
            "x0".into(),
            "x1".into(),
            "x2".into(),
            "x3".into(),
            "x4".into(),
        ]),
    )
    .unwrap();

    // Eigenvalues of factor AR should be inside the unit circle (stationary factors).
    let a = &result.factor_ar;
    let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
    let trace = a[[0, 0]] + a[[1, 1]];
    let disc = trace * trace - 4.0 * det;
    let (e1, e2) = if disc >= 0.0 {
        let s = disc.sqrt();
        ((trace + s) / 2.0, (trace - s) / 2.0)
    } else {
        let r = (-disc).sqrt() / 2.0;
        let m = trace / 2.0;
        (m.hypot(r), m.hypot(r))
    };
    assert!(e1 < 1.0 && e2 < 1.0);
    assert!(result.factor_cov.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_dfm_input_validation() {
    let x = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();

    // n_factors = 0.
    assert!(DFM::fit(&x, 0, 10, None).is_err());

    // n_factors >= n_series.
    assert!(DFM::fit(&x, 3, 10, None).is_err());

    // Too few observations.
    let short = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
    assert!(DFM::fit(&short, 1, 10, None).is_err());
}
