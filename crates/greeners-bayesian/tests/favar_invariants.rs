use greeners_bayesian::favar::FAVAR;
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn generate_favar_data(
    t: usize,
    n_series: usize,
    n_observed: usize,
    noise_sd: f64,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, noise_sd).unwrap();

    // Single common factor following a stationary AR(1).
    let mut f = 0.0;
    let mut factor = Vec::with_capacity(t);
    for _ in 0..t {
        f = 0.7 * f + noise.sample(&mut rng);
        factor.push(f);
    }

    // Generate observed panel with loadings on the factor.
    let mut x = Array2::zeros((t, n_series));
    for i in 0..t {
        for j in 0..n_series {
            x[[i, j]] = (j as f64 + 1.0) * 0.2 * factor[i] + noise.sample(&mut rng);
        }
    }

    // Observed policy variable is partly driven by the factor.
    let mut observed = Array2::zeros((t, n_observed));
    for i in 0..t {
        for j in 0..n_observed {
            observed[[i, j]] = 0.5 * factor[i] + noise.sample(&mut rng);
        }
    }

    (x, observed)
}

#[test]
fn test_favar_runs_and_produces_finite_output() {
    let (x, observed) = generate_favar_data(80, 5, 1, 0.5, 9001);

    let result = FAVAR::fit(&x, &observed, 1, 1, 5, None, None).unwrap();

    assert_eq!(result.n_series, 5);
    assert_eq!(result.n_observed, 1);
    assert_eq!(result.n_factors, 1);
    assert_eq!(result.lags, 1);
    assert_eq!(result.factors.shape(), &[80, 1]);
    assert_eq!(result.loadings.shape(), &[5, 1]);
    assert!(result.loadings.iter().all(|&v| v.is_finite()));
    assert!(result.factors.iter().all(|&v| v.is_finite()));
    assert!(result.var_coeffs.iter().all(|&v| v.is_finite()));
    assert!(result.var_sigma.iter().all(|&v| v.is_finite()));
    assert_eq!(result.irf.shape(), &[5, 2, 2]);
    assert!(result.irf.iter().all(|&v| v.is_finite()));
    assert!(result.total_variance_explained > 0.0 && result.total_variance_explained <= 1.0);
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_favar_factor_recovery() {
    let (x, observed) = generate_favar_data(100, 5, 1, 0.4, 9002);

    let result = FAVAR::fit(
        &x,
        &observed,
        1,
        1,
        0,
        Some(vec!["F".into()]),
        Some(vec!["R".into()]),
    )
    .unwrap();

    // All loadings should have the same sign because all series load positively.
    let first_sign = result.loadings[[0, 0]].signum();
    assert!(first_sign != 0.0);
    assert!(result
        .loadings
        .iter()
        .all(|&v| v.signum() == first_sign || v == 0.0));

    // The policy variable should load positively on the factor.
    // Coefficient layout: var_coeffs[[1, 1]] is the effect of the factor on R.
    let k = result.n_factors + result.n_observed;
    assert_eq!(result.var_coeffs.shape(), &[1 + k, k]);
    assert!(result.var_coeffs[[1, 1]].is_finite());
}

#[test]
fn test_favar_input_validation() {
    let x = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();
    let observed = Array2::from_shape_vec((9, 1), vec![1.0; 9]).unwrap();

    // Mismatched row count.
    assert!(FAVAR::fit(&x, &observed, 1, 1, 0, None, None).is_err());

    // n_factors = 0 or lags = 0.
    assert!(FAVAR::fit(&x, &x, 0, 1, 0, None, None).is_err());
    assert!(FAVAR::fit(&x, &x, 1, 0, 0, None, None).is_err());

    // Too few observations.
    let small = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
    assert!(FAVAR::fit(&small, &small, 1, 1, 0, None, None).is_err());
}
