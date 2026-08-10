use greeners::{CanCorr, FactorAnalysis, Rotation, MANOVA};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn make_factor_data(seed: u64, n: usize, p: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let mut data = Vec::new();
    for i in 0..n {
        let f1 = (i as f64 * 0.1).sin();
        let f2 = (i as f64 * 0.07).cos();
        for j in 0..p {
            let loading = if j % 2 == 0 { 1.0 } else { 0.5 };
            let factor = if j < 2 { f1 } else { f2 };
            data.push(factor * loading + noise.sample(&mut rng));
        }
    }
    Array2::from_shape_vec((n, p), data).unwrap()
}

/// Factor analysis returns loadings and communalities with the expected shapes.
#[test]
fn test_factor_analysis_invariants() {
    let data = make_factor_data(12345, 50, 4);
    let n_factors = 2;

    let r = FactorAnalysis::fit(&data, n_factors, Rotation::None).unwrap();
    assert_eq!(r.n_obs, 50);
    assert_eq!(r.n_factors, n_factors);
    assert_eq!(r.loadings.shape(), [4, n_factors]);
    assert_eq!(r.communalities.len(), 4);
    assert_eq!(r.uniquenesses.len(), 4);
    assert_eq!(r.eigenvalues.len(), n_factors);
    assert!(r.loadings.iter().all(|v| v.is_finite()));
    assert!(r.communalities.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(r.uniquenesses.iter().all(|v| v.is_finite() && *v >= 0.0));
}

/// MANOVA and canonical correlation return finite multivariate test statistics.
#[test]
fn test_manova_cancorr_invariants() {
    let n = 60;
    let mut rng = StdRng::seed_from_u64(67890);
    let noise = Normal::new(0.0, 1.0).unwrap();

    // MANOVA: two dependent variables with three groups
    let mut y = Array2::zeros((n, 2));
    let mut groups = Vec::new();
    for i in 0..n {
        let g = i / 20;
        groups.push(g);
        let base = g as f64 * 2.0;
        y[[i, 0]] = base + noise.sample(&mut rng);
        y[[i, 1]] = base * 0.5 + noise.sample(&mut rng);
    }
    let groups = Array1::from_vec(groups);

    let manova = MANOVA::fit(&y, &groups).unwrap();
    assert_eq!(manova.n_obs, n);
    assert_eq!(manova.n_groups, 3);
    assert_eq!(manova.n_vars, 2);
    assert!(manova.wilks_lambda.is_finite());
    assert!(manova.p_values.iter().all(|&p| p >= 0.0 && p <= 1.0));

    // CanCorr: two blocks of two variables each, sharing a common factor
    let mut x = Array2::zeros((n, 2));
    let mut yy = Array2::zeros((n, 2));
    for i in 0..n {
        let common = (i as f64) * 0.1;
        x[[i, 0]] = common + noise.sample(&mut rng);
        x[[i, 1]] = common * 0.7 + noise.sample(&mut rng);
        yy[[i, 0]] = common + noise.sample(&mut rng);
        yy[[i, 1]] = common * 0.4 + noise.sample(&mut rng);
    }

    let cancorr = CanCorr::fit(&x, &yy).unwrap();
    assert_eq!(cancorr.n_obs, n);
    assert_eq!(cancorr.cancorr.len(), 2);
    assert_eq!(cancorr.x_weights.shape(), [2, 2]);
    assert_eq!(cancorr.y_weights.shape(), [2, 2]);
    assert!(cancorr.cancorr.iter().all(|&r| r >= 0.0 && r <= 1.0));
    assert!(cancorr.p_value >= 0.0 && cancorr.p_value <= 1.0);
}

/// Input validation catches insufficient observations or mismatched dimensions.
#[test]
fn test_multivariate_input_validation() {
    let data = make_factor_data(11111, 50, 4);
    assert!(FactorAnalysis::fit(
        &data.slice(ndarray::s![0..1, ..]).to_owned(),
        2,
        Rotation::None
    )
    .is_err());

    let y = Array2::from_shape_vec((6, 2), (1..=12).map(|v| v as f64).collect()).unwrap();
    let groups = Array1::from_vec(vec![0; 6]);
    assert!(MANOVA::fit(&y, &groups).is_err());

    let x = Array2::from_shape_vec((5, 2), (1..=10).map(|v| v as f64).collect()).unwrap();
    let yy = Array2::from_shape_vec((6, 2), (1..=12).map(|v| v as f64).collect()).unwrap();
    assert!(CanCorr::fit(&x, &yy).is_err());
}
