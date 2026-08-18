use greeners::Spatial;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_spatial_data(seed: u64, n: usize) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    // Row-standardized nearest-neighbour weights (each unit connected to the next)
    let mut w = Array2::zeros((n, n));
    for i in 0..n {
        w[(i, (i + 1) % n)] = 1.0;
    }

    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) / n as f64;
        x.push(1.0);
        x.push(x1);
        y.push(1.0 + 0.5 * x1 + noise.sample(&mut rng));
    }
    (
        Array1::from_vec(y),
        Array2::from_shape_vec((n, 2), x).unwrap(),
        w,
    )
}

/// SAR and SEM fits return finite coefficients, spatial parameters and diagnostics.
#[test]
fn test_spatial_sar_sem_invariants() {
    let (y, x, w) = make_spatial_data(12345, 40);
    let names = Some(vec!["const".into(), "x1".into()]);

    let sar = Spatial::fit_sar(&y, &x, &w, names.clone()).unwrap();
    assert_eq!(sar.n_obs, 40);
    assert_eq!(sar.beta.len(), 2);
    assert!(sar.beta.iter().all(|v| v.is_finite()));
    assert!(sar.spatial_param.is_finite());
    assert!(sar.log_likelihood.is_finite());
    assert!(sar.r_squared.is_finite());
    assert_eq!(sar.model_type, "sar");

    let sem = Spatial::fit_sem(&y, &x, &w, names).unwrap();
    assert_eq!(sem.n_obs, 40);
    assert_eq!(sem.beta.len(), 2);
    assert!(sem.beta.iter().all(|v| v.is_finite()));
    assert!(sem.spatial_param.is_finite());
    assert!(sem.log_likelihood.is_finite());
    assert!(sem.r_squared.is_finite());
    assert_eq!(sem.model_type, "sem");
}

/// Input validation rejects mismatched dimensions for the weights or design matrix.
#[test]
fn test_spatial_input_validation() {
    let (y, x, w) = make_spatial_data(11111, 40);

    let y_short = y.slice(ndarray::s![0..30]).to_owned();
    assert!(Spatial::fit_sar(&y_short, &x, &w, None).is_err());

    let w_bad = Array2::zeros((30, 30));
    assert!(Spatial::fit_sar(&y, &x, &w_bad, None).is_err());

    let x_bad = x.slice(ndarray::s![0..30, ..]).to_owned();
    assert!(Spatial::fit_sem(&y, &x_bad, &w, None).is_err());
}

/// A pure-noise dependent variable still yields a bounded spatial parameter and R-squared.
#[test]
fn test_spatial_pure_noise() {
    let mut rng = StdRng::seed_from_u64(22222);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let n = 40;
    let y = Array1::from_vec((0..n).map(|_| noise.sample(&mut rng)).collect());
    let x = Array2::ones((n, 1));
    let mut w = Array2::zeros((n, n));
    for i in 0..n {
        w[(i, (i + 1) % n)] = 1.0;
    }

    let sar = Spatial::fit_sar(&y, &x, &w, None).unwrap();
    assert!(sar.spatial_param.is_finite());
    assert!(sar.r_squared.is_finite());
    assert!(sar.log_likelihood.is_finite());

    let sem = Spatial::fit_sem(&y, &x, &w, None).unwrap();
    assert!(sem.spatial_param.is_finite());
    assert!(sem.r_squared.is_finite());
    assert!(sem.log_likelihood.is_finite());
}
