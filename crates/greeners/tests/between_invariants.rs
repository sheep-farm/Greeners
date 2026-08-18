use greeners::{BetweenEstimator, OLS};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Between estimator recovers the between slope when all variation in X is
/// cross-sectional.
#[test]
fn test_between_recovery() {
    let n_entities = 15;
    let t = 5;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(5001);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut entity_ids = Vec::with_capacity(n);
    for e in 0..n_entities {
        let x_b = e as f64 / 2.0;
        let alpha = noise.sample(&mut rng);
        for _ in 0..t {
            entity_ids.push(e as i64);
            x_vec.push(1.0);
            x_vec.push(x_b);
            y_vec.push(1.0 + 2.0 * x_b + alpha + noise.sample(&mut rng));
        }
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let ids = Array1::from_vec(entity_ids);

    let result = BetweenEstimator::fit(&y, &x, &ids).unwrap();
    approx_zero((result.params[0] - 1.0).abs(), 0.3);
    approx_zero((result.params[1] - 2.0).abs(), 0.15);
    assert_eq!(result.n_entities, n_entities);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);

    // OLS on the entity-level means should match the between estimator.
    let mut y_means = Vec::with_capacity(n_entities);
    let mut x_means = Vec::with_capacity(n_entities * 2);
    for e in 0..n_entities {
        let start = e * t;
        let end = start + t;
        let y_bar = y.slice(ndarray::s![start..end]).mean().unwrap();
        let x0_bar = 1.0;
        let x1_bar = x.slice(ndarray::s![start..end, 1]).mean().unwrap();
        y_means.push(y_bar);
        x_means.push(x0_bar);
        x_means.push(x1_bar);
    }
    let x_b = Array2::from_shape_vec((n_entities, 2), x_means).unwrap();
    let y_b = Array1::from_vec(y_means);
    let ols = OLS::fit(&y_b, &x_b, greeners::CovarianceType::NonRobust).unwrap();
    for j in 0..2 {
        approx_zero((result.params[j] - ols.params[j]).abs(), 1e-10);
    }
}

/// Input validation.
#[test]
fn test_between_input_validation() {
    let y = Array1::from_vec(vec![1.0; 10]);
    let x = Array2::from_shape_vec((10, 1), vec![1.0; 10]).unwrap();
    let ids_short = Array1::from_vec(vec![0i64; 5]);
    assert!(BetweenEstimator::fit(&y, &x, &ids_short).is_err());
}
