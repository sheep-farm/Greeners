use greeners::{CovarianceType, RobustNorm, OLS, RLM};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// RLM with LeastSquares norm is equivalent to OLS.
#[test]
fn test_rlm_least_squares_equals_ols() {
    let n = 30;
    let x = Array2::from_shape_vec(
        (n, 2),
        (0..n).flat_map(|i| vec![1.0, i as f64 / 10.0]).collect(),
    )
    .unwrap();
    let y = x.column(0).to_owned() * 1.0 + x.column(1).to_owned() * 2.0;

    let rlm = RLM::fit(&y, &x, &RobustNorm::LeastSquares, CovarianceType::NonRobust).unwrap();
    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    for j in 0..ols.params.len() {
        approx_zero((rlm.params[j] - ols.params[j]).abs(), 1e-6);
    }
}

/// Huber RLM is close to OLS on clean normal data.
#[test]
fn test_rlm_huber_clean_data() {
    let n = 80;
    let mut rng = StdRng::seed_from_u64(987);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x2 = i as f64 / 20.0;
        x_vec.push(1.0);
        x_vec.push(x2);
        y_vec.push(1.0 + 2.0 * x2 + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let rlm = RLM::fit(&y, &x, &RobustNorm::Huber(1.345), CovarianceType::NonRobust).unwrap();
    approx_zero((rlm.params[0] - 1.0).abs(), 0.2);
    approx_zero((rlm.params[1] - 2.0).abs(), 0.15);
    assert!(rlm.scale > 0.0);
}

/// Huber RLM is less sensitive to an outlier than OLS.
#[test]
fn test_rlm_huber_resists_outlier() {
    let n = 40;
    let mut rng = StdRng::seed_from_u64(654);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x2 = i as f64 / 10.0;
        x_vec.push(1.0);
        x_vec.push(x2);
        y_vec.push(1.0 + 2.0 * x2 + noise.sample(&mut rng));
    }
    // Add a large outlier at the last observation.
    y_vec[n - 1] += 50.0;

    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let rlm = RLM::fit(&y, &x, &RobustNorm::Huber(1.345), CovarianceType::NonRobust).unwrap();

    // RLM slope should be much closer to 2.0 than OLS slope.
    let ols_dev = (ols.params[1] - 2.0).abs();
    let rlm_dev = (rlm.params[1] - 2.0).abs();
    assert!(
        rlm_dev < ols_dev,
        "RLM dev {} not smaller than OLS dev {}",
        rlm_dev,
        ols_dev
    );
}

/// Input validation.
#[test]
fn test_rlm_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((2, 1), vec![1.0; 2]).unwrap();
    assert!(RLM::fit(&y, &x, &RobustNorm::Huber(1.345), CovarianceType::NonRobust).is_err());
}
