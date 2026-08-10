use greeners::{CovarianceType, Tobit, OLS};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Without censoring, Tobit MLE should coincide with OLS.
#[test]
fn test_tobit_no_censoring_equals_ols() {
    let n = 80;
    let mut rng = StdRng::seed_from_u64(888);
    let norm = Normal::new(0.0, 0.5).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x2 = i as f64 / 20.0;
        x_vec.push(1.0);
        x_vec.push(x2);
        y_vec.push(1.0 + 2.0 * x2 + norm.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let tobit = Tobit::fit(&y, &x, -100.0, None).unwrap(); // ll below all y

    assert_eq!(tobit.n_censored, 0);
    for j in 0..ols.params.len() {
        approx_zero((tobit.params[j] - ols.params[j]).abs(), 1e-3);
    }
    assert!(tobit.sigma > 0.0);
}

/// A model with all observations censored must not return a normal result.
/// (The current implementation panics on empty uncensored subsample.)
#[test]
#[should_panic]
fn test_tobit_all_censored_fails() {
    let y = Array1::from(vec![0.0; 10]);
    let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
    let _ = Tobit::fit(&y, &x, 0.0, None).is_ok();
}

/// Tobit recovers the true latent coefficients when censoring is present.
#[test]
fn test_tobit_censored_recovery() {
    let n = 300;
    let mut rng = StdRng::seed_from_u64(222);
    let norm = Normal::new(0.0, 1.0).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut n_censored = 0;
    for i in 0..n {
        let x2 = i as f64 / 50.0;
        x_vec.push(1.0);
        x_vec.push(x2);
        let y_star = 1.0 + 2.0 * x2 + norm.sample(&mut rng);
        let y_cens = y_star.max(0.0);
        if y_cens == 0.0 {
            n_censored += 1;
        }
        y_vec.push(y_cens);
    }
    assert!(n_censored > 0, "expected some censoring");
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = Tobit::fit(&y, &x, 0.0, None).unwrap();
    assert!(result.n_censored > 0);
    approx_zero((result.params[0] - 1.0).abs(), 0.4);
    approx_zero((result.params[1] - 2.0).abs(), 0.2);
    assert!(result.sigma > 0.0);
}

/// Input validation.
#[test]
fn test_tobit_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0; 3]).unwrap();
    assert!(Tobit::fit(&y, &x, 0.0, None).is_err());

    let y_bad = Array1::from(vec![1.0, f64::NAN, 3.0, 4.0]);
    let x_good = Array2::from_shape_vec((4, 1), vec![1.0; 4]).unwrap();
    assert!(Tobit::fit(&y_bad, &x_good, 0.0, None).is_err());
}
