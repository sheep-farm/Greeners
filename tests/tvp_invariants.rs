use greeners::TVP;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn make_tvp_data(seed: u64, n: usize) -> (Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let x0 = 1.0;
        let x1 = t / n as f64;
        x.push(x0);
        x.push(x1);
        let beta0 = 1.0;
        let beta1 = 2.0 + 0.5 * (t / n as f64);
        y.push(beta0 * x0 + beta1 * x1 + noise.sample(&mut rng));
    }

    let y_arr = Array1::from_vec(y);
    let x_arr = Array2::from_shape_vec((n, 2), x).unwrap();
    (y_arr, x_arr)
}

/// TVP fit returns smoothed coefficients with the expected shape and finite statistics.
#[test]
fn test_tvp_invariants() {
    let (y, x) = make_tvp_data(12345, 80);
    let result = TVP::fit(&y, &x, Some(vec!["const".into(), "x1".into()])).unwrap();

    assert_eq!(result.n_obs, 80);
    assert_eq!(result.k(), 2);
    assert_eq!(result.beta_smoothed.shape(), &[80, 2]);
    assert_eq!(result.beta_se.shape(), &[80, 2]);
    assert!(result.beta_smoothed.iter().all(|v| v.is_finite()));
    assert!(result.beta_se.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.sigma_epsilon.is_finite() && result.sigma_epsilon > 0.0);
    assert!(result.sigma_eta.is_finite() && result.sigma_eta >= 0.0);
    assert!(result.log_likelihood.is_finite());
}

/// TVP rejects a mismatch between y length and the number of x rows.
#[test]
fn test_tvp_input_validation() {
    let (y, x) = make_tvp_data(11111, 80);
    let y_short = y.slice(ndarray::s![0..70]).to_owned();
    let result = TVP::fit(&y_short, &x, None);
    assert!(result.is_err());

    let x_short = x.slice(ndarray::s![0..10, ..]).to_owned();
    let result2 = TVP::fit(&y, &x_short, None);
    assert!(result2.is_err());
}

/// The smoothed coefficient path is finite and retains the correct shape.
#[test]
fn test_tvp_smoothed_shape() {
    let (y, x) = make_tvp_data(22222, 100);
    let result = TVP::fit(&y, &x, None).unwrap();

    let mid = result.n_obs / 2;
    let first = result.beta_smoothed[(0, 0)];
    let middle = result.beta_smoothed[(mid, 0)];
    let last = result.beta_smoothed[(result.n_obs - 1, 0)];
    assert!(first.is_finite());
    assert!(middle.is_finite());
    assert!(last.is_finite());
    assert_eq!(result.beta_smoothed.nrows(), result.n_obs);
    assert_eq!(result.beta_smoothed.ncols(), result.k());
}
