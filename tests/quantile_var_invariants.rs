use greeners::QuantileVAR;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn generate_var1(
    c: &Array1<f64>,
    a: &Array2<f64>,
    y0: &Array1<f64>,
    t: usize,
    noise_sd: f64,
    seed: u64,
) -> Array2<f64> {
    let k = c.len();
    let mut data = Array2::zeros((t, k));
    data.row_mut(0).assign(y0);
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, noise_sd).unwrap();
    for i in 1..t {
        let y_prev = data.row(i - 1).to_owned();
        let y_t = c + a.dot(&y_prev);
        data.row_mut(i).assign(&y_t);
        for j in 0..k {
            data[[i, j]] += noise.sample(&mut rng);
        }
    }
    data
}

#[test]
fn test_quantile_var_runs_and_produces_finite_output() {
    let c = Array1::from(vec![1.0, 2.0]);
    let a = Array2::from_shape_vec((2, 2), vec![0.3, 0.2, 0.1, 0.4]).unwrap();
    let y0 = Array1::from(vec![0.0, 0.0]);
    let data = generate_var1(&c, &a, &y0, 80, 0.5, 8001);

    // n_boot = 0 keeps the test deterministic; standard errors are NaN but
    // the point estimates are still finite and useful to inspect.
    let result = QuantileVAR::fit(&data, 1, 0.5, 0, None).unwrap();

    assert_eq!(result.n_vars, 2);
    assert_eq!(result.lags, 1);
    assert_eq!(result.tau, 0.5);
    assert_eq!(result.coeffs.shape(), &[2, 3]); // intercept + 2 lags
    assert!(result.coeffs.iter().all(|&v| v.is_finite()));
    assert!(result
        .pseudo_r2
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    assert!(result.n_obs > 0);
    assert_eq!(result.var_names, vec!["y0", "y1"]);
}

#[test]
fn test_quantile_var_irf_shape_and_median_recovery() {
    let c = Array1::from(vec![0.0, 0.0]);
    let a = Array2::from_shape_vec((2, 2), vec![0.3, 0.1, 0.05, 0.4]).unwrap();
    let y0 = Array1::from(vec![0.0, 0.0]);
    let data = generate_var1(&c, &a, &y0, 120, 0.4, 8002);

    let result = QuantileVAR::fit(&data, 1, 0.5, 0, Some(vec!["a".into(), "b".into()])).unwrap();

    let irf = QuantileVAR::irf(&result, 5);
    assert_eq!(irf.shape(), &[5, 2, 2]);
    assert!(irf.iter().all(|&v| v.is_finite()));

    // At the median, own-lag coefficients should be positive and moderately sized.
    assert!(result.coeffs[[0, 1]] > 0.0 && result.coeffs[[0, 1]] < 0.7);
    assert!(result.coeffs[[1, 2]] > 0.0 && result.coeffs[[1, 2]] < 0.7);
    // Cross-lag coefficients should be small.
    assert!(result.coeffs[[0, 2]].abs() < 0.3);
    assert!(result.coeffs[[1, 1]].abs() < 0.3);
}

#[test]
fn test_quantile_var_input_validation() {
    let data = Array2::from_shape_vec((8, 2), vec![1.0; 16]).unwrap();

    // Invalid tau values.
    assert!(QuantileVAR::fit(&data, 1, 0.0, 0, None).is_err());
    assert!(QuantileVAR::fit(&data, 1, 1.0, 0, None).is_err());

    // Zero lags.
    assert!(QuantileVAR::fit(&data, 0, 0.5, 0, None).is_err());

    // Too few observations.
    let short = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
    assert!(QuantileVAR::fit(&short, 1, 0.5, 0, None).is_err());
}
