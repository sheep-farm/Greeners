use greeners::MFVAR;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn generate_mfvar_data(
    t_low: usize,
    agg_ratio: usize,
    noise_sd: f64,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    let t_high = t_low * agg_ratio;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, noise_sd).unwrap();

    // High-frequency regressor.
    let mut x_high = Array2::zeros((t_high, 1));
    let mut prev = 0.0;
    for i in 0..t_high {
        x_high[[i, 0]] = 0.5 * prev + noise.sample(&mut rng);
        prev = x_high[[i, 0]];
    }

    // Aggregate to low frequency by simple average.
    let mut x_low = Array2::zeros((t_low, 1));
    for i in 0..t_low {
        let mut s = 0.0;
        for h in 0..agg_ratio {
            s += x_high[[i * agg_ratio + h, 0]];
        }
        x_low[[i, 0]] = s / agg_ratio as f64;
    }

    // Low-frequency outcome: AR(1) on its own lag plus the aggregated high-freq variable.
    let mut y_low = Array2::zeros((t_low, 1));
    for i in 0..t_low {
        let own_lag = if i == 0 { 0.0 } else { y_low[[i - 1, 0]] };
        y_low[[i, 0]] = 0.3 * own_lag + 0.7 * x_low[[i, 0]] + noise.sample(&mut rng);
    }

    (y_low, x_high)
}

#[test]
fn test_mfvar_runs_and_produces_finite_output() {
    let (y_low, y_high) = generate_mfvar_data(30, 3, 0.3, 1001);

    let result = MFVAR::fit(&y_low, &y_high, 3, 1, None, None).unwrap();

    assert_eq!(result.n_obs, y_low.nrows() - 1);
    assert_eq!(result.n_vars, 2); // y_low + aggregated x
    assert_eq!(result.lags, 1);
    assert_eq!(result.agg_ratio, 3);
    assert_eq!(result.coeffs.shape(), &[2, 2]);
    assert!(result.coeffs.iter().all(|&v| v.is_finite()));
    assert!(result.std_errors.iter().all(|&v| v.is_finite()));
    assert!(result.t_values.iter().all(|&v| v.is_finite()));
    assert!(result
        .p_values
        .iter()
        .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)));
    assert_eq!(result.aggregated.shape(), &[30, 1]);
    assert_eq!(result.midas_weights.len(), 3);
    assert!((result.midas_weights.sum() - 1.0).abs() < 1e-6);
    assert!(result.midas_weights.iter().all(|&v| v >= 0.0));
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_mfvar_recovers_positive_high_freq_effect() {
    let (y_low, y_high) = generate_mfvar_data(50, 3, 0.2, 1002);

    let result = MFVAR::fit(
        &y_low,
        &y_high,
        3,
        1,
        Some(vec!["y".into()]),
        Some(vec!["x_high".into()]),
    )
    .unwrap();

    // y equation: own lag and aggregated x. The high-freq effect should be positive.
    // coeffs layout: row 0 = y equation, col 0 = y.L1, col 1 = x.L1.
    assert!(result.coeffs[[0, 0]] > 0.0 && result.coeffs[[0, 0]] < 0.8);
    assert!(result.coeffs[[0, 1]] > 0.0 && result.coeffs[[0, 1]] < 1.2);
    assert!(result.coeffs[[1, 1]].abs() < 1.0);
}

#[test]
fn test_mfvar_input_validation() {
    let y_low = Array2::from_shape_vec((10, 1), vec![1.0; 10]).unwrap();
    let y_high = Array2::from_shape_vec((28, 1), vec![1.0; 28]).unwrap();

    // agg_ratio = 0.
    assert!(MFVAR::fit(&y_low, &y_high, 0, 1, None, None).is_err());

    // y_high too short for 3:1 aggregation.
    assert!(MFVAR::fit(&y_low, &y_high, 3, 1, None, None).is_err());

    // lags = 0.
    assert!(MFVAR::fit(
        &y_low,
        &(Array2::from_shape_vec((30, 1), vec![1.0; 30]).unwrap()),
        3,
        0,
        None,
        None
    )
    .is_err());
}
