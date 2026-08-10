use greeners::TVAR;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn generate_tvar_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.25).unwrap();
    let k = 2;
    let mut data = Array2::zeros((n, k));

    // Threshold variable is the first series lag 1.
    for i in 1..n {
        let y_lag = data.row(i - 1).to_owned();
        let threshold = y_lag[0];
        let y_t = if threshold <= 0.0 {
            Array1::from(vec![
                0.4 * y_lag[0] - 0.2 * y_lag[1],
                0.1 * y_lag[0] + 0.3 * y_lag[1],
            ])
        } else {
            Array1::from(vec![
                -0.2 * y_lag[0] + 0.1 * y_lag[1],
                0.3 * y_lag[0] + 0.2 * y_lag[1],
            ])
        };
        data.row_mut(i).assign(&y_t);
        for j in 0..k {
            data[[i, j]] += noise.sample(&mut rng);
        }
    }

    // Threshold variable is y0 (used as q in TVAR fit).
    let q = data.column(0).to_owned();
    (data, q)
}

#[test]
fn test_tvar_runs_and_produces_finite_output() {
    let (data, q) = generate_tvar_data(250, 14001);

    let result = TVAR::fit(&data, &q, 1, 1, None).unwrap();

    assert_eq!(result.n_vars, 2);
    assert_eq!(result.lags, 1);
    assert!(result.delay >= 1 && result.delay <= 1);
    assert!(result.threshold.is_finite());
    assert!(result.n_low > 0);
    assert!(result.n_high > 0);
    assert_eq!(result.coeffs_low.shape(), &[2, 2]);
    assert_eq!(result.coeffs_high.shape(), &[2, 2]);
    assert_eq!(result.se_low.shape(), &[2, 2]);
    assert_eq!(result.se_high.shape(), &[2, 2]);
    assert_eq!(result.cov_low.shape(), &[2, 2]);
    assert_eq!(result.cov_high.shape(), &[2, 2]);
    assert!(result.coeffs_low.iter().all(|&v| v.is_finite()));
    assert!(result.coeffs_high.iter().all(|&v| v.is_finite()));
    assert!(result.rss >= 0.0);
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
    assert_eq!(result.n_low + result.n_high, data.nrows() - 1);
}

#[test]
fn test_tvar_regime_directions_are_reasonable() {
    let (data, q) = generate_tvar_data(300, 14002);

    let result = TVAR::fit(&data, &q, 1, 1, Some(vec!["y0".into(), "y1".into()])).unwrap();

    // Low regime: y0 own-lag coefficient should be positive.
    assert!(result.coeffs_low[[0, 0]] > 0.0 && result.coeffs_low[[0, 0]] < 0.8);
    // High regime: y0 own-lag coefficient should be negative.
    assert!(result.coeffs_high[[0, 0]] < 0.0 && result.coeffs_high[[0, 0]] > -0.6);
}

#[test]
fn test_tvar_input_validation() {
    let data = Array2::from_shape_vec((12, 2), vec![1.0; 24]).unwrap();
    let q = Array1::from(vec![1.0; 11]);

    // Mismatched q length.
    assert!(TVAR::fit(&data, &q, 1, 1, None).is_err());

    // lags = 0.
    let q_ok = Array1::from(vec![1.0; 12]);
    assert!(TVAR::fit(&data, &q_ok, 0, 1, None).is_err());

    // Too few observations.
    let short = Array2::from_shape_vec((6, 2), vec![1.0; 12]).unwrap();
    let q_short = Array1::from(vec![1.0; 6]);
    assert!(TVAR::fit(&short, &q_short, 1, 1, None).is_err());
}
