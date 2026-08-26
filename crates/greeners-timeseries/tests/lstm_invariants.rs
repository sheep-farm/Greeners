use greeners_timeseries::lstm::LSTM;
use ndarray::Array1;

/// LSTM returns fitted and forecast series of the expected lengths.
#[test]
fn test_lstm_invariants() {
    let n = 30;
    let y: Array1<f64> = (0..n)
        .map(|i| 1.0 + 0.5 * (i as f64) + (i as f64).sin())
        .collect();
    let n_fc = 4;

    let r = LSTM::fit(&y, Some(8), Some(5), Some(0.01), Some(50), Some(n_fc)).unwrap();

    assert_eq!(r.n_obs, n);
    assert_eq!(r.fitted.len(), n);
    assert_eq!(r.forecast.len(), n_fc);
    assert!(r.fitted.iter().all(|v| v.is_finite()));
    assert!(r.forecast.iter().all(|v| v.is_finite()));
    assert!(r.mse.is_finite());
    assert!(r.r_squared.is_finite());
    assert!(r.final_hidden.is_finite());
    assert!(r.final_cell.is_finite());
    assert_eq!(r.n_hidden, 8);
    assert_eq!(r.seq_len, 5);
    assert!(r.learning_rate > 0.0);
    assert!(r.n_epochs > 0);
}

/// Forecast length and n_hidden follow the defaults and requested values.
#[test]
fn test_lstm_defaults_invariants() {
    let n = 25;
    let y: Array1<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();

    let r = LSTM::fit(&y, None, None, None, None, Some(3)).unwrap();

    assert_eq!(r.n_obs, n);
    assert_eq!(r.forecast.len(), 3);
    assert!(r.n_hidden >= 1);
    assert!(r.seq_len >= 1);
}

/// Input validation rejects short series and zero variance.
#[test]
fn test_lstm_input_validation() {
    let short = Array1::from_vec(vec![1.0; 10]);
    assert!(LSTM::fit(&short, None, None, None, None, None).is_err());

    let zero_var = Array1::from_vec(vec![5.0; 30]);
    assert!(LSTM::fit(&zero_var, None, None, None, None, None).is_err());
}
