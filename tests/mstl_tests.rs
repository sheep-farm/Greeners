use greeners::{MSTLResult, MSTL};
use ndarray::Array1;
use std::f64::consts::PI;

#[test]
fn test_mstl_decomposition_reconstructs_original() {
    let n = 365;
    let y: Array1<f64> = Array1::from_vec(
        (0..n)
            .map(|t| {
                let t = t as f64;
                let trend = 0.01 * t;
                let seasonal_7 = (2.0 * PI * t / 7.0).sin();
                let seasonal_30 = 0.5 * (2.0 * PI * t / 30.0).sin();
                let noise = 0.05 * ((t * 1.3).sin() + (t * 0.7).cos()); // deterministic "noise"
                trend + seasonal_7 + seasonal_30 + noise
            })
            .collect(),
    );

    let result = MSTL::fit(&y, &[7, 30]).expect("MSTL fit should succeed");

    assert_eq!(result.n_obs, n);
    assert_eq!(result.periods.len(), 2);
    assert_eq!(result.periods, vec![7, 30]);
    assert_eq!(result.seasonal.len(), 2);

    // Reconstruction should match original
    let reconstructed = result.observed();
    let max_err: f64 = (&y - &reconstructed)
        .iter()
        .map(|x| x.abs())
        .fold(0.0, f64::max);
    assert!(
        max_err < 1e-10,
        "Reconstruction error too large: {}",
        max_err
    );
}

#[test]
fn test_mstl_seasonal_shapes() {
    let n = 365;
    let y: Array1<f64> = Array1::from_vec(
        (0..n)
            .map(|t| {
                let t = t as f64;
                0.01 * t + (2.0 * PI * t / 7.0).sin() + 0.5 * (2.0 * PI * t / 30.0).sin()
            })
            .collect(),
    );

    let result = MSTL::fit(&y, &[7, 30]).expect("MSTL fit should succeed");

    // Each seasonal component should have length n
    for s in &result.seasonal {
        assert_eq!(s.len(), n);
    }
    assert_eq!(result.trend.len(), n);
    assert_eq!(result.resid.len(), n);
}

#[test]
fn test_mstl_display() {
    let n = 100;
    let y: Array1<f64> = Array1::from_vec(
        (0..n)
            .map(|t| {
                let t = t as f64;
                t * 0.01 + (2.0 * PI * t / 7.0).sin()
            })
            .collect(),
    );

    let result = MSTL::fit(&y, &[7]).expect("MSTL fit should succeed");
    let display = format!("{}", result);
    assert!(display.contains("MSTL Decomposition"));
    assert!(display.contains("Observations:"));
}

#[test]
fn test_mstl_periods_sorted() {
    let n = 200;
    let y: Array1<f64> = Array1::from_vec(
        (0..n)
            .map(|t| {
                let t = t as f64;
                (2.0 * PI * t / 10.0).sin() + 0.5 * (2.0 * PI * t / 5.0).sin()
            })
            .collect(),
    );

    // Pass periods in descending order; they should be sorted ascending internally
    let result = MSTL::fit(&y, &[10, 5]).expect("MSTL fit should succeed");
    assert_eq!(result.periods, vec![5, 10]);
}

#[test]
fn test_mstl_error_on_invalid_period() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(MSTL::fit(&y, &[1]).is_err()); // period < 2
    assert!(MSTL::fit(&y, &[]).is_err()); // no periods
    assert!(MSTL::fit(&y, &[10]).is_err()); // period > n
}

#[test]
fn test_mstl_short_series_error() {
    let y = Array1::from_vec(vec![1.0, 2.0]);
    assert!(MSTL::fit(&y, &[2]).is_err());
}
