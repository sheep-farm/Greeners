use greeners_timeseries::mstl::MSTL;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn test_mstl_runs_and_reconstructs_series() {
    let n = 80;
    let mut rng = StdRng::seed_from_u64(19001);
    let noise = Normal::new(0.0, 0.2).unwrap();

    // Build a series with linear trend, two seasonal components, and noise.
    let y = Array1::from(
        (0..n)
            .map(|i| {
                let trend = 0.05 * i as f64;
                let s1 = [1.0, 0.5, -0.5, -1.0][i % 4];
                let s2 = (2.0 * std::f64::consts::PI * (i as f64) / 8.0).sin();
                trend + s1 + s2 + noise.sample(&mut rng)
            })
            .collect::<Vec<_>>(),
    );

    let result = MSTL::fit(&y, &[4, 8]).unwrap();

    assert_eq!(result.n_obs, n);
    assert_eq!(result.periods, vec![4, 8]);
    assert_eq!(result.trend.len(), n);
    assert_eq!(result.resid.len(), n);
    assert_eq!(result.seasonal.len(), 2);
    for s in &result.seasonal {
        assert_eq!(s.len(), n);
        assert!(s.iter().all(|&v| v.is_finite()));
    }
    assert!(result.trend.iter().all(|&v| v.is_finite()));
    assert!(result.resid.iter().all(|&v| v.is_finite()));

    // Reconstruction equals the original (up to floating point noise).
    let recon = result.observed();
    assert_eq!(recon.len(), n);
    let max_diff: f64 = (0..n).map(|i| (recon[i] - y[i]).abs()).fold(0.0, f64::max);
    assert!(max_diff < 1e-8);
}

#[test]
fn test_mstl_trend_is_smooth_and_seasonals_oscillate() {
    let n = 80;
    let mut rng = StdRng::seed_from_u64(19002);
    let noise = Normal::new(0.0, 0.1).unwrap();

    let y = Array1::from(
        (0..n)
            .map(|i| {
                let trend = 0.1 * i as f64;
                let seasonal = [2.0, 1.0, -1.0, -2.0][i % 4];
                trend + seasonal + noise.sample(&mut rng)
            })
            .collect::<Vec<_>>(),
    );

    let result = MSTL::fit(&y, &[4]).unwrap();

    // Trend should be roughly monotonically increasing (or at least smooth).
    let diffs: Vec<f64> = (1..n)
        .map(|i| result.trend[i] - result.trend[i - 1])
        .collect();
    let sum_pos = diffs.iter().filter(|&&d| d > 0.0).count();
    assert!(sum_pos > n / 4);

    // The seasonal component should have period-4 pattern.
    let seasonal_std = result.seasonal[0]
        .iter()
        .map(|&v| v * v)
        .sum::<f64>()
        .sqrt()
        / (n as f64).sqrt();
    assert!(seasonal_std > 0.5);
}

#[test]
fn test_mstl_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);

    // Too short.
    assert!(MSTL::fit(&y, &[2]).is_err());

    // Empty periods.
    assert!(MSTL::fit(&Array1::from(vec![1.0; 10]), &[]).is_err());

    // Period < 2.
    assert!(MSTL::fit(&Array1::from(vec![1.0; 10]), &[1]).is_err());

    // Period exceeds series length.
    assert!(MSTL::fit(&Array1::from(vec![1.0; 10]), &[12]).is_err());
}
