use greeners_timeseries::statespace::LocalLevel;

/// Local-level model on a simulated random walk plus noise.
/// The estimated sigma_obs should be positive and dominate sigma_state
/// for a series with little drift.
#[test]
fn test_local_level_mle() {
    let n = 100;
    let mut y = Vec::with_capacity(n);
    let mut mu = 0.0;
    for t in 0..n {
        mu += 0.01; // small drift
        let obs = mu + ((t * 7) as f64 % 5.0) / 5.0 - 0.5; // moderate noise
        y.push(obs);
    }

    let result = LocalLevel::fit(&y).unwrap();
    assert!(result.sigma_obs > 0.0);
    assert!(result.sigma_state > 0.0);
    assert!(result.log_likelihood.is_finite());
    assert_eq!(result.filtered_states.len(), n);
    assert_eq!(result.smoothed_states.len(), n);
}
