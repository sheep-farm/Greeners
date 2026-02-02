use greeners::{EGARCH, GARCH, GJRGARCH};
use ndarray::Array1;

/// Generate synthetic GARCH(1,1) data for testing
fn generate_garch_data(n: usize) -> Array1<f64> {
    // Deterministic pseudo-random using simple LCG
    let mut state: u64 = 42;
    let mut next_normal = || -> f64 {
        // Box-Muller from LCG
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-10);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let omega: f64 = 0.05;
    let alpha: f64 = 0.1;
    let beta: f64 = 0.85;
    let mu: f64 = 0.01;

    let mut y = vec![0.0; n];
    let mut h = vec![omega / (1.0 - alpha - beta); n];

    for t in 1..n {
        h[t] = omega + alpha * (y[t - 1] - mu).powi(2) + beta * h[t - 1];
        y[t] = mu + h[t].sqrt() * next_normal();
    }
    Array1::from_vec(y)
}

#[test]
fn test_garch_11_normal() {
    let y = generate_garch_data(500);
    let result = GARCH::fit(&y, 1, 1).unwrap();

    assert!(result.converged);
    assert_eq!(result.n_obs, 500);
    assert_eq!(result.p, 1);
    assert_eq!(result.q, 1);
    assert_eq!(result.model_type, greeners::GarchModelType::GARCH);
    assert_eq!(result.dist, greeners::GarchDist::Normal);
    // Should have 4 params: mu, omega, alpha, beta
    assert_eq!(result.params.len(), 4);
    // omega should be positive
    assert!(result.params[1] > 0.0);
    // alpha + beta < 1
    assert!(result.params[2] + result.params[3] < 1.0);
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_garch_11_student_t() {
    let y = generate_garch_data(500);
    let result = GARCH::fit_t(&y, 1, 1).unwrap();

    assert!(result.converged);
    assert_eq!(result.params.len(), 5); // mu, omega, alpha, beta, nu
    assert_eq!(result.dist, greeners::GarchDist::StudentT);
    // nu > 2
    assert!(result.params[4] > 2.0);
    assert!(result.log_likelihood.is_finite());
}

#[test]
fn test_arch_1() {
    // ARCH(1) = GARCH(0,1)
    let y = generate_garch_data(300);
    let result = GARCH::fit(&y, 0, 1).unwrap();

    assert_eq!(result.p, 0);
    assert_eq!(result.q, 1);
    assert_eq!(result.params.len(), 3); // mu, omega, alpha
    assert!(result.params[1] > 0.0); // omega
    assert!(result.params[2] >= 0.0); // alpha
}

#[test]
fn test_egarch() {
    let y = generate_garch_data(500);
    let result = EGARCH::fit(&y, 1, 1).unwrap();

    assert!(result.converged);
    assert_eq!(result.model_type, greeners::GarchModelType::EGARCH);
    // params: mu, omega, alpha, gamma, beta
    assert_eq!(result.params.len(), 5);
    assert!(result.log_likelihood.is_finite());
    assert_eq!(result.conditional_variance.len(), 500);
    // All conditional variances should be positive
    assert!(result.conditional_variance.iter().all(|v| *v > 0.0));
}

#[test]
fn test_gjr_garch() {
    let y = generate_garch_data(500);
    let result = GJRGARCH::fit(&y, 1, 1).unwrap();

    assert!(result.converged);
    assert_eq!(result.model_type, greeners::GarchModelType::GJRGARCH);
    // params: mu, omega, alpha, beta, gamma
    assert_eq!(result.params.len(), 5);
    assert!(result.params[1] > 0.0); // omega
    assert!(result.params[2] >= 0.0); // alpha
    assert!(result.params[3] >= 0.0); // beta
    assert!(result.params[4] >= 0.0); // gamma
    assert!(result.log_likelihood.is_finite());
}

#[test]
fn test_forecast() {
    let y = generate_garch_data(500);
    let result = GARCH::fit(&y, 1, 1).unwrap();

    let fc = result.forecast(5);
    assert_eq!(fc.len(), 5);
    // All forecasts should be positive
    assert!(fc.iter().all(|v| *v > 0.0));

    let vol = result.forecast_volatility(5);
    assert_eq!(vol.len(), 5);
    // Volatility should be sqrt of variance
    for i in 0..5 {
        assert!((vol[i] - fc[i].sqrt()).abs() < 1e-10);
    }
}

#[test]
fn test_standardized_residuals() {
    let y = generate_garch_data(500);
    let result = GARCH::fit(&y, 1, 1).unwrap();

    assert_eq!(result.standardized_residuals.len(), 500);
    // Standardized residuals should have variance roughly 1
    let mean_z: f64 = result.standardized_residuals.iter().sum::<f64>() / 500.0;
    let var_z: f64 = result
        .standardized_residuals
        .iter()
        .map(|z| (z - mean_z).powi(2))
        .sum::<f64>()
        / 500.0;
    // Should be roughly 1, allow wide tolerance for finite sample
    assert!(var_z > 0.3 && var_z < 3.0);
}

#[test]
fn test_display() {
    let y = generate_garch_data(200);
    let result = GARCH::fit(&y, 1, 1).unwrap();
    let display = format!("{}", result);

    assert!(display.contains("GARCH(1,1)"));
    assert!(display.contains("Normal"));
    assert!(display.contains("mu"));
    assert!(display.contains("omega"));
    assert!(display.contains("alpha[1]"));
    assert!(display.contains("beta[1]"));
    assert!(display.contains("Log-Likelihood"));
    assert!(display.contains("AIC"));
    assert!(display.contains("BIC"));
}
