use greeners::{EGARCH, GARCH, GJRGARCH};
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

fn generate_garch11(
    n: usize,
    mu: f64,
    omega: f64,
    alpha: f64,
    beta: f64,
    seed: u64,
) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut y = vec![0.0; n];
    let mut h = omega / (1.0 - alpha - beta); // unconditional variance
    for t in 0..n {
        let z = norm.sample(&mut rng);
        let eps = z * h.sqrt();
        y[t] = mu + eps;
        h = omega + alpha * eps * eps + beta * h;
    }
    Array1::from(y)
}

/// Manually recompute GARCH(1,1) conditional variance from the returned
/// parameters and verify that the result matches the model output.
#[test]
fn test_garch_conditional_variance_formula() {
    let y = generate_garch11(120, 0.0, 0.1, 0.1, 0.8, 12345);
    let result = GARCH::fit(&y, 1, 1).unwrap();

    let mu = result.params[0];
    let omega = result.params[1];
    let alpha = result.params[2];
    let beta = result.params[3];

    let eps: Vec<f64> = y.iter().map(|v| v - mu).collect();
    let var_init = eps.iter().map(|e| e * e).sum::<f64>() / eps.len() as f64;
    let mut h = vec![var_init; eps.len()];
    for t in 1..eps.len() {
        h[t] = (omega + alpha * eps[t - 1] * eps[t - 1] + beta * h[t - 1]).max(1e-10);
    }

    for t in 0..y.len() {
        approx_zero((result.conditional_variance[t] - h[t]).abs(), 1e-8);
        approx_zero((result.residuals[t] - eps[t]).abs(), 1e-8);
        approx_zero(
            (result.standardized_residuals[t]
                - result.residuals[t] / result.conditional_variance[t].sqrt())
            .abs(),
            1e-8,
        );
    }
}

/// A stationary GARCH(1,1) must satisfy persistence < 1.
#[test]
fn test_garch_stationarity() {
    let y = generate_garch11(120, 0.0, 0.1, 0.1, 0.8, 42);
    let result = GARCH::fit(&y, 1, 1).unwrap();

    let persistence = result.params[2] + result.params[3];
    assert!(
        persistence < 1.0,
        "persistence = {} is not < 1",
        persistence
    );
    assert!(
        persistence > 0.0,
        "persistence = {} is not > 0",
        persistence
    );
}

/// Multi-step GARCH(1,1) forecast converges to the unconditional variance
/// omega / (1 - alpha - beta).
#[test]
fn test_garch_forecast_converges_to_unconditional_variance() {
    let y = generate_garch11(120, 0.0, 0.1, 0.1, 0.8, 2025);
    let result = GARCH::fit(&y, 1, 1).unwrap();

    let omega = result.params[1];
    let alpha = result.params[2];
    let beta = result.params[3];
    let uncond = omega / (1.0 - alpha - beta);

    let forecasts = result.forecast(100);
    let last = forecasts[forecasts.len() - 1];
    approx_zero((last - uncond).abs(), 1e-3 * uncond.max(1.0));
}

/// Input validation for GARCH.
#[test]
fn test_garch_input_validation() {
    let y = Array1::from(vec![0.0; 5]);
    assert!(GARCH::fit(&y, 1, 1).is_err()); // too short

    let y = Array1::from(vec![0.0; 20]);
    assert!(GARCH::fit(&y, 1, 0).is_err()); // q must be >= 1
}

/// EGARCH and GJR-GARCH return positive conditional variances and
/// standardized residuals.
#[test]
fn test_egarch_gjrgarch_output_sanity() {
    let y = generate_garch11(80, 0.0, 0.1, 0.1, 0.8, 99);

    let egarch = EGARCH::fit(&y, 1, 1).unwrap();
    for &v in egarch.conditional_variance.iter() {
        assert!(v > 0.0, "EGARCH conditional variance must be positive");
    }

    let gjrgarch = GJRGARCH::fit(&y, 1, 1).unwrap();
    for &v in gjrgarch.conditional_variance.iter() {
        assert!(v > 0.0, "GJR-GARCH conditional variance must be positive");
    }
}
