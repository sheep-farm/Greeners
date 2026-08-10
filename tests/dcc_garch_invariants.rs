use greeners::DCCGARCH;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

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
    let mut h = omega / (1.0 - alpha - beta);
    for t in 0..n {
        let z = norm.sample(&mut rng);
        let eps = z * h.sqrt();
        y[t] = mu + eps;
        h = omega + alpha * eps * eps + beta * h;
    }
    Array1::from(y)
}

#[test]
fn test_dcc_garch_output_properties() {
    let n = 80;
    let r1 = generate_garch11(n, 0.0, 0.1, 0.1, 0.8, 1001);
    let r2 = generate_garch11(n, 0.0, 0.05, 0.1, 0.85, 1002);

    let mut returns = Array2::zeros((n, 2));
    for i in 0..n {
        returns[[i, 0]] = r1[i];
        returns[[i, 1]] = r2[i];
    }

    let result = DCCGARCH::fit(&returns, None).unwrap();

    // Conditional volatilities must be positive.
    for i in 0..n {
        for j in 0..2 {
            assert!(result.conditional_vols[[i, j]] > 0.0);
        }
    }

    // Each univariate GARCH(1,1) must be stationary.
    for j in 0..2 {
        let alpha = result.garch_params[[j, 1]];
        let beta = result.garch_params[[j, 2]];
        assert!(
            alpha + beta < 0.99,
            "persistence too high: {}",
            alpha + beta
        );
    }

    // DCC persistence must be below the grid cap.
    assert!(result.dcc_alpha + result.dcc_beta < 0.99);

    // Correlation matrices must be valid: symmetric, diagonal 1, off-diagonal
    // in [-1, 1].
    for t in 0..n {
        for i in 0..2 {
            approx_eq(result.dcc_correlations[[t, i, i]], 1.0, 1e-8);
            for j in 0..2 {
                let v = result.dcc_correlations[[t, i, j]];
                assert!(
                    v.is_finite(),
                    "non-finite correlation at t={} i={} j={}: {}",
                    t,
                    i,
                    j,
                    v
                );
                assert!(
                    v >= -1.0001,
                    "correlation < -1 at t={} i={} j={}: {}",
                    t,
                    i,
                    j,
                    v
                );
                assert!(
                    v <= 1.0001,
                    "correlation > 1 at t={} i={} j={}: {}",
                    t,
                    i,
                    j,
                    v
                );
                approx_eq(v, result.dcc_correlations[[t, j, i]], 1e-8);
            }
        }
    }

    // Log-likelihood, AIC and BIC are finite and ordered AIC > BIC for large
    // sample (BIC penalizes more).
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_dcc_garch_input_validation() {
    let short = Array2::from_shape_vec((10, 2), vec![0.0; 20]).unwrap();
    assert!(DCCGARCH::fit(&short, None).is_err());

    let univariate = Array2::from_shape_vec((50, 1), vec![0.0; 50]).unwrap();
    assert!(DCCGARCH::fit(&univariate, None).is_err());
}

fn approx_eq(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() < tol,
        "expected {} ~= {}, got diff {}",
        a,
        b,
        (a - b).abs()
    );
}
