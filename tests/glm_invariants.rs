use greeners::{CovarianceType, Family, GLM, OLS};
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// GLM with Gaussian/identity is equivalent to OLS.
#[test]
fn test_glm_gaussian_identity_equals_ols() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
    let y = Array1::from(vec![4.0, 5.0, 7.0, 10.0]);

    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let glm = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();

    assert!(glm.converged, "GLM did not converge");
    for i in 0..ols.params.len() {
        approx_zero((glm.params[i] - ols.params[i]).abs(), 1e-10);
    }
}

/// Poisson with log link: for y = exp(β0 + β1 x) the MLE recovers β.
#[test]
fn test_glm_poisson_log_exact() {
    let x_col = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let x = Array2::from_shape_vec((4, 2), {
        let mut v = vec![];
        for &xi in x_col.iter() {
            v.push(1.0);
            v.push(xi);
        }
        v
    })
    .unwrap();

    let eta = &x.column(0) * 1.0 + &x.column(1) * 2.0;
    let y = eta.mapv(|e: f64| e.exp());

    let glm = GLM::fit(&y, &x, Family::Poisson, CovarianceType::NonRobust).unwrap();
    assert!(glm.converged, "GLM did not converge");

    approx_zero((glm.params[0] - 1.0).abs(), 1e-8);
    approx_zero((glm.params[1] - 2.0).abs(), 1e-8);
}

/// Score equation at IRLS convergence: X' (y - μ) = 0 for canonical link.
/// For logit, the canonical link gives g'(μ) = 1/V(μ), so the score reduces
/// to this simple form.
#[test]
fn test_glm_score_equation_logit() {
    // Simple logit DGP: y = P(y=1) = invlogit(1 + 2x)
    // We fit using the probabilities themselves, so the true β = [1, 2].
    let x_col = Array1::from(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    let x = Array2::from_shape_vec((5, 2), {
        let mut v = vec![];
        for &xi in x_col.iter() {
            v.push(1.0);
            v.push(xi);
        }
        v
    })
    .unwrap();

    let eta = &x.column(0) * 1.0 + &x.column(1) * 2.0;
    let y = eta.mapv(|e: f64| {
        let ex = e.exp();
        ex / (1.0 + ex)
    });

    let glm = GLM::fit(&y, &x, Family::Binomial, CovarianceType::NonRobust).unwrap();
    assert!(glm.converged, "GLM did not converge");

    // Compute fitted μ from the GLM estimate and check the score equation.
    let eta_hat = x.dot(&glm.params);
    let mu = eta_hat.mapv(|e: f64| {
        let ex = e.exp();
        ex / (1.0 + ex)
    });

    let residuals = &y - &mu;
    let score = x.t().dot(&residuals);
    for &v in score.iter() {
        approx_zero(v, 1e-6);
    }

    // Also check that the estimated parameters are close to the true ones.
    approx_zero((glm.params[0] - 1.0).abs(), 1e-6);
    approx_zero((glm.params[1] - 2.0).abs(), 1e-6);
}
