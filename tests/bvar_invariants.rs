use greeners::linalg::LinalgInverse as _;
use greeners::BVAR;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Generate a VAR(1) process with independent normal noise.
fn generate_var1_with_noise(
    a: &Array2<f64>,
    y0: &Array1<f64>,
    t: usize,
    noise_sd: f64,
    seed: u64,
) -> Array2<f64> {
    let k = a.ncols();
    let mut data = Array2::zeros((t, k));
    data.row_mut(0).assign(y0);
    let mut rng = StdRng::seed_from_u64(seed);
    let norm = Normal::new(0.0, noise_sd).unwrap();
    for i in 1..t {
        let y_prev = data.row(i - 1).to_owned();
        let y_t = a.dot(&y_prev);
        data.row_mut(i).assign(&y_t);
        for j in 0..k {
            data[[i, j]] += norm.sample(&mut rng);
        }
    }
    data
}

/// With a very diffuse Minnesota prior and a well-conditioned, noisy data
/// set, BVAR posterior mean converges to the OLS estimate on the lag design.
#[test]
fn test_bvar_diffuse_prior_equals_ols() {
    let a = Array2::from_shape_vec((2, 2), vec![0.3, 0.2, 0.1, 0.4]).unwrap();
    let y0 = Array1::from(vec![1.0, 2.0]);
    // Enough noise so the OLS residual variance is moderate and the prior
    // precision (which scales with sigma2_ols) becomes negligible.
    let data = generate_var1_with_noise(&a, &y0, 80, 1.0, 42);

    // Build the BVAR design matrix (no intercept) exactly as BVAR::fit does.
    let t = data.nrows();
    let k = data.ncols();
    let lags = 1;
    let n_eff = t - lags;
    let n_reg = k * lags;
    let mut x = Array2::zeros((n_eff, n_reg));
    let mut y_dep = Array2::zeros((n_eff, k));
    for i in 0..n_eff {
        for j in 0..k {
            y_dep[[i, j]] = data[[lags + i, j]];
        }
        for lag in 0..lags {
            for j in 0..k {
                x[[i, lag * k + j]] = data[[lags + i - 1 - lag, j]];
            }
        }
    }

    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&y_dep);
    let ols_beta = xtx.inv().unwrap().dot(&xty);

    // Diffuse prior: large lambdas make V0 large and V0^{-1} negligible
    // relative to X'X, so the posterior is dominated by the likelihood.
    let bvar = BVAR::fit(&data, 1, Some(1e4), Some(1e4), Some(1.0), None).unwrap();

    for eq in 0..k {
        for col in 0..n_reg {
            let ols_coef = ols_beta[[col, eq]];
            let bvar_coef = bvar.coeffs[[eq, col]];
            approx_zero((bvar_coef - ols_coef).abs(), 1e-4);
        }
    }
}

/// With a very tight Minnesota prior centered on a random walk, the posterior
/// shrinks the own-lag-1 coefficient toward 1 and cross-lags toward 0.
#[test]
fn test_bvar_tight_prior_shrinks_to_random_walk() {
    // Weak signal VAR with small coefficients and moderate noise.
    let a = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 0.05, 0.05]).unwrap();
    let y0 = Array1::from(vec![1.0, 1.0]);
    let data = generate_var1_with_noise(&a, &y0, 50, 0.5, 123);

    // Tight prior around random walk (own lag 1 = 1, others = 0).
    let bvar = BVAR::fit(&data, 1, Some(1e-8), Some(1e-8), Some(1.0), None).unwrap();

    // Own first-lag coefficients should be close to 1.
    approx_zero((bvar.coeffs[[0, 0]] - 1.0).abs(), 1e-4);
    approx_zero((bvar.coeffs[[1, 1]] - 1.0).abs(), 1e-4);

    // Cross-lag coefficients should be close to 0.
    approx_zero(bvar.coeffs[[0, 1]].abs(), 1e-4);
    approx_zero(bvar.coeffs[[1, 0]].abs(), 1e-4);
}

/// BVAR rejects lags = 0 and too few observations.
#[test]
fn test_bvar_input_validation() {
    let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert!(BVAR::fit(&data, 0, None, None, None, None).is_err());
    assert!(BVAR::fit(&data, 2, None, None, None, None).is_err());
}
