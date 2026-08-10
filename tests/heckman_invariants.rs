use greeners::{CovarianceType, Heckman, OLS};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// When selection is independent of the outcome error (rho = 0), the
/// Heckman two-step outcome coefficients should be close to OLS on the
/// selected sample and the inverse-Mills coefficient should be near 0.
#[test]
fn test_heckman_no_selection_bias() {
    let n = 300;
    let mut rng = StdRng::seed_from_u64(777);
    let norm = Normal::new(0.0, 1.0).unwrap();

    // Generate regressors and errors.
    let x: Vec<f64> = (0..n).map(|_| norm.sample(&mut rng)).collect();
    let w: Vec<f64> = (0..n).map(|_| norm.sample(&mut rng)).collect();
    let u: Vec<f64> = (0..n).map(|_| norm.sample(&mut rng)).collect();
    let v: Vec<f64> = (0..n).map(|_| norm.sample(&mut rng)).collect();

    // Selection: z = 1 iff 0.5 + w > v (independent of u).
    let z: Vec<f64> = (0..n)
        .map(|i| if 0.5 + w[i] > v[i] { 1.0 } else { 0.0 })
        .collect();

    // Outcome: y = 1 + 2x + u.
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + u[i]).collect();

    // Build x_out with intercept and x.
    let mut x_out_vec = Vec::with_capacity(n * 2);
    for i in 0..n {
        x_out_vec.push(1.0);
        x_out_vec.push(x[i]);
    }
    let x_out = Array2::from_shape_vec((n, 2), x_out_vec).unwrap();

    // Build x_sel with intercept and w.
    let mut x_sel_vec = Vec::with_capacity(n * 2);
    for i in 0..n {
        x_sel_vec.push(1.0);
        x_sel_vec.push(w[i]);
    }
    let x_sel = Array2::from_shape_vec((n, 2), x_sel_vec).unwrap();

    let y = Array1::from(y);
    let z = Array1::from(z);

    let result = Heckman::fit(&y, &x_out, &z, &x_sel, None, None).unwrap();

    // Outcome equation intercept and slope should be close to the true values.
    approx_zero((result.params[0] - 1.0).abs(), 0.3);
    approx_zero((result.params[1] - 2.0).abs(), 0.1);

    // Inverse-Mills coefficient (delta) and rho should be small (rho ≈ 0).
    assert!(
        result.delta.abs() < 0.5,
        "delta too large: {}",
        result.delta
    );
    assert!(result.rho.abs() < 0.3, "rho too large: {}", result.rho);

    // Compare to OLS on selected observations.
    let selected_idx: Vec<usize> = (0..n).filter(|&i| z[i] == 1.0).collect();
    let n1 = selected_idx.len();
    let mut x_sel_ols = Array2::zeros((n1, 2));
    let mut y_sel = Array1::zeros(n1);
    for (r, &i) in selected_idx.iter().enumerate() {
        x_sel_ols[[r, 0]] = 1.0;
        x_sel_ols[[r, 1]] = x[i];
        y_sel[r] = y[i];
    }
    let ols = OLS::fit(&y_sel, &x_sel_ols, CovarianceType::NonRobust).unwrap();

    approx_zero((result.params[0] - ols.params[0]).abs(), 0.1);
    approx_zero((result.params[1] - ols.params[1]).abs(), 0.1);

    assert!(result.sigma > 0.0);
    assert!(result.n_obs == n);
    assert!(result.n_selected > 0);
}

/// Input validation.
#[test]
fn test_heckman_input_validation() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x_out = Array2::from_shape_vec((5, 1), vec![1.0; 5]).unwrap();
    let x_sel = Array2::from_shape_vec((5, 1), vec![1.0; 5]).unwrap();

    let z_good = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0]);
    let _ = Heckman::fit(&y, &x_out, &z_good, &x_sel, None, None).unwrap();

    let z_bad = Array1::from(vec![0.0, 2.0, 1.0, 1.0, 1.0]);
    assert!(Heckman::fit(&y, &x_out, &z_bad, &x_sel, None, None).is_err());

    let z_short = Array1::from(vec![0.0, 1.0, 1.0, 1.0]);
    assert!(Heckman::fit(&y, &x_out, &z_short, &x_sel, None, None).is_err());
}
