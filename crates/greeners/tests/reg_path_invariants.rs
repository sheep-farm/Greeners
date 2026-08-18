use greeners::linalg::LinalgInverse as _;
use greeners::RegPath;
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Ridge coefficients must equal the closed-form formula applied to the
/// standardized data and then un-standardized.
#[test]
fn test_ridge_matches_closed_form() {
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 5.0, 6.0, 6.0, 7.0, 8.0, 8.0, 7.0, 9.0,
            9.0, 10.0, 10.0,
        ],
    )
    .unwrap();
    let y = x.column(0).to_owned() * 2.0 - x.column(1).to_owned() * 1.5 + 3.0;

    // Use two lambda points (max and min) to avoid the degenerate n=1 path.
    let result = RegPath::fit(&y, &x, "ridge", Some(1.0), Some(2), None).unwrap();

    // Compute the same standardization as RegPath.
    let n = x.nrows();
    let p = x.ncols();
    let x_mean: Array1<f64> = (0..p)
        .map(|j| (0..n).map(|i| x[[i, j]]).sum::<f64>() / n as f64)
        .collect();
    let x_std: Array1<f64> = (0..p)
        .map(|j| {
            let var = (0..n).map(|i| (x[[i, j]] - x_mean[j]).powi(2)).sum::<f64>() / n as f64;
            var.sqrt().max(1e-10)
        })
        .collect();
    let y_mean = y.mean().unwrap_or(0.0);
    let y_std = y.std(0.0).max(1e-10);

    let x_norm: Array2<f64> = {
        let mut data = Vec::with_capacity(n * p);
        for i in 0..n {
            for j in 0..p {
                data.push((x[[i, j]] - x_mean[j]) / x_std[j]);
            }
        }
        Array2::from_shape_vec((n, p), data).unwrap()
    };
    let y_norm: Array1<f64> = y.mapv(|v| (v - y_mean) / y_std);

    let xty = x_norm.t().dot(&y_norm);
    let lambda_max = xty.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max) / (n as f64);
    let lambda_min = lambda_max * 1e-4;

    // Closed-form ridge on standardized data for both lambdas.
    let expected_for = |lambda: f64| {
        let xtx = x_norm.t().dot(&x_norm);
        let mut xtx_reg = xtx.clone();
        for j in 0..p {
            xtx_reg[[j, j]] += lambda;
        }
        let xtx_inv = xtx_reg.inv().unwrap();
        let beta_norm = xtx_inv.dot(&xty);

        let mut coefs = Array1::zeros(p);
        for j in 0..p {
            coefs[j] = beta_norm[j] * y_std / x_std[j];
        }
        let intercept = y_mean - (0..p).map(|j| coefs[j] * x_mean[j]).sum::<f64>();
        (coefs, intercept)
    };

    let (coefs_max, intercept_max) = expected_for(lambda_max);
    let (coefs_min, intercept_min) = expected_for(lambda_min);

    for j in 0..p {
        approx_zero((result.coef_path[[0, j]] - coefs_max[j]).abs(), 1e-8);
        approx_zero((result.coef_path[[1, j]] - coefs_min[j]).abs(), 1e-8);
    }
    approx_zero((result.intercept_path[0] - intercept_max).abs(), 1e-8);
    approx_zero((result.intercept_path[1] - intercept_min).abs(), 1e-8);

    let best_idx = if (result.optimal_lambda - lambda_max).abs() < 1e-10 {
        0
    } else {
        1
    };
    for j in 0..p {
        approx_zero(
            (result.optimal_coefs[j] - result.coef_path[[best_idx, j]]).abs(),
            1e-12,
        );
    }
    approx_zero(
        (result.optimal_intercept - result.intercept_path[best_idx]).abs(),
        1e-12,
    );
}

/// With a strong Lasso penalty all slope coefficients are zero.
#[test]
fn test_lasso_strong_penalty_zeros_coefficients() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 0.1, 1.0, 0.2, 1.0, -0.1, 1.0, -0.2, 1.0, 0.0],
    )
    .unwrap();
    let y = Array1::from(vec![1.0, 1.05, 0.95, 0.9, 1.0]);

    // n_lambdas=2; the larger lambda (lambda_max) should dominate.
    let result = RegPath::fit(&y, &x, "lasso", Some(1.0), Some(2), None).unwrap();

    // Either row may be selected, but the larger-lambda row must have all
    // slope coefficients zero for this weak-signal design.
    for j in 0..result.n_pred {
        approx_zero(result.coef_path[[0, j]], 1e-10);
    }
}

/// ElasticNet with alpha=0.0 falls back to Ridge and returns finite
/// coefficients.
#[test]
fn test_elasticnet_alpha_zero_is_ridge() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 5.0],
    )
    .unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let result = RegPath::fit(&y, &x, "elasticnet", Some(0.0), Some(2), None).unwrap();

    // alpha=0 means only L2 penalty; all coefficients should be finite.
    assert!(result.optimal_coefs.iter().all(|&v| v.is_finite()));
}
