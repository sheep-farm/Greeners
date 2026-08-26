use greeners_timeseries::var::VAR;
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Generate a deterministic VAR(1) process:
///   y_t = c + A y_{t-1}
/// with no noise.
fn generate_var1(c: &Array1<f64>, a: &Array2<f64>, y0: &Array1<f64>, t: usize) -> Array2<f64> {
    let k = c.len();
    let mut data = Array2::zeros((t, k));
    data.row_mut(0).assign(y0);
    for i in 1..t {
        let y_prev = data.row(i - 1).to_owned();
        let y_t = c + a.dot(&y_prev);
        data.row_mut(i).assign(&y_t);
    }
    data
}

#[test]
fn test_var1_exact_recovery() {
    // True parameters
    let c = Array1::from(vec![1.0, 2.0]);
    let a = Array2::from_shape_vec((2, 2), vec![0.3, 0.2, 0.1, 0.4]).unwrap();
    let y0 = Array1::from(vec![0.0, 0.0]);

    let data = generate_var1(&c, &a, &y0, 30);
    let result = VAR::fit(&data, 1, None).unwrap();

    // result.params is (1 + k*p) x k.
    // Row 0: intercept (transposed -> column in params).
    // Rows 1..1+k: AR matrix, row-major by equation.
    let k = 2;
    for j in 0..k {
        approx_zero((result.params[[0, j]] - c[j]).abs(), 1e-10);
    }

    // params[1..3, :] contains A (stacked by row of A):
    // Row 1 = first row of A? Actually X has columns [1, y1(t-1), y2(t-1)] and
    // there are two equations (columns of Y). The parameter matrix has shape
    // (1+k*p) x k. Row 1 is the coefficient of y1(t-1), row 2 of y2(t-1);
    // column 0 is equation for y1, column 1 for y2.
    // So params[[1,0]] = A[0,0], params[[2,0]] = A[0,1],
    //    params[[1,1]] = A[1,0], params[[2,1]] = A[1,1].
    approx_zero((result.params[[1, 0]] - a[[0, 0]]).abs(), 1e-10);
    approx_zero((result.params[[2, 0]] - a[[0, 1]]).abs(), 1e-10);
    approx_zero((result.params[[1, 1]] - a[[1, 0]]).abs(), 1e-10);
    approx_zero((result.params[[2, 1]] - a[[1, 1]]).abs(), 1e-10);
}

#[test]
fn test_var_residuals_orthogonal_to_regressors() {
    let c = Array1::from(vec![1.0, 2.0]);
    let a = Array2::from_shape_vec((2, 2), vec![0.3, 0.2, 0.1, 0.4]).unwrap();
    let y0 = Array1::from(vec![0.0, 0.0]);

    let data = generate_var1(&c, &a, &y0, 30);
    let result = VAR::fit(&data, 1, None).unwrap();

    // Build the X and Y used by the estimator.
    let lags = 1;
    let y_eff = data.slice(ndarray::s![lags.., ..]).to_owned();
    let n_obs = y_eff.nrows();
    let k = data.ncols();
    let n_cols_x = 1 + k * lags;
    let mut x_mat = Array2::zeros((n_obs, n_cols_x));
    x_mat.column_mut(0).fill(1.0);
    for i in 0..n_obs {
        for j in 0..k {
            x_mat[[i, 1 + j]] = data[[lags + i - 1, j]];
        }
    }

    let preds = x_mat.dot(&result.params);
    let residuals = &y_eff - &preds;

    // Orthogonality: X' U = 0 for each equation.
    let xtu = x_mat.t().dot(&residuals);
    for &v in xtu.iter() {
        approx_zero(v, 1e-10);
    }
}

#[test]
fn test_var_lag_greater_than_observations_fails() {
    let data = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let result = VAR::fit(&data, 5, None);
    assert!(result.is_err());
}
