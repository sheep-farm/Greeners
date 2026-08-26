use greeners_timeseries::var::VAR;
use greeners_timeseries::varma::VARMA;
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Generate a deterministic VAR(1) process.
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

/// VARMA(1,0) should recover the same AR parameters as VAR(1) on
/// deterministic data.
#[test]
fn test_varma_q0_reduces_to_var() {
    let c = Array1::from(vec![1.0, 2.0]);
    let a = Array2::from_shape_vec((2, 2), vec![0.3, 0.2, 0.1, 0.4]).unwrap();
    let y0 = Array1::from(vec![0.0, 0.0]);

    // Need enough observations for Hannan-Rissanen p_long.
    let data = generate_var1(&c, &a, &y0, 120);

    let var = VAR::fit(&data, 1, None).unwrap();
    let varma = VARMA::fit(&data, 1, 0).unwrap();

    // First row of each column is the intercept.
    for j in 0..2 {
        approx_zero((varma.ar_params[[0, j]] - var.params[[0, j]]).abs(), 1e-6);
    }

    // AR parameters are stacked row by row in the same way for both.
    for r in 1..3 {
        for c_idx in 0..2 {
            approx_zero(
                (varma.ar_params[[r, c_idx]] - var.params[[r, c_idx]]).abs(),
                1e-6,
            );
        }
    }

    // With q=0, the MA parameter block should be empty / zero-sized.
    assert_eq!(varma.ma_params.nrows(), 0);
}

/// VARMA with q > 0 on a mildly non-deterministic series succeeds and
/// returns parameters of the expected shape.
#[test]
fn test_varma_q1_shape_and_residual_covariance() {
    // Generate a simple 1D ARMA-like process with a deterministic shock
    // sequence:  y_t = 0.5 y_{t-1} + u_t + 0.3 u_{t-1},  u_t = sin(t)/10
    let t = 200;
    let mut u = Array1::zeros(t);
    for i in 0..t {
        u[i] = (i as f64).sin() / 10.0;
    }

    let mut data = Array2::zeros((t, 1));
    data[[0, 0]] = u[0];
    for i in 1..t {
        let ar_part = 0.5 * data[[i - 1, 0]];
        let ma_part = u[i] + 0.3 * u[i - 1];
        data[[i, 0]] = ar_part + ma_part;
    }

    let result = VARMA::fit(&data, 1, 1).unwrap();

    // Parameter matrices have the expected dimensions.
    assert_eq!(result.ar_params.nrows(), 2); // 1 + p * k
    assert_eq!(result.ma_params.nrows(), 1); // q * k
    assert_eq!(result.ar_params.ncols(), 1);
    assert_eq!(result.ma_params.ncols(), 1);

    // Residual covariance should be finite and non-negative.
    assert!(result.sigma_u[[0, 0]].is_finite());
    assert!(result.sigma_u[[0, 0]] >= 0.0);
}
