use greeners::{CovarianceType, Equation, SurEquation, ThreeSLS, IV, OLS, SUR};
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Two-equation SUR design with orthogonal residual vectors.
/// X = [const, x], x = [1,2,3,4].
/// Eq1: y1 = 1 + 2x + e1,  e1 = [1,0,-3,2]
/// Eq2: y2 = 3 + 4x + e2,  e2 = [-4,7,-2,-1]
/// e1 and e2 are each orthogonal to X and to each other.
fn sur_exact_design() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
    let e1 = Array1::from(vec![1.0, 0.0, -3.0, 2.0]);
    let e2 = Array1::from(vec![-4.0, 7.0, -2.0, -1.0]);

    let y1 = &(&x.column(0) * 1.0 + &x.column(1) * 2.0) + &e1;
    let y2 = &(&x.column(0) * 3.0 + &x.column(1) * 4.0) + &e2;

    (x, y1, y2)
}

/// SUR with uncorrelated equation errors reduces to OLS equation-by-equation.
#[test]
fn test_sur_equals_ols_with_uncorrelated_errors() {
    let (x, y1, y2) = sur_exact_design();

    let ols1 = OLS::fit(&y1, &x, CovarianceType::NonRobust).unwrap();
    let ols2 = OLS::fit(&y2, &x, CovarianceType::NonRobust).unwrap();

    let equations = vec![
        SurEquation {
            y: y1,
            x: x.clone(),
            name: "eq1".into(),
        },
        SurEquation {
            y: y2,
            x,
            name: "eq2".into(),
        },
    ];
    let sur = SUR::fit(&equations).unwrap();

    for i in 0..2 {
        approx_zero((sur.equations[0].params[i] - ols1.params[i]).abs(), 1e-12);
        approx_zero((sur.equations[1].params[i] - ols2.params[i]).abs(), 1e-12);
    }

    // Cross-equation residual covariance should be zero for this design.
    approx_zero(sur.sigma_cross[[0, 1]], 1e-12);
    approx_zero(sur.sigma_cross[[1, 0]], 1e-12);
}

/// 3SLS with a single equation reduces to 2SLS.
#[test]
fn test_3sls_single_equation_equals_2sls() {
    // DGP: x = 2 + 3z,  y = 1 + 5x
    let z = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let x = z.mapv(|v| 2.0 + 3.0 * v);
    let y = x.mapv(|v| 1.0 + 5.0 * v);

    let n = z.len();
    let x_mat = Array2::from_shape_vec((n, 2), {
        let mut v = vec![];
        for i in 0..n {
            v.push(1.0);
            v.push(x[i]);
        }
        v
    })
    .unwrap();

    let z_mat = Array2::from_shape_vec((n, 2), {
        let mut v = vec![];
        for i in 0..n {
            v.push(1.0);
            v.push(z[i]);
        }
        v
    })
    .unwrap();

    let iv = IV::fit(&y, &x_mat, &z_mat, CovarianceType::NonRobust).unwrap();

    let equations = vec![Equation {
        y: y,
        x: x_mat,
        name: "eq1".into(),
        var_names: vec!["const".into(), "x".into()],
    }];
    let threesl = ThreeSLS::fit(&equations, &z_mat).unwrap();

    for i in 0..iv.params.len() {
        approx_zero((threesl.equations[0].params[i] - iv.params[i]).abs(), 1e-12);
    }
}
