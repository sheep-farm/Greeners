//! Edge case tests: NaN, n < k, perfect collinearity, single observation,
//! constant y, dimension mismatches. These should return Err, not panic.
//!
//! NOTE: Tests with n=0 (empty matrices) are omitted because LAPACK segfaults
//! on 0-row matrices and SIGSEGV cannot be caught by catch_unwind.
//! The library should add n>0 guards before calling into LAPACK.

use greeners::{
    CovarianceType, DataFrame, FixedEffects, Formula, Logit, Probit, QuantileReg, RandomEffects,
    SurEquation, TimeSeries, FGLS, IV, OLS, SUR,
};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};

/// Catch panics (but not SIGSEGV). Use for tests where we suspect panic but not segfault.
fn catch_panic<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce() -> R + std::panic::UnwindSafe,
{
    std::panic::catch_unwind(f).map_err(|e| {
        if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        }
    })
}

// ============================================================================
// OLS edge cases
// ============================================================================

#[test]
fn test_ols_n_less_than_k() {
    let y = Array1::from(vec![1.0]);
    let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    assert!(OLS::fit(&y, &x, CovarianceType::NonRobust).is_err());
}

#[test]
fn test_ols_n_equals_k() {
    let y = Array1::from(vec![1.0, 2.0]);
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 2.0]).unwrap();
    assert!(OLS::fit(&y, &x, CovarianceType::NonRobust).is_err());
}

#[test]
fn test_ols_y_x_dimension_mismatch() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert!(OLS::fit(&y, &x, CovarianceType::NonRobust).is_err());
}

#[test]
fn test_ols_nan_in_y_does_not_panic() {
    let y = Array1::from(vec![1.0, f64::NAN, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = OLS::fit(&y, &x, CovarianceType::NonRobust);
    assert!(result.is_err(), "OLS with NaN in y should return Err");
}

#[test]
fn test_ols_nan_in_x_does_not_panic() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, f64::NAN, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = OLS::fit(&y, &x, CovarianceType::NonRobust);
    assert!(result.is_err(), "OLS with NaN in X should return Err");
}

#[test]
fn test_ols_perfect_collinearity_with_formula_drops_variable() {
    let mut data = IndexMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    );
    data.insert(
        "x2".to_string(),
        Array1::from(vec![2.0, 4.0, 6.0, 8.0, 10.0]),
    ); // x2 = 2*x1

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1 + x2").unwrap();

    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust);
    match result {
        Ok(res) => {
            assert!(
                !res.omitted_vars.is_empty(),
                "Should have omitted collinear variable"
            );
        }
        Err(_) => {} // also acceptable
    }
}

#[test]
fn test_ols_perfect_collinearity_without_names_singular() {
    // Without names, collinearity detection is skipped — matrix is singular
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    #[rustfmt::skip]
    let x = Array2::from_shape_vec((5, 3), vec![
        1.0, 1.0, 2.0,
        1.0, 2.0, 4.0,
        1.0, 3.0, 6.0,
        1.0, 4.0, 8.0,
        1.0, 5.0, 10.0,
    ]).unwrap();

    let result = catch_panic(|| OLS::fit(&y, &x, CovarianceType::NonRobust));
    match result {
        Ok(Err(_)) => {} // SingularMatrix — good
        Ok(Ok(_)) => {}  // some LAPACK may return garbage — at least no panic
        Err(msg) => panic!("Panicked on singular matrix: {}", msg),
    }
}

#[test]
fn test_ols_constant_y() {
    let y = Array1::from(vec![5.0, 5.0, 5.0, 5.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = catch_panic(|| OLS::fit(&y, &x, CovarianceType::NonRobust));
    match result {
        Ok(Ok(res)) => {
            assert!(
                res.params[1].abs() < 1e-10,
                "Slope should be ~0 for constant y"
            );
        }
        Ok(Err(_)) => {}
        Err(msg) => panic!("Panicked on constant y: {}", msg),
    }
}

#[test]
fn test_ols_inf_in_y_does_not_panic() {
    let y = Array1::from(vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = OLS::fit(&y, &x, CovarianceType::NonRobust);
    assert!(result.is_err(), "OLS with Inf in y should return Err");
}

// ============================================================================
// IV edge cases
// ============================================================================

#[test]
fn test_iv_fewer_instruments_than_regressors() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();
    let z = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    assert!(IV::fit(&y, &x, &z, CovarianceType::NonRobust).is_err());
}

#[test]
fn test_iv_dimension_mismatch() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    let z = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    assert!(IV::fit(&y, &x, &z, CovarianceType::NonRobust).is_err());
}

#[test]
fn test_iv_nan_does_not_panic() {
    let y = Array1::from(vec![1.0, f64::NAN, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let z = x.clone();

    let result = IV::fit(&y, &x, &z, CovarianceType::NonRobust);
    assert!(result.is_err(), "IV with NaN should return Err");
}

// ============================================================================
// Logit / Probit edge cases
// ============================================================================

#[test]
fn test_logit_n_less_than_k() {
    let y = Array1::from(vec![0.0, 1.0]);
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0]).unwrap();

    let result = catch_panic(|| Logit::fit(&y, &x));
    match result {
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Expected error for n < k in logit"),
        Err(msg) => panic!("Panicked on n < k in logit: {}", msg),
    }
}

#[test]
fn test_logit_all_zeros_y_does_not_panic() {
    let y = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = catch_panic(|| Logit::fit(&y, &x));
    match result {
        Ok(Err(_)) => {}
        Ok(Ok(res)) => {
            assert!(res.log_likelihood.is_finite() || res.log_likelihood.is_nan());
        }
        Err(msg) => panic!("Panicked on all-zero y in logit: {}", msg),
    }
}

#[test]
fn test_logit_all_ones_y_does_not_panic() {
    let y = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = catch_panic(|| Logit::fit(&y, &x));
    match result {
        Ok(_) => {} // error or extreme params, both ok
        Err(msg) => panic!("Panicked on all-one y in logit: {}", msg),
    }
}

#[test]
fn test_logit_nan_does_not_panic() {
    let y = Array1::from(vec![0.0, 1.0, f64::NAN, 0.0, 1.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = Logit::fit(&y, &x);
    assert!(result.is_err(), "Logit with NaN should return Err");
}

#[test]
fn test_probit_n_less_than_k() {
    let y = Array1::from(vec![0.0, 1.0]);
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0]).unwrap();

    let result = catch_panic(|| Probit::fit(&y, &x));
    match result {
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Expected error for n < k in probit"),
        Err(msg) => panic!("Panicked on n < k in probit: {}", msg),
    }
}

// ============================================================================
// Panel FE/RE edge cases
// ============================================================================

#[test]
fn test_fe_single_entity_does_not_panic() {
    // 1 entity, within-transform removes mean → may have df issues
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = vec![1i64, 1, 1, 1, 1];

    let result = catch_panic(|| FixedEffects::fit(&y, &x, &ids));
    match result {
        Ok(Err(_)) => {}
        Ok(Ok(res)) => {
            assert!(res.params.iter().all(|p| p.is_finite()));
        }
        Err(msg) => panic!("Panicked on single entity FE: {}", msg),
    }
}

#[test]
fn test_fe_mismatched_ids_length() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let ids = vec![1i64, 2]; // wrong length

    let result = catch_panic(|| FixedEffects::fit(&y, &x, &ids));
    match result {
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Expected error for mismatched entity_ids"),
        Err(msg) => panic!("Panicked on mismatched ids: {}", msg),
    }
}

#[test]
fn test_re_mismatched_ids() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let ids = Array1::from(vec![0i64, 1]); // wrong length

    let result = RandomEffects::fit(&y, &x, &ids);
    assert!(result.is_err());
}

// ============================================================================
// Quantile regression edge cases
// ============================================================================

#[test]
fn test_quantile_n_less_than_k() {
    let y = Array1::from(vec![1.0, 2.0]);
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0]).unwrap();

    let result = catch_panic(|| QuantileReg::fit(&y, &x, 0.5, 100));
    match result {
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Expected error for n < k in quantile"),
        Err(msg) => panic!("Panicked on n < k quantile: {}", msg),
    }
}

#[test]
fn test_quantile_nboot_zero_returns_error() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = QuantileReg::fit(&y, &x, 0.5, 0);
    assert!(
        result.is_err(),
        "QuantileReg with n_boot=0 should return Err"
    );
}

#[test]
fn test_quantile_invalid_tau_zero() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    assert!(QuantileReg::fit(&y, &x, 0.0, 100).is_err());
}

#[test]
fn test_quantile_invalid_tau_one() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    assert!(QuantileReg::fit(&y, &x, 1.0, 100).is_err());
}

#[test]
fn test_quantile_invalid_tau_negative() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    assert!(QuantileReg::fit(&y, &x, -0.5, 100).is_err());
}

// ============================================================================
// WLS edge cases
// ============================================================================

#[test]
fn test_wls_negative_weights_does_not_panic() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let w = Array1::from(vec![1.0, -1.0, 1.0, 1.0, 1.0]);

    let result = catch_panic(|| FGLS::wls(&y, &x, &w));
    match result {
        Ok(_) => {} // error or NaN from sqrt of negative, both ok
        Err(msg) => panic!("WLS with negative weights should not panic: {}", msg),
    }
}

#[test]
fn test_wls_zero_weights_does_not_panic() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();
    let w = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0]);

    let result = catch_panic(|| FGLS::wls(&y, &x, &w));
    match result {
        Ok(_) => {} // singular or garbage, both ok
        Err(msg) => panic!("Panicked on zero weights: {}", msg),
    }
}

#[test]
fn test_wls_mismatched_weights_length() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let w = Array1::from(vec![1.0, 1.0]); // wrong length

    let result = catch_panic(|| FGLS::wls(&y, &x, &w));
    match result {
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Expected error for weight length mismatch"),
        Err(msg) => panic!("Panicked on weight mismatch: {}", msg),
    }
}

// ============================================================================
// SUR edge cases
// ============================================================================

#[test]
fn test_sur_single_equation_does_not_panic() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let equations = vec![SurEquation {
        y,
        x,
        name: "eq1".to_string(),
    }];

    let result = catch_panic(|| SUR::fit(&equations));
    match result {
        Ok(Ok(res)) => assert_eq!(res.equations.len(), 1),
        Ok(Err(_)) => {}
        Err(msg) => panic!("Panicked on single-equation SUR: {}", msg),
    }
}

#[test]
fn test_sur_mismatched_n() {
    let equations = vec![
        SurEquation {
            y: Array1::from(vec![1.0, 2.0, 3.0]),
            x: Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap(),
            name: "eq1".to_string(),
        },
        SurEquation {
            y: Array1::from(vec![1.0, 2.0]),
            x: Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap(),
            name: "eq2".to_string(),
        },
    ];

    let result = SUR::fit(&equations);
    assert!(result.is_err());
}

// ============================================================================
// TimeSeries ADF edge cases
// ============================================================================

#[test]
fn test_adf_too_short() {
    let arr = Array1::from(vec![1.0, 2.0, 3.0]);
    assert!(TimeSeries::adf(&arr, None).is_err());
}

#[test]
fn test_adf_constant_series_does_not_panic() {
    let arr = Array1::from(vec![5.0; 50]);

    let result = catch_panic(|| TimeSeries::adf(&arr, Some(1)));
    match result {
        Ok(_) => {} // error or extreme stat, both ok
        Err(msg) => panic!("Panicked on constant series ADF: {}", msg),
    }
}

// ============================================================================
// DataFrame edge cases
// ============================================================================

#[test]
fn test_dataframe_empty_hashmap_does_not_panic() {
    let data: IndexMap<String, Array1<f64>> = IndexMap::new();

    let result = catch_panic(|| DataFrame::new(data));
    match result {
        Ok(_) => {} // empty or error, both ok
        Err(msg) => panic!("Panicked on empty HashMap: {}", msg),
    }
}

#[test]
fn test_dataframe_mismatched_column_lengths() {
    let mut data = IndexMap::new();
    data.insert("a".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    data.insert("b".to_string(), Array1::from(vec![1.0, 2.0]));

    let result = catch_panic(|| DataFrame::new(data));
    match result {
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Expected error for mismatched column lengths"),
        Err(msg) => panic!("Panicked on mismatched columns: {}", msg),
    }
}

#[test]
fn test_formula_missing_variable() {
    let mut data = IndexMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    data.insert("x1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1 + x_missing").unwrap();

    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust);
    assert!(result.is_err());
}

#[test]
fn test_formula_empty_string() {
    let result = Formula::parse("");
    assert!(result.is_err());
}

#[test]
fn test_formula_no_tilde() {
    let result = Formula::parse("y x1 x2");
    assert!(result.is_err());
}
