use greeners_imputation::mice::MiceChained;
use ndarray::Array2;

fn make_missing_matrix() -> Array2<f64> {
    let n = 20;
    let k = 3;
    let mut data = Array2::zeros((n, k));
    for i in 0..n {
        data[[i, 0]] = i as f64;
        data[[i, 1]] = 2.0 * i as f64 + 0.5;
        data[[i, 2]] = -1.0 * i as f64 + 3.0;
    }

    // Spread missing values so no column is fully missing
    data[[2, 0]] = f64::NAN;
    data[[7, 0]] = f64::NAN;
    data[[5, 1]] = f64::NAN;
    data[[12, 1]] = f64::NAN;
    data[[9, 2]] = f64::NAN;
    data[[14, 2]] = f64::NAN;

    data
}

/// MiceChained returns a completed matrix with the expected shapes.
#[test]
fn test_mice_chained_invariants() {
    let data = make_missing_matrix();
    let names = vec!["x0".into(), "x1".into(), "x2".into()];

    let r = MiceChained::fit(&data, Some(3), Some(5), Some(names)).unwrap();

    assert_eq!(r.n_obs, 20);
    assert_eq!(r.n_vars, 3);
    assert_eq!(r.n_imputations, 3);
    assert_eq!(r.n_iterations, 5);
    assert!(r.n_missing > 0);
    assert_eq!(r.imputed_data.shape(), [20, 3]);
    assert!(r.imputed_data.iter().all(|v| v.is_finite()));

    assert_eq!(r.variable_names.len(), 3);
    assert_eq!(r.missing_per_var.len(), 3);
    assert_eq!(r.pooled_mean.len(), 3);
    assert_eq!(r.pooled_variance.len(), 3);
    assert_eq!(r.within_variance.len(), 3);
    assert_eq!(r.between_variance.len(), 3);
    assert_eq!(r.missing_info_rate.len(), 3);

    assert!(r.pooled_mean.iter().all(|v| v.is_finite()));
    assert!(r.pooled_variance.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(r
        .missing_info_rate
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
}

/// Different requested numbers of imputations and iterations are honored.
#[test]
fn test_mice_chained_parameters() {
    let data = make_missing_matrix();

    let r1 = MiceChained::fit(&data, Some(2), Some(3), None).unwrap();
    assert_eq!(r1.n_imputations, 2);
    assert_eq!(r1.n_iterations, 3);

    let r2 = MiceChained::fit(&data, Some(5), Some(10), None).unwrap();
    assert_eq!(r2.n_imputations, 5);
    assert_eq!(r2.n_iterations, 10);

    // Both fill all missing values
    assert!(r1.imputed_data.iter().all(|v| v.is_finite()));
    assert!(r2.imputed_data.iter().all(|v| v.is_finite()));
}

/// Input validation rejects data without missing values and small dimensions.
#[test]
fn test_mice_chained_input_validation() {
    let complete = Array2::from_shape_vec((10, 2), (1..=20).map(|v| v as f64).collect()).unwrap();
    assert!(MiceChained::fit(&complete, None, None, None).is_err());

    let small = Array2::from_shape_vec((4, 2), (1..=8).map(|v| v as f64).collect()).unwrap();
    assert!(MiceChained::fit(&small, None, None, None).is_err());

    let one_col_2d = Array2::from_shape_vec((20, 1), (1..=20).map(|v| v as f64).collect()).unwrap();
    assert!(MiceChained::fit(&one_col_2d, None, None, None).is_err());

    // Fully missing column is rejected
    let mut fully_missing = make_missing_matrix();
    for i in 0..20 {
        fully_missing[[i, 0]] = f64::NAN;
    }
    assert!(MiceChained::fit(&fully_missing, None, None, None).is_err());
}
