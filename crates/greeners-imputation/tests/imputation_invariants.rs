use greeners_imputation::imputation::BayesGaussMI;
use greeners_imputation::imputation::MICE;
use indexmap::IndexMap;
use ndarray::Array1;

fn make_missing_data() -> IndexMap<String, Array1<f64>> {
    let n = 20;
    let mut col0 = Array1::from_vec((1..=n).map(|i| i as f64).collect());
    let mut col1: Array1<f64> = col0.mapv(|v| 2.0 * v + 0.5);
    let mut col2: Array1<f64> = col0.mapv(|v| -1.0 * v + 3.0);

    // Spread a few distinct missing values across columns so complete cases remain
    col0[2] = f64::NAN;
    col0[7] = f64::NAN;
    col1[5] = f64::NAN;
    col1[12] = f64::NAN;
    col2[9] = f64::NAN;
    col2[14] = f64::NAN;

    let mut data = IndexMap::new();
    data.insert("x0".into(), col0);
    data.insert("x1".into(), col1);
    data.insert("x2".into(), col2);
    data
}

/// MICE produces the requested number of imputed datasets without NaNs.
#[test]
fn test_mice_impute_invariants() {
    let data = make_missing_data();
    let n = data["x0"].len();

    let r = MICE::impute(&data, 3, 5).unwrap();
    assert_eq!(r.n_obs, n);
    assert_eq!(r.n_vars, 3);
    assert_eq!(r.n_imputations, 3);
    assert_eq!(r.n_iter, 5);
    assert_eq!(r.datasets.len(), 3);

    for ds in &r.datasets {
        assert_eq!(ds.len(), 3);
        for (name, col) in ds {
            assert_eq!(col.len(), n);
            assert!(col.iter().all(|v| v.is_finite()));
            assert!(data.contains_key(name));
        }
    }
}

/// Bayesian Gaussian MI produces imputed datasets of the expected shape.
#[test]
fn test_bayes_gauss_mi_invariants() {
    let data = make_missing_data();
    let n = data["x0"].len();

    let r = BayesGaussMI::impute(&data, 4).unwrap();
    assert_eq!(r.n_obs, n);
    assert_eq!(r.n_vars, 3);
    assert_eq!(r.n_imputations, 4);
    assert_eq!(r.datasets.len(), 4);

    for ds in &r.datasets {
        assert_eq!(ds.len(), 3);
        for (name, col) in ds {
            assert_eq!(col.len(), n);
            assert!(col.iter().all(|v| v.is_finite()));
            assert!(data.contains_key(name));
        }
    }
}

/// Input validation rejects empty data and length mismatches.
#[test]
fn test_imputation_input_validation() {
    let empty: IndexMap<String, Array1<f64>> = IndexMap::new();
    assert!(MICE::impute(&empty, 1, 1).is_err());
    assert!(BayesGaussMI::impute(&empty, 1).is_err());

    let mut bad = IndexMap::new();
    bad.insert("a".into(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
    bad.insert("b".into(), Array1::from_vec(vec![1.0, 2.0]));
    assert!(MICE::impute(&bad, 1, 1).is_err());
}
