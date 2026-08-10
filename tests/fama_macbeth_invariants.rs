use greeners::{DataFrame, FamaMacBeth, Formula};
use indexmap::IndexMap;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Fama-MacBeth mean coefficients equal the average of the period-by-period
/// cross-sectional OLS estimates.
#[test]
fn test_fama_macbeth_mean_coefficients() {
    // Three periods, five observations each, with a small noise so the
    // period-by-period coefficients vary and standard errors are positive.
    let mut rng = StdRng::seed_from_u64(123);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let slopes = [1.0, 2.0, 3.0];
    let mut y_vec = Vec::new();
    let mut x_vec = Vec::new();
    let mut t_vec = Vec::new();
    for (period, &slope) in slopes.iter().enumerate() {
        for i in 0..5 {
            let x = i as f64;
            y_vec.push(slope * x + noise.sample(&mut rng));
            x_vec.push(x);
            t_vec.push((period + 1) as f64);
        }
    }

    let mut data = IndexMap::new();
    data.insert("y".to_string(), Array1::from_vec(y_vec));
    data.insert("x".to_string(), Array1::from_vec(x_vec));
    data.insert("t".to_string(), Array1::from_vec(t_vec));
    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();

    let result = FamaMacBeth::fit(&formula, &df, "t", 0).unwrap();

    // Intercept should be close to 0 (no constant in the per-period DGP)
    approx_zero(result.params[0], 0.3);
    // Mean slope should be close to 2.0 (the average of 1, 2, 3).
    approx_zero(result.params[1] - 2.0, 0.15);
    assert_eq!(result.n_periods, 3);
    assert_eq!(result.n_obs_total, 15);

    // Standard errors are positive and finite when T > 1.
    for &se in result.std_errors.iter() {
        assert!(se > 0.0 && se.is_finite());
    }
}

/// Newey-West lags > 0 produce finite standard errors (variance adjusted).
#[test]
fn test_fama_macbeth_newey_west_finite() {
    let mut rng = StdRng::seed_from_u64(456);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut y_vec = Vec::new();
    let mut x_vec = Vec::new();
    let mut t_vec = Vec::new();
    for period in 0..4 {
        let slope = 1.0 + 0.1 * period as f64;
        for i in 0..5 {
            let x = i as f64;
            y_vec.push(slope * x + noise.sample(&mut rng));
            x_vec.push(x);
            t_vec.push((period + 1) as f64);
        }
    }

    let mut data = IndexMap::new();
    data.insert("y".to_string(), Array1::from_vec(y_vec));
    data.insert("x".to_string(), Array1::from_vec(x_vec));
    data.insert("t".to_string(), Array1::from_vec(t_vec));
    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();

    let result = FamaMacBeth::fit(&formula, &df, "t", 2).unwrap();
    assert!(result.params[1] > 0.0);
    for &se in result.std_errors.iter() {
        assert!(se > 0.0 && se.is_finite());
    }
}

/// Input validation.
#[test]
fn test_fama_macbeth_input_validation() {
    let mut data = IndexMap::new();
    data.insert("y".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
    data.insert("x".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
    data.insert("t".to_string(), Array1::from_vec(vec![1.0, 1.0, 1.0]));
    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();

    // Only one time period (after dropping <3 obs periods, there is none).
    assert!(FamaMacBeth::fit(&formula, &df, "t", 0).is_err());

    // Missing time column.
    assert!(FamaMacBeth::fit(&formula, &df, "missing", 0).is_err());
}
