use greeners_panel::dynamic_panel::ArellanoBond;
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Two entities, 5 time periods each. First-difference regression of
/// Δy_t on Δy_{t-1} with instrument y_{t-2}. The IV estimator is
/// β_IV = (Σ z Δy_t) / (Σ z Δy_{t-1}).
#[test]
fn test_arellano_bond_just_identified_equals_iv() {
    // Entity 1
    let y1 = vec![1.0, 2.0, 4.0, 7.0, 11.0];
    // Entity 2
    let y2 = vec![1.0, 3.0, 5.0, 9.0, 14.0];

    let mut y = Vec::new();
    let mut entity_ids = Vec::new();
    let mut time_ids = Vec::new();

    for (e, vals) in [y1, y2].iter().enumerate() {
        for (t, &v) in vals.iter().enumerate() {
            y.push(v);
            entity_ids.push((e + 1) as i64);
            time_ids.push((t + 1) as i64);
        }
    }

    let y = Array1::from(y);
    let x = Array2::zeros((y.len(), 0));

    // Manual IV
    let mut z_dy = 0.0;
    let mut z_dyl = 0.0;
    for vals in [
        vec![1.0, 2.0, 4.0, 7.0, 11.0],
        vec![1.0, 3.0, 5.0, 9.0, 14.0],
    ]
    .iter()
    {
        for t in 2..vals.len() {
            let dy = vals[t] - vals[t - 1];
            let dyl = vals[t - 1] - vals[t - 2];
            let z = vals[t - 2];
            z_dy += z * dy;
            z_dyl += z * dyl;
        }
    }
    let iv_beta = z_dy / z_dyl;

    // Arellano-Bond with max_lags = 1 (one instrument y_{t-2})
    let result = ArellanoBond::fit(&y, &x, &entity_ids, &time_ids, 1, false, None).unwrap();

    assert_eq!(result.n_entities, 2);
    assert_eq!(result.n_obs, 6); // (5-2) * 2
    assert_eq!(result.n_instruments, 1);
    assert_eq!(result.params.len(), 1);
    approx_zero((result.params[0] - iv_beta).abs(), 1e-10);
}

/// With max_lags > 1 the model becomes over-identified, so the Sargan test
/// has positive degrees of freedom and returns a valid statistic.
#[test]
fn test_arellano_bond_overidentified_sargan_positive_df() {
    // Entity 1
    let y1 = vec![1.0, 2.0, 4.0, 7.0, 11.0];
    // Entity 2
    let y2 = vec![1.0, 3.0, 5.0, 9.0, 14.0];

    let mut y = Vec::new();
    let mut entity_ids = Vec::new();
    let mut time_ids = Vec::new();

    for (e, vals) in [y1, y2].iter().enumerate() {
        for (t, &v) in vals.iter().enumerate() {
            y.push(v);
            entity_ids.push((e + 1) as i64);
            time_ids.push((t + 1) as i64);
        }
    }

    let y = Array1::from(y);
    let x = Array2::zeros((y.len(), 0));

    let result = ArellanoBond::fit(&y, &x, &entity_ids, &time_ids, 2, false, None).unwrap();

    // One regressor (LD.y), two instruments (y_{t-2}, y_{t-3}): df = 2 - 1 = 1.
    assert_eq!(result.sargan_df, 1);
    assert!(result.sargan_stat.is_finite());
    assert!((0.0..=1.0).contains(&result.sargan_pvalue));
}
