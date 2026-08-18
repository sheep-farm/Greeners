use greeners::{Column, DataFrame, SyntheticControl};
use indexmap::IndexMap;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_sc_data(seed: u64, n_units: usize, t: usize) -> DataFrame {
    let n = n_units * t;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut unit = Vec::with_capacity(n);
    let mut time = Vec::with_capacity(n);

    for u in 0..n_units {
        let unit_fe = noise.sample(&mut rng) * 2.0;
        for tt in 0..t {
            unit.push(format!("unit_{u}"));
            time.push(tt as f64);
            let y_val = 1.0 + 0.2 * (tt as f64) + unit_fe + noise.sample(&mut rng);
            y.push(y_val);
        }
    }

    let mut columns: IndexMap<String, Column> = IndexMap::new();
    columns.insert("y".into(), Column::Float(Array1::from_vec(y)));
    columns.insert("time".into(), Column::Float(Array1::from_vec(time)));
    columns.insert("unit".into(), Column::String(Array1::from_vec(unit)));

    DataFrame::from_columns(columns).unwrap()
}

#[test]
fn synthetic_control_runs_and_produces_weights() {
    let df = make_sc_data(11, 10, 8);

    let result = SyntheticControl::fit("y", "unit_0", 5.0, &df, "unit", "time", None).unwrap();

    assert_eq!(result.synthetic_series.len(), 8);
    assert_eq!(result.actual_series.len(), 8);
    assert_eq!(result.time_index.len(), 8);
    assert!(result.rmspe_pre >= 0.0);
    assert!(result
        .weights
        .iter()
        .all(|(_, w)| w.is_finite() && *w >= -1e-9));
    let wsum: f64 = result.weights.iter().map(|(_, w)| w).sum();
    assert!(
        (wsum - 1.0).abs() < 1e-6,
        "weights should sum to 1, got {}",
        wsum
    );
    assert_eq!(result.treated_unit, "unit_0");
    assert_eq!(result.t_pre, 5);
    assert_eq!(result.t_post, 3);
}

#[test]
fn synthetic_control_rejects_invalid_treated_unit() {
    let df = make_sc_data(22, 5, 6);

    assert!(SyntheticControl::fit("y", "missing_unit", 3.0, &df, "unit", "time", None).is_err());
}

#[test]
fn synthetic_control_rejects_too_few_pre_periods() {
    let df = make_sc_data(33, 5, 6);

    // t0 is the first post period; 0.0 gives 0 pre periods
    assert!(SyntheticControl::fit("y", "unit_0", 0.0, &df, "unit", "time", None).is_err());
}
