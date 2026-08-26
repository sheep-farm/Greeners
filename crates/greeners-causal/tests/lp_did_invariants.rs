use greeners_causal::lp_did::LpDid;
use greeners_core::column::Column;
use greeners_core::dataframe::DataFrame;
use indexmap::IndexMap;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_lp_did_data(seed: u64, n: usize, t: usize, treat_at: i64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y = Vec::with_capacity(n * t);
    let mut unit = Vec::with_capacity(n * t);
    let mut time = Vec::with_capacity(n * t);
    let mut first_treat = Vec::with_capacity(n * t);

    for u in 0..n {
        let treat = u < n / 2;
        let ft = if treat { treat_at } else { 0 };
        let unit_fe = noise.sample(&mut rng) * 2.0;
        for tt in 0..t {
            unit.push(format!("unit_{u}"));
            time.push(tt as i64);
            first_treat.push(ft);
            let post = if treat && (tt as i64) >= treat_at {
                1.0
            } else {
                0.0
            };
            let y_val = 1.0 + 0.2 * (tt as f64) + unit_fe + 1.5 * post + noise.sample(&mut rng);
            y.push(y_val);
        }
    }

    let mut columns: IndexMap<String, Column> = IndexMap::new();
    columns.insert("y".into(), Column::Float(Array1::from_vec(y)));
    columns.insert("unit".into(), Column::String(Array1::from_vec(unit)));
    columns.insert("time".into(), Column::Int(Array1::from_vec(time)));
    columns.insert(
        "first_treat".into(),
        Column::Int(Array1::from_vec(first_treat)),
    );

    DataFrame::from_columns(columns).unwrap()
}

#[test]
fn lp_did_runs_and_produces_horizons() {
    let df = make_lp_did_data(11, 40, 8, 4);

    let result = LpDid::new()
        .fit(&df, "y", "unit", "time", Some("first_treat"), None, None)
        .unwrap();

    assert!(result.n_obs > 0);
    assert!(result.n_treated_units > 0);
    assert!(result.n_control_units > 0);
    assert!(!result.horizons.is_empty());
    assert_eq!(result.estimates.len(), result.horizons.len());
    assert_eq!(result.standard_errors.len(), result.horizons.len());
    assert!(result.estimates.iter().any(|v| v.is_finite()));
    assert!(result.max_post > 0);
}

#[test]
fn lp_did_post_effects_positive() {
    let df = make_lp_did_data(22, 80, 8, 4);

    let result = LpDid::new()
        .with_target_estimand("ra")
        .fit(&df, "y", "unit", "time", Some("first_treat"), None, None)
        .unwrap();

    let post_effects: Vec<f64> = result
        .horizons
        .iter()
        .zip(result.estimates.iter())
        .filter(|(h, v)| **h >= 0 && v.is_finite())
        .map(|(_, v)| *v)
        .collect();

    if !post_effects.is_empty() {
        let mean = post_effects.iter().sum::<f64>() / post_effects.len() as f64;
        assert!(
            mean > 0.0,
            "mean post effect should be positive, got {}",
            mean
        );
    }
}

#[test]
fn lp_did_input_validation() {
    let n = 20;
    let t = 6;
    let df = make_lp_did_data(33, n, t, 3);

    // Neither first_treat nor treatment column provided
    assert!(LpDid::new()
        .fit(&df, "y", "unit", "time", None, None, None)
        .is_err());

    // Missing outcome column
    assert!(LpDid::new()
        .fit(
            &df,
            "missing",
            "unit",
            "time",
            Some("first_treat"),
            None,
            None
        )
        .is_err());
}
