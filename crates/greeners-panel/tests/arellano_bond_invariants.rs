use greeners_panel::dynamic_panel::ArellanoBond;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_panel(
    seed: u64,
    n_entities: usize,
    t: usize,
) -> (Array1<f64>, Array2<f64>, Vec<i64>, Vec<i64>) {
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n * 2);
    let mut entity_ids = Vec::with_capacity(n);
    let mut time_ids = Vec::with_capacity(n);

    for e in 0..n_entities {
        let alpha = noise.sample(&mut rng);
        let mut y_lag = 0.0;
        for tt in 0..t {
            let _i = e * t + tt;
            entity_ids.push(e as i64);
            time_ids.push(tt as i64);

            let x_val = 0.5 * tt as f64 + noise.sample(&mut rng);
            x.push(1.0); // intercept / constant column
            x.push(x_val);

            let y_val = if tt == 0 {
                1.0 + 2.0 * x_val + alpha + noise.sample(&mut rng)
            } else {
                0.5 * y_lag + 1.0 + 2.0 * x_val + alpha + noise.sample(&mut rng)
            };
            y.push(y_val);
            y_lag = y_val;
        }
    }

    (
        Array1::from_vec(y),
        Array2::from_shape_vec((n, 2), x).unwrap(),
        entity_ids,
        time_ids,
    )
}

#[test]
fn arellano_bond_runs_and_recovers_parameters() {
    let (y, x, entity_ids, time_ids) = make_panel(11, 40, 7);

    let result = ArellanoBond::fit(
        &y,
        &x,
        &entity_ids,
        &time_ids,
        2,
        false,
        Some(vec!["const".into(), "x".into()]),
    )
    .unwrap();

    assert!(result.params.iter().all(|v| v.is_finite()));
    assert_eq!(result.params.len(), 2);
    assert_eq!(result.std_errors.len(), 2);
    assert_eq!(result.t_values.len(), 2);
    assert_eq!(result.p_values.len(), 2);
    assert!(result.n_obs > 0);
    assert!(result.n_entities > 0);
    assert!(result.n_instruments >= result.params.len());
    assert!(result.sargan_stat.is_finite());
    assert!(result.m1_stat.is_finite());
    assert!(result.m2_stat.is_finite());

    // Approximate recovery: lag coefficient ~0.5, slope on x ~2.0
    assert!(
        (result.params[0] - 0.5).abs() < 0.4,
        "lag coef out of range: {}",
        result.params[0]
    );
    assert!(
        (result.params[1] - 2.0).abs() < 0.6,
        "x coef out of range: {}",
        result.params[1]
    );
}

#[test]
fn arellano_bond_two_step_produces_finite_output() {
    let (y, x, entity_ids, time_ids) = make_panel(22, 30, 6);

    let result = ArellanoBond::fit(
        &y,
        &x,
        &entity_ids,
        &time_ids,
        2,
        true,
        Some(vec!["const".into(), "x".into()]),
    )
    .unwrap();

    assert!(result.params.iter().all(|v| v.is_finite()));
    assert_eq!(result.params.len(), 2);
    assert_eq!(result.step, 2);
    assert!(result.sargan_pvalue >= 0.0 && result.sargan_pvalue <= 1.0);
}

#[test]
fn arellano_bond_input_validation() {
    let (y, x, entity_ids, time_ids) = make_panel(33, 10, 5);

    // max_lags must be >= 1
    assert!(ArellanoBond::fit(&y, &x, &entity_ids, &time_ids, 0, false, None).is_err());

    // Mismatched id lengths
    let short_ids = entity_ids[..entity_ids.len() - 1].to_vec();
    assert!(ArellanoBond::fit(&y, &x, &short_ids, &time_ids, 2, false, None).is_err());
}
