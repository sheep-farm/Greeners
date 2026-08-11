use greeners::PanelQuantile;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn panel_quantile_runs_and_recovers_parameters() {
    let n_entities = 20;
    let t = 5;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n * 2);
    let mut entity_ids = Vec::with_capacity(n);

    for e in 0..n_entities {
        let alpha = noise.sample(&mut rng) * 2.0;
        for _tt in 0..t {
            entity_ids.push(e as i64);

            let x1 = noise.sample(&mut rng);
            let x2 = noise.sample(&mut rng);
            x.push(x1);
            x.push(x2);

            let y_val = 1.0 + 2.0 * x1 - 0.5 * x2 + alpha + noise.sample(&mut rng);
            y.push(y_val);
        }
    }

    let y = Array1::from_vec(y);
    let x = Array2::from_shape_vec((n, 2), x).unwrap();

    let result = PanelQuantile::fit(
        &y,
        &x,
        &entity_ids,
        0.5,
        Some(vec!["x1".into(), "x2".into()]),
    )
    .unwrap();

    assert!(result.beta.iter().all(|v| v.is_finite()));
    assert_eq!(result.beta.len(), 2);
    assert_eq!(result.std_errors.len(), 2);
    assert!(result.n_obs == n);
    assert!(result.n_entities == n_entities);
    assert!(result.pseudo_r2 >= 0.0 && result.pseudo_r2 <= 1.0);

    // Approximate recovery of median coefficients.
    assert!(
        (result.beta[0] - 2.0).abs() < 0.6,
        "x1 coef out of range: {}",
        result.beta[0]
    );
    assert!(
        (result.beta[1] - (-0.5)).abs() < 0.6,
        "x2 coef out of range: {}",
        result.beta[1]
    );
}

#[test]
fn panel_quantile_input_validation() {
    let n = 20;
    let y = Array1::from_vec(vec![0.0; n]);
    let x = Array2::zeros((n, 1));
    let entity_ids = vec![0i64; n];

    // tau outside (0, 1)
    assert!(PanelQuantile::fit(&y, &x, &entity_ids, 0.0, None).is_err());
    assert!(PanelQuantile::fit(&y, &x, &entity_ids, 1.0, None).is_err());

    // Mismatched entity ids
    let short_ids = vec![0i64; n - 1];
    assert!(PanelQuantile::fit(&y, &x, &short_ids, 0.5, None).is_err());
}
