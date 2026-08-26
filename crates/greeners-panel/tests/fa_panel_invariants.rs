use greeners_panel::fa_panel::FAPanel;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn fa_panel_runs_and_recovers_parameters() {
    let n_entities = 20;
    let t = 10;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n * 2);
    let mut entity_ids = Vec::with_capacity(n);
    let mut period_ids = Vec::with_capacity(n);

    // Two common factors
    let f1: Vec<f64> = (0..t).map(|tt| (tt as f64) * 0.1).collect();
    let f2: Vec<f64> = (0..t).map(|tt| ((t - tt) as f64) * 0.1).collect();

    for e in 0..n_entities {
        let alpha = noise.sample(&mut rng);
        for tt in 0..t {
            entity_ids.push(e as i64);
            period_ids.push(tt as i64);

            let x_val = 0.5 * tt as f64 + noise.sample(&mut rng);
            x.push(1.0);
            x.push(x_val);

            let y_val =
                1.0 + 2.0 * x_val + 0.5 * f1[tt] - 0.5 * f2[tt] + alpha + noise.sample(&mut rng);
            y.push(y_val);
        }
    }

    let y = Array1::from_vec(y);
    let x = Array2::from_shape_vec((n, 2), x).unwrap();

    // Auxiliary panel: T x n_aux with the factors plus noise
    let mut aux = Array2::zeros((t, 5));
    for tt in 0..t {
        aux[(tt, 0)] = f1[tt] + noise.sample(&mut rng);
        aux[(tt, 1)] = f2[tt] + noise.sample(&mut rng);
        for j in 2..5 {
            aux[(tt, j)] = noise.sample(&mut rng);
        }
    }

    let result = FAPanel::fit(
        &y,
        &x,
        &aux,
        &entity_ids,
        &period_ids,
        2,
        Some(vec!["const".into(), "x".into()]),
    )
    .unwrap();

    assert!(result.beta.iter().all(|v| v.is_finite()));
    assert!(result.gamma.iter().all(|v| v.is_finite()));
    assert_eq!(result.beta.len(), 2);
    assert_eq!(result.gamma.len(), 2);
    assert_eq!(result.factors.nrows(), t);
    assert_eq!(result.factors.ncols(), 2);
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_entities, n_entities);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);

    // The slope on x should be recovered approximately.
    assert!(
        (result.beta[1] - 2.0).abs() < 0.6,
        "x coef out of range: {}",
        result.beta[1]
    );
}

#[test]
fn fa_panel_input_validation() {
    let n = 60;
    let y = Array1::from_vec(vec![0.0; n]);
    let x = Array2::zeros((n, 1));
    let entity_ids = vec![0i64; n];
    let period_ids = vec![0i64; n];
    let aux = Array2::zeros((3, 5));

    // n_factors == 0 is invalid
    assert!(FAPanel::fit(&y, &x, &aux, &entity_ids, &period_ids, 0, None).is_err());

    // n_factors >= n_aux is invalid
    assert!(FAPanel::fit(&y, &x, &aux, &entity_ids, &period_ids, 5, None).is_err());

    // Mismatched dimensions
    let short_period = vec![0i64; n - 1];
    assert!(FAPanel::fit(&y, &x, &aux, &entity_ids, &short_period, 2, None).is_err());
}
