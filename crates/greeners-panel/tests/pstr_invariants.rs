use greeners_panel::pstr::PSTR;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

#[test]
fn pstr_runs_and_recovers_parameters() {
    let n_entities = 25;
    let t = 8;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 0.4).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n * 2);
    let mut q = Vec::with_capacity(n);
    let mut entity_ids = Vec::with_capacity(n);

    for e in 0..n_entities {
        let alpha = noise.sample(&mut rng) * 2.0;
        let q_shift = noise.sample(&mut rng);
        for _tt in 0..t {
            entity_ids.push(e as i64);

            let x1 = noise.sample(&mut rng);
            let x2 = noise.sample(&mut rng);
            x.push(x1);
            x.push(x2);

            let q_val = x1 + q_shift;
            q.push(q_val);

            let g = 1.0 / (1.0 + (-2.0_f64 * q_val).exp());
            let y_val =
                alpha + 1.0 * x1 + 0.5 * x2 + (2.0 * x1 + 1.0 * x2) * g + noise.sample(&mut rng);
            y.push(y_val);
        }
    }

    let y = Array1::from_vec(y);
    let x = Array2::from_shape_vec((n, 2), x).unwrap();
    let q = Array1::from_vec(q);

    let result = PSTR::fit(
        &y,
        &x,
        &q,
        &entity_ids,
        Some(vec!["x1".into(), "x2".into()]),
    )
    .unwrap();

    assert!(result.beta0.iter().all(|v| v.is_finite()));
    assert!(result.beta1.iter().all(|v| v.is_finite()));
    assert_eq!(result.beta0.len(), 2);
    assert_eq!(result.beta1.len(), 2);
    assert!(result.gamma > 0.0);
    assert!(result.c.is_finite());
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert!(result.n_obs == n);
    assert!(result.n_entities == n_entities);

    // The combined coefficient on x1 should be in [1, 3] and positive.
    let b1_total = result.beta0[0] + result.beta1[0];
    assert!(
        b1_total > 0.5 && b1_total < 4.0,
        "combined x1 coef out of range: {}",
        b1_total
    );
}

#[test]
fn pstr_input_validation() {
    let n = 20;
    let y = Array1::from_vec(vec![0.0; n]);
    let x = Array2::zeros((n, 1));
    let q_short = Array1::from_vec(vec![0.0; n - 1]);
    let entity_ids = vec![0i64; n];

    assert!(PSTR::fit(&y, &x, &q_short, &entity_ids, None).is_err());

    let x_short = Array2::zeros((n - 1, 1));
    assert!(PSTR::fit(
        &y,
        &x_short,
        &Array1::from_vec(vec![0.0; n - 1]),
        &entity_ids,
        None
    )
    .is_err());
}
