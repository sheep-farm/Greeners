use greeners::{GlsPanels, PanelGLS, OLS};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_balanced_panel(
    seed: u64,
    n_entities: usize,
    t: usize,
) -> (Array1<f64>, Array2<f64>, Vec<i64>, Vec<i64>) {
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n * 2);
    let mut entity_ids = Vec::with_capacity(n);
    let mut time_ids = Vec::with_capacity(n);

    for e in 0..n_entities {
        for tt in 0..t {
            entity_ids.push(e as i64);
            time_ids.push(tt as i64);
            let x2 = 0.2 * (e as f64) + 0.1 * (tt as f64) + noise.sample(&mut rng);
            x.push(1.0);
            x.push(x2);
            y.push(1.0 + 2.0 * x2 + noise.sample(&mut rng));
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
fn panel_gls_hetero_close_to_ols() {
    let (y, x, entity_ids, time_ids) = make_balanced_panel(11, 5, 10);

    let ols = OLS::fit(&y, &x, greeners::CovarianceType::NonRobust).unwrap();
    let gls = PanelGLS::fit(&y, &x, &entity_ids, &time_ids, GlsPanels::Hetero, None).unwrap();

    assert_eq!(gls.params.len(), 2);
    assert!(gls.params.iter().all(|v| v.is_finite()));
    assert!(gls.r_squared >= 0.0 && gls.r_squared <= 1.0);

    for j in 0..2 {
        assert!(
            (gls.params[j] - ols.params[j]).abs() < 0.5,
            "param {} diff too large",
            j
        );
    }
}

#[test]
fn panel_gls_correlated_finite() {
    let (y, x, entity_ids, time_ids) = make_balanced_panel(22, 4, 8);

    let gls = PanelGLS::fit(&y, &x, &entity_ids, &time_ids, GlsPanels::Correlated, None).unwrap();

    assert_eq!(gls.params.len(), 2);
    assert!(gls.params.iter().all(|v| v.is_finite()));
    assert!(
        (gls.params[1] - 2.0).abs() < 0.5,
        "slope out of range: {}",
        gls.params[1]
    );
    assert!(gls.r_squared >= 0.0 && gls.r_squared <= 1.0);
    assert!(gls.n_entities == 4);
    assert!(gls.t_periods == 8);
}

#[test]
fn panel_gls_input_validation() {
    let (y, x, entity_ids, time_ids) = make_balanced_panel(33, 5, 6);
    let short_ids = time_ids[..time_ids.len() - 1].to_vec();

    assert!(PanelGLS::fit(&y, &x, &entity_ids, &short_ids, GlsPanels::Hetero, None).is_err());
}
