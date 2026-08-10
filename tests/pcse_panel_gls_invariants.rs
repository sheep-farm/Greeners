use greeners::{GlsPanels, PanelGLS, OLS, PCSE};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// PCSE point estimates coincide with pooled OLS on a balanced panel with
/// homoskedastic errors.
#[test]
fn test_pcse_equals_ols_homoskedastic() {
    let n_entities = 5;
    let t = 10;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(77);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut entity_ids = Vec::with_capacity(n);
    let mut time_ids = Vec::with_capacity(n);
    for i in 0..n {
        let e = i / t;
        let tt = i % t;
        entity_ids.push(e as i64);
        time_ids.push(tt as i64);
        let x2 = i as f64 / 20.0;
        x_vec.push(1.0);
        x_vec.push(x2);
        y_vec.push(1.0 + 2.0 * x2 + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let ols = OLS::fit(&y, &x, greeners::CovarianceType::NonRobust).unwrap();
    let pcse = PCSE::fit(&y, &x, &entity_ids, &time_ids, None).unwrap();

    for j in 0..ols.params.len() {
        approx_zero((pcse.params[j] - ols.params[j]).abs(), 1e-6);
    }
    assert!(pcse.r_squared >= 0.0 && pcse.r_squared <= 1.0);
    assert!(pcse.n_entities == n_entities);
    assert!(pcse.t_periods == t);
}

/// PanelGLS with panels=Hetero on a balanced panel with equal variances is
/// close to pooled OLS.
#[test]
fn test_panel_gls_hetero_equals_ols() {
    let n_entities = 5;
    let t = 10;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(88);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut entity_ids = Vec::with_capacity(n);
    let mut time_ids = Vec::with_capacity(n);
    for i in 0..n {
        let e = i / t;
        let tt = i % t;
        entity_ids.push(e as i64);
        time_ids.push(tt as i64);
        let x2 = i as f64 / 20.0;
        x_vec.push(1.0);
        x_vec.push(x2);
        y_vec.push(1.0 + 2.0 * x2 + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let ols = OLS::fit(&y, &x, greeners::CovarianceType::NonRobust).unwrap();
    let gls = PanelGLS::fit(&y, &x, &entity_ids, &time_ids, GlsPanels::Hetero, None).unwrap();

    for j in 0..ols.params.len() {
        approx_zero((gls.params[j] - ols.params[j]).abs(), 0.15);
    }
    assert!(gls.r_squared >= 0.0 && gls.r_squared <= 1.0);
}

/// PanelGLS with panels=Correlated returns finite coefficients on a balanced
/// panel.
#[test]
fn test_panel_gls_correlated_converges() {
    let n_entities = 4;
    let t = 8;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(99);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut entity_ids = Vec::with_capacity(n);
    let mut time_ids = Vec::with_capacity(n);
    for i in 0..n {
        let e = i / t;
        let tt = i % t;
        entity_ids.push(e as i64);
        time_ids.push(tt as i64);
        let x2 = i as f64 / 20.0;
        x_vec.push(1.0);
        x_vec.push(x2);
        y_vec.push(1.0 + 2.0 * x2 + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let gls = PanelGLS::fit(&y, &x, &entity_ids, &time_ids, GlsPanels::Correlated, None).unwrap();
    approx_zero((gls.params[1] - 2.0).abs(), 0.4);
    assert!(gls.r_squared >= 0.0 && gls.r_squared <= 1.0);
}

/// Input validation.
#[test]
fn test_pcse_panel_gls_input_validation() {
    let y = Array1::from_vec(vec![1.0; 10]);
    let x = Array2::from_shape_vec((10, 1), vec![1.0; 10]).unwrap();
    let entity_ids = vec![0i64; 10];
    let time_ids_short = vec![0i64; 5];

    assert!(PCSE::fit(&y, &x, &entity_ids, &time_ids_short, None).is_err());
    assert!(PanelGLS::fit(
        &y,
        &x,
        &entity_ids,
        &time_ids_short,
        GlsPanels::Hetero,
        None
    )
    .is_err());
}
