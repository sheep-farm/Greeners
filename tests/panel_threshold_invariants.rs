use greeners::{FixedEffects, PanelThreshold};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Panel threshold regression recovers the true threshold and the two regime
/// slopes.
#[test]
fn test_panel_threshold_recovery() {
    let n_entities = 30;
    let t = 5;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(9101);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let gamma_true = 0.0;
    let beta1 = 1.0;
    let beta2 = 3.0;

    let mut x_vec = Vec::with_capacity(n);
    let mut q_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut entity_ids = Vec::with_capacity(n);
    for e in 0..n_entities {
        let entity_effect = noise.sample(&mut rng) * 2.0;
        for _ in 0..t {
            let x = noise.sample(&mut rng);
            let q = noise.sample(&mut rng);
            let y = if q <= gamma_true {
                beta1 * x + entity_effect + noise.sample(&mut rng)
            } else {
                beta2 * x + entity_effect + noise.sample(&mut rng)
            };
            x_vec.push(x);
            q_vec.push(q);
            y_vec.push(y);
            entity_ids.push(e as i64);
        }
    }

    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let q = Array1::from_vec(q_vec);
    let ids = Array1::from_vec(entity_ids);

    let result = PanelThreshold::fit(&y, &x, &q, &ids).unwrap();
    approx_zero((result.threshold_gamma - gamma_true).abs(), 0.3);
    approx_zero((result.params_regime1[0] - beta1).abs(), 0.3);
    approx_zero((result.params_regime2[0] - beta2).abs(), 0.3);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert!(result.ssr_min >= 0.0);
    assert!(result.n_search > 0);
}

/// Panel threshold reduces to fixed effects when the slope is the same in both
/// regimes.
#[test]
fn test_panel_threshold_single_regime() {
    let n_entities = 20;
    let t = 5;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(9102);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let beta = 2.0;

    let mut x_vec = Vec::with_capacity(n);
    let mut q_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut entity_ids = Vec::with_capacity(n);
    for e in 0..n_entities {
        let entity_effect = noise.sample(&mut rng);
        for _ in 0..t {
            let x = noise.sample(&mut rng);
            let q = noise.sample(&mut rng);
            let y = beta * x + entity_effect + noise.sample(&mut rng);
            x_vec.push(x);
            q_vec.push(q);
            y_vec.push(y);
            entity_ids.push(e as i64);
        }
    }

    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let q = Array1::from_vec(q_vec);
    let ids = Array1::from_vec(entity_ids);

    let result = PanelThreshold::fit(&y, &x, &q, &ids).unwrap();
    assert!(result.params_regime1[0].is_finite());
    assert!(result.params_regime2[0].is_finite());
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);

    // Compare with plain FE to ensure the best threshold model is at least as
    // good as the single-regime FE.
    let fe = FixedEffects::fit(&y, &x, ids.as_slice().unwrap()).unwrap();
    assert!(result.ssr_min <= fe.sigma.powi(2) * (fe.df_resid as f64) + 1e-6);
    approx_zero((result.r_squared - fe.r_squared).abs(), 0.05);
}

/// Input validation.
#[test]
fn test_panel_threshold_input_validation() {
    let y = Array1::from_vec(vec![1.0; 10]);
    let x = Array2::from_shape_vec((10, 1), vec![1.0; 10]).unwrap();
    let q_short = Array1::from_vec(vec![1.0; 5]);
    let ids = Array1::from_vec(vec![0i64; 10]);
    assert!(PanelThreshold::fit(&y, &x, &q_short, &ids).is_err());

    let q_const = Array1::from_vec(vec![1.0; 10]);
    assert!(PanelThreshold::fit(&y, &x, &q_const, &ids).is_err());
}
