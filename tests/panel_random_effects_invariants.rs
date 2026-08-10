use greeners::{FixedEffects, RandomEffects, OLS};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// With no entity-specific effects, Random Effects is close to pooled OLS.
#[test]
fn test_random_effects_no_entity_effect_equals_ols() {
    let n_entities = 20;
    let t_per_entity = 10;
    let n = n_entities * t_per_entity;
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    let mut ids = Vec::with_capacity(n);
    let mut id = 0;
    for i in 0..n {
        if i % t_per_entity == 0 && i > 0 {
            id += 1;
        }
        ids.push(id);
        let x2 = i as f64 / 50.0;
        x_vec.push(1.0);
        x_vec.push(x2);
        y_vec.push(1.0 + 2.0 * x2 + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let entity_ids = Array1::from_vec(ids.iter().map(|&v| v as i64).collect());

    let ols = OLS::fit(&y, &x, greeners::CovarianceType::NonRobust).unwrap();
    let re = RandomEffects::fit(&y, &x, &entity_ids).unwrap();

    for j in 0..ols.params.len() {
        approx_zero((re.params[j] - ols.params[j]).abs(), 0.1);
    }
    assert!(re.theta >= 0.0 && re.theta <= 1.0);
    // No entity effect -> theta should be small.
    assert!(re.theta < 0.2, "theta too large: {}", re.theta);
}

/// With strong entity-specific effects, the slope in RE is close to the
/// within (Fixed Effects) slope.
#[test]
fn test_random_effects_slope_matches_fixed_effects() {
    let n_entities = 20;
    let t_per_entity = 10;
    let n = n_entities * t_per_entity;
    let mut rng = StdRng::seed_from_u64(22);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let entity_noise = Normal::new(0.0, 3.0).unwrap();

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut ids = Vec::with_capacity(n);
    let mut id = 0;
    let mut alpha = 0.0;
    for i in 0..n {
        if i % t_per_entity == 0 && i > 0 {
            id += 1;
            alpha = entity_noise.sample(&mut rng);
        } else if i == 0 {
            alpha = entity_noise.sample(&mut rng);
        }
        ids.push(id);
        let x2 = (i % t_per_entity) as f64 + 0.5 * (id as f64);
        x_vec.push(x2);
        y_vec.push(1.0 + 2.0 * x2 + alpha + noise.sample(&mut rng));
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let entity_ids = Array1::from_vec(ids.iter().map(|&v| v as i64).collect());

    let re = RandomEffects::fit(&y, &x, &entity_ids).unwrap();
    let fe = FixedEffects::fit(&y, &x, &ids[..]).unwrap();

    approx_zero((re.params[0] - fe.params[0]).abs(), 0.2);
    assert!(re.theta > 0.5, "theta too small: {}", re.theta);
}

/// Input validation.
#[test]
fn test_random_effects_input_validation() {
    let y = Array1::from_vec(vec![1.0; 10]);
    let x = Array2::from_shape_vec((10, 1), vec![1.0; 10]).unwrap();
    let ids_short = Array1::from_vec(vec![0i64; 5]);
    assert!(RandomEffects::fit(&y, &x, &ids_short).is_err());
}
