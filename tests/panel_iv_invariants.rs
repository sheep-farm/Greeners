use greeners::FE2SLS;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// FE-2SLS recovers the structural coefficients when using a valid
/// instrument for the endogenous regressor.
#[test]
fn test_fe2sls_recovery() {
    let n_entities = 20;
    let t = 5;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(8001);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let b1 = 1.0; // coefficient on exogenous x1
    let b2 = 2.0; // coefficient on endogenous x2

    let mut y_vec = Vec::with_capacity(n);
    let mut x_vec = Vec::with_capacity(n * 2);
    let mut z_vec = Vec::with_capacity(n * 2);
    let mut groups = Vec::with_capacity(n);

    for e in 0..n_entities {
        let entity_effect = noise.sample(&mut rng);
        for _ in 0..t {
            let z = noise.sample(&mut rng);
            let x1 = noise.sample(&mut rng);
            // Endogenous x2: depends on z (instrument), x1 and an error v
            let v = noise.sample(&mut rng);
            let x2 = 0.5 + 1.5 * z + 0.5 * x1 + v;
            // y depends on x1, x2, the entity effect and an error u
            let u = noise.sample(&mut rng);
            let y = 0.5 + b1 * x1 + b2 * x2 + entity_effect + u;

            groups.push(e as i64);
            y_vec.push(y);
            x_vec.push(x1);
            x_vec.push(x2);
            z_vec.push(x1); // x1 is its own instrument
            z_vec.push(z); // excluded instrument
        }
    }

    let y = Array1::from_vec(y_vec);
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let z = Array2::from_shape_vec((n, 2), z_vec).unwrap();

    let result = FE2SLS::fit(&y, &x, &z, &groups, None).unwrap();
    approx_zero((result.params[0] - b1).abs(), 0.2);
    approx_zero((result.params[1] - b2).abs(), 0.2);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert_eq!(result.n_entities, n_entities);
    assert!(result.sigma > 0.0);
    assert!(result.df_resid > 0);
    assert!(result.params.iter().all(|&v| v.is_finite()));
}

/// FE-2SLS rejects invalid inputs.
#[test]
fn test_fe2sls_input_validation() {
    let y = Array1::from_vec(vec![1.0; 10]);
    let x = Array2::from_shape_vec((10, 2), vec![1.0; 20]).unwrap();
    let z_short = Array2::from_shape_vec((10, 1), vec![1.0; 10]).unwrap();
    let groups: Vec<i64> = (0..10).map(|i| (i / 2) as i64).collect();

    assert!(FE2SLS::fit(&y, &x, &z_short, &groups, None).is_err());

    let z_nan = Array2::from_shape_vec((10, 2), vec![f64::NAN; 20]).unwrap();
    assert!(FE2SLS::fit(&y, &x, &z_nan, &groups, None).is_err());

    let groups_short: Vec<i64> = vec![0; 5];
    assert!(FE2SLS::fit(&y, &x, &x, &groups_short, None).is_err());
}
