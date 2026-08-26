use greeners_panel::hausman::HausmanTest;
use greeners_panel::panel::FixedEffects;
use greeners_panel::panel::RandomEffects;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

/// Hausman test rejects Random Effects when the entity effects are correlated
/// with the regressor.
#[test]
fn test_hausman_rejects_when_re_inconsistent() {
    let n_entities = 30;
    let t = 5;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(9001);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);
    for e in 0..n_entities {
        // Entity effect that is correlated with the entity mean of x
        let entity_effect = e as f64 * 0.3;
        for _ in 0..t {
            let x = e as f64 + noise.sample(&mut rng);
            x_vec.push(x);
            y_vec.push(1.0 + 2.0 * x + entity_effect + noise.sample(&mut rng));
            groups.push(e as i64);
        }
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let entity_ids = Array1::from_vec(groups.clone());

    let fe = FixedEffects::fit(&y, &x, &groups).unwrap();
    let re = RandomEffects::fit(&y, &x, &entity_ids).unwrap();
    let output = HausmanTest::compare(&fe, &re);

    assert!(output.contains("Chi2 Statistic"));
    assert!(output.contains("P-Value"));
    assert!(output.contains("Reject H0"));
    assert!(output.contains("FIXED EFFECTS"));
}

/// Hausman test fails to reject Random Effects when the entity effects are
/// independent of the regressor.
#[test]
fn test_hausman_fails_to_reject_when_re_consistent() {
    let n_entities = 30;
    let t = 5;
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(9002);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);
    for e in 0..n_entities {
        // Entity effect independent of x
        let entity_effect = noise.sample(&mut rng);
        for _ in 0..t {
            let x = noise.sample(&mut rng);
            x_vec.push(x);
            y_vec.push(1.0 + 2.0 * x + entity_effect + noise.sample(&mut rng));
            groups.push(e as i64);
        }
    }
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let entity_ids = Array1::from_vec(groups.clone());

    let fe = FixedEffects::fit(&y, &x, &groups).unwrap();
    let re = RandomEffects::fit(&y, &x, &entity_ids).unwrap();
    let output = HausmanTest::compare(&fe, &re);

    assert!(output.contains("P-Value"));
    assert!(output.contains("Fail to reject H0"));
    assert!(output.contains("RANDOM EFFECTS"));
}
