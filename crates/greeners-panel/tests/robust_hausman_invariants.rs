use greeners_panel::panel::FixedEffects;
use greeners_panel::panel::RandomEffects;
use greeners_panel::panel_robust::RobustHausman;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_panel(seed: u64, n_entities: usize, t: usize) -> (Array1<f64>, Array2<f64>, Vec<i64>) {
    let n = n_entities * t;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut ids = Vec::with_capacity(n);

    for e in 0..n_entities {
        let alpha = noise.sample(&mut rng) * 1.5;
        for tt in 0..t {
            ids.push(e as i64);
            let x_val = 0.5 * (e as f64) + 0.1 * (tt as f64) + noise.sample(&mut rng);
            x.push(x_val);
            y.push(1.0 + 2.0 * x_val + alpha + noise.sample(&mut rng));
        }
    }

    (
        Array1::from_vec(y),
        Array2::from_shape_vec((n, 1), x).unwrap(),
        ids,
    )
}

fn make_vcov(se: &Array1<f64>, scale: f64) -> Array2<f64> {
    let k = se.len();
    let mut m = Array2::zeros((k, k));
    for i in 0..k {
        m[(i, i)] = se[i].powi(2) * scale;
    }
    m
}

#[test]
fn robust_hausman_classical_runs() {
    let (y, x, ids) = make_panel(11, 20, 5);
    let entity_ids = Array1::from_vec(ids.clone());

    let fe = FixedEffects::fit(&y, &x, &ids).unwrap();
    let re = RandomEffects::fit(&y, &x, &entity_ids).unwrap();

    let result = RobustHausman::classical(&fe, &re).unwrap();

    assert!(result.chi2.is_finite() && result.chi2 >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert_eq!(result.df, fe.params.len());
    assert_eq!(result.beta_diff.len(), fe.params.len());
    assert!(!result.recommendation.is_empty());
}

#[test]
fn robust_hausman_compare_arrays_runs() {
    let (y, x, ids) = make_panel(22, 20, 5);
    let entity_ids = Array1::from_vec(ids);

    let fe = FixedEffects::fit(&y, &x, &entity_ids.to_vec()).unwrap();
    let re = RandomEffects::fit(&y, &x, &entity_ids).unwrap();

    let fe_vcov = make_vcov(&fe.std_errors, 1.5);
    let re_vcov = make_vcov(&re.std_errors, 0.8);

    let result =
        RobustHausman::compare_arrays(&fe.params, &re.params, &fe_vcov, &re_vcov, None).unwrap();

    assert!(result.chi2.is_finite() && result.chi2 >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert_eq!(result.n_coef, fe.params.len());
    assert_eq!(result.method, "robust");
}

#[test]
fn robust_hausman_compare_runs() {
    let (y, x, ids) = make_panel(33, 20, 5);
    let entity_ids = Array1::from_vec(ids);

    let fe = FixedEffects::fit(&y, &x, &entity_ids.to_vec()).unwrap();
    let re = RandomEffects::fit(&y, &x, &entity_ids).unwrap();

    let fe_vcov = make_vcov(&fe.std_errors, 1.5);
    let re_vcov = make_vcov(&re.std_errors, 0.8);

    let result = RobustHausman::compare(&fe, &re, &fe_vcov, &re_vcov).unwrap();

    assert!(result.chi2.is_finite() && result.chi2 >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert_eq!(result.n_coef, fe.params.len());
}

#[test]
fn robust_hausman_input_validation() {
    let k = 2;
    let fe_beta = Array1::from_vec(vec![1.0, 2.0]);
    let re_beta = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let fe_vcov = Array2::from_shape_vec((k, k), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let re_vcov = Array2::from_shape_vec((k, k), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    assert!(RobustHausman::compare_arrays(&fe_beta, &re_beta, &fe_vcov, &re_vcov, None).is_err());
}
