use greeners::{OLS, PCSE};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

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
fn pcse_equals_ols_homoskedastic() {
    let (y, x, entity_ids, time_ids) = make_balanced_panel(11, 5, 10);

    let ols = OLS::fit(&y, &x, greeners::CovarianceType::NonRobust).unwrap();
    let pcse = PCSE::fit(&y, &x, &entity_ids, &time_ids, None).unwrap();

    assert_eq!(pcse.params.len(), 2);
    assert!(pcse.params.iter().all(|v| v.is_finite()));
    assert!(pcse.r_squared >= 0.0 && pcse.r_squared <= 1.0);

    for j in 0..2 {
        assert!(
            (pcse.params[j] - ols.params[j]).abs() < 1e-6,
            "param {} diff too large",
            j
        );
    }
    assert!(pcse.n_entities == 5);
    assert!(pcse.t_periods == 10);
}

#[test]
fn pcse_rejects_non_finite() {
    let mut y = Array1::from_vec(vec![1.0; 10]);
    y[0] = f64::NAN;
    let x = Array2::from_shape_vec((10, 1), vec![1.0; 10]).unwrap();
    let ids = vec![0i64; 10];
    let times = vec![0i64; 10];

    assert!(PCSE::fit(&y, &x, &ids, &times, None).is_err());
}

#[test]
fn pcse_input_validation() {
    let (y, x, entity_ids, time_ids) = make_balanced_panel(22, 4, 8);
    let short_times = time_ids[..time_ids.len() - 1].to_vec();
    assert!(PCSE::fit(&y, &x, &entity_ids, &short_times, None).is_err());
}
