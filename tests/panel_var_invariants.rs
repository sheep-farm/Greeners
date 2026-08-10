use greeners::PanelVAR;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn generate_panel_var(
    a: &Array2<f64>,
    n_entities: usize,
    t_per_entity: usize,
    noise_sd: f64,
    seed: u64,
) -> (Array2<f64>, Vec<i64>) {
    let k = a.ncols();
    let mut data = Array2::zeros((n_entities * t_per_entity, k));
    let mut entity_ids = Vec::with_capacity(n_entities * t_per_entity);

    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, noise_sd).unwrap();

    for e in 0..n_entities {
        let mut y_prev = Array1::zeros(k);
        for t in 0..t_per_entity {
            let row = e * t_per_entity + t;
            entity_ids.push(e as i64);
            let y_t = a.dot(&y_prev);
            data.row_mut(row).assign(&y_t);
            for j in 0..k {
                data[[row, j]] += noise.sample(&mut rng);
            }
            y_prev.assign(&data.row(row));
        }
    }
    (data, entity_ids)
}

#[test]
fn test_panel_var_runs_and_produces_finite_output() {
    let a = Array2::from_shape_vec((2, 2), vec![0.4, 0.1, 0.1, 0.3]).unwrap();
    let (data, entity_ids) = generate_panel_var(&a, 8, 15, 0.2, 7001);

    let result = PanelVAR::fit(&data, &entity_ids, 1, None).unwrap();

    assert_eq!(result.n_entities, 8);
    assert_eq!(result.n_vars, 2);
    assert_eq!(result.lags, 1);
    assert_eq!(result.coeffs.shape(), &[2, 2]);
    assert_eq!(result.std_errors.shape(), &[2, 2]);
    assert_eq!(result.t_values.shape(), &[2, 2]);
    assert_eq!(result.p_values.shape(), &[2, 2]);
    assert!(result.coeffs.iter().all(|&v| v.is_finite()));
    assert!(result.std_errors.iter().all(|&v| v.is_finite()));
    assert!(result.j_stat.is_finite());
    assert!(result.j_p.is_finite());
    assert_eq!(result.var_names, vec!["y0", "y1"]);
}

#[test]
fn test_panel_var_coefficients_and_pvalues_are_reasonable() {
    let a = Array2::from_shape_vec((2, 2), vec![0.4, 0.1, 0.1, 0.3]).unwrap();
    let (data, entity_ids) = generate_panel_var(&a, 12, 20, 0.15, 7002);

    let result = PanelVAR::fit(&data, &entity_ids, 1, Some(vec!["x".into(), "y".into()])).unwrap();

    // Coefficients are finite and in a plausible stable range.
    assert!(result
        .coeffs
        .iter()
        .all(|&v| v.is_finite() && v.abs() < 2.0));
    // P-values are probabilities in [0, 1].
    assert!(result
        .p_values
        .iter()
        .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)));
    assert!(result.t_values.iter().all(|&v| v.is_finite()));
    assert!(result.n_obs > 0);
    assert!(result.n_instruments > 0);
}

#[test]
fn test_panel_var_input_validation() {
    let data = Array2::from_shape_vec((6, 2), vec![1.0; 12]).unwrap();
    let ids = vec![0_i64, 0, 0, 1, 1, 1];

    // Mismatched entity id length.
    assert!(PanelVAR::fit(&data, &ids[..4], 1, None).is_err());

    // Zero lags.
    assert!(PanelVAR::fit(&data, &ids, 0, None).is_err());

    // Too few observations after differencing.
    let small = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();
    assert!(PanelVAR::fit(&small, &vec![0, 0], 1, None).is_err());
}
