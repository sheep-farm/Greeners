use greeners::SpatialDurbin;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn make_panel_spatial_data(
    seed: u64,
    n_entities: usize,
    n_periods: usize,
    k: usize,
) -> (Array1<f64>, Array2<f64>, Array2<f64>, Vec<i64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.4).unwrap();
    let n = n_entities * n_periods;

    let mut w = Array2::zeros((n_entities, n_entities));
    for i in 0..n_entities {
        if i + 1 < n_entities {
            w[(i, i + 1)] = 1.0;
        }
        if i > 0 {
            w[(i, i - 1)] = 1.0;
        }
    }
    // Row-normalise
    for i in 0..n_entities {
        let row_sum = w.row(i).sum();
        if row_sum > 0.0 {
            for j in 0..n_entities {
                w[(i, j)] /= row_sum;
            }
        }
    }

    let mut x_data = Vec::with_capacity(n * k);
    let mut y = Vec::with_capacity(n);
    let mut entity_ids = Vec::with_capacity(n);

    for t in 0..n_periods {
        for e in 0..n_entities {
            entity_ids.push(e as i64);
            let x1 = (e as f64 + t as f64) / (n_entities as f64 + n_periods as f64);
            x_data.push(1.0);
            x_data.push(x1);
            for _ in 2..k {
                x_data.push(noise.sample(&mut rng));
            }
            y.push(1.0 + 0.5 * x1 + noise.sample(&mut rng));
        }
    }

    (
        Array1::from_vec(y),
        Array2::from_shape_vec((n, k), x_data).unwrap(),
        w,
        entity_ids,
    )
}

/// Spatial Durbin panel fit returns direct and indirect effects with the expected shapes.
#[test]
fn test_spatial_durbin_invariants() {
    let (y, x, w, entity_ids) = make_panel_spatial_data(12345, 5, 6, 2);
    let result = SpatialDurbin::fit(
        &y,
        &x,
        &w,
        &entity_ids,
        Some(vec!["const".into(), "x1".into()]),
    )
    .unwrap();

    assert_eq!(result.n_obs, 30);
    assert_eq!(result.n_entities, 5);
    assert_eq!(result.n_regressors, 2);
    assert_eq!(result.beta.len(), 2);
    assert_eq!(result.theta.len(), 2);
    assert!(result.beta.iter().all(|v| v.is_finite()));
    assert!(result.theta.iter().all(|v| v.is_finite()));
    assert!(result.rho.is_finite());
    assert!(result.r_squared.is_finite());
    assert!(result.log_likelihood.is_finite());
}

/// Input validation rejects mismatched dimensions or an inconsistent panel layout.
#[test]
fn test_spatial_durbin_input_validation() {
    let (y, x, w, entity_ids) = make_panel_spatial_data(11111, 5, 6, 2);

    let y_short = y.slice(ndarray::s![0..20]).to_owned();
    assert!(SpatialDurbin::fit(&y_short, &x, &w, &entity_ids, None).is_err());

    let mut bad_ids = entity_ids.clone();
    bad_ids.pop();
    assert!(SpatialDurbin::fit(&y, &x, &w, &bad_ids, None).is_err());

    let w_bad = Array2::zeros((10, 10));
    assert!(SpatialDurbin::fit(&y, &x, &w_bad, &entity_ids, None).is_err());
}

/// Spatial Durbin handles a larger cross-section with a denser weights matrix.
#[test]
fn test_spatial_durbin_dense_weights() {
    let (y, x, _w, entity_ids) = make_panel_spatial_data(22222, 8, 4, 2);
    let n_entities = 8;
    // Fully connected, equal weights
    let mut w = Array2::from_elem((n_entities, n_entities), 1.0 / n_entities as f64);
    for i in 0..n_entities {
        w[(i, i)] = 0.0;
    }
    // Re-normalise rows
    for i in 0..n_entities {
        let row_sum = w.row(i).sum();
        for j in 0..n_entities {
            w[(i, j)] /= row_sum;
        }
    }

    let result = SpatialDurbin::fit(&y, &x, &w, &entity_ids, None).unwrap();
    assert!(result.beta.iter().all(|v| v.is_finite()));
    assert!(result.theta.iter().all(|v| v.is_finite()));
    assert!(result.rho.is_finite());
}
