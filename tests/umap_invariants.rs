use greeners::{UmapResult, UMAP};
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_umap_data(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut data = Vec::new();
    let centers = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)];
    let n_per = n / 2;
    for (cx, cy, cz) in centers {
        for _ in 0..n_per {
            data.push(cx + noise.sample(&mut rng));
            data.push(cy + noise.sample(&mut rng));
            data.push(cz + noise.sample(&mut rng));
        }
    }
    Array2::from_shape_vec((n_per * centers.len(), 3), data).unwrap()
}

fn assert_umap_result_finite(result: &UmapResult, n: usize, n_comp: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, 3);
    assert_eq!(result.n_components, n_comp);
    assert_eq!(result.embedding.nrows(), n);
    assert_eq!(result.embedding.ncols(), n_comp);
    assert!(result.embedding.iter().all(|v| v.is_finite()));
    assert!(result.loss.is_finite());
    assert!(result.n_iter > 0);
    assert!(result.n_neighbors >= 2);
    assert!(result.min_dist >= 0.0);
}

/// UMAP returns an embedding with the requested number of dimensions.
#[test]
fn test_umap_embedding_shape_and_finite() {
    let n = 20;
    let x = make_umap_data(n, 9441);
    let result = UMAP::fit(&x, Some(5), Some(2), Some(0.1), Some(100)).unwrap();
    assert_umap_result_finite(&result, n, 2);
}

/// UMAP supports 1D output and respects input constraints.
#[test]
fn test_umap_1d_and_validation() {
    let n = 20;
    let x = make_umap_data(n, 9442);
    let result = UMAP::fit(&x, Some(5), Some(1), Some(0.1), Some(100)).unwrap();
    assert_umap_result_finite(&result, n, 1);

    let x_short = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
    assert!(UMAP::fit(&x_short, None, None, None, None).is_err());
}
