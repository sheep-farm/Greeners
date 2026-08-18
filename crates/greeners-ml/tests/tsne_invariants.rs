use greeners_ml::tsne::TsneResult;
use greeners_ml::tsne::TSNE;
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_tsne_data(n: usize, seed: u64) -> Array2<f64> {
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

fn assert_tsne_result_finite(result: &TsneResult, n: usize, n_comp: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, 3);
    assert_eq!(result.n_components, n_comp);
    assert_eq!(result.embedding.nrows(), n);
    assert_eq!(result.embedding.ncols(), n_comp);
    assert!(result.embedding.iter().all(|v| v.is_finite()));
    assert!(result.kl_divergence.is_finite());
    assert!(result.n_iter > 0);
    assert!(result.perplexity >= 5.0);
    assert!(result.learning_rate > 0.0);
}

/// t-SNE returns an embedding with the requested number of dimensions.
#[test]
fn test_tsne_embedding_shape_and_finite() {
    let n = 20;
    let x = make_tsne_data(n, 9439);
    let result = TSNE::fit(&x, Some(5.0), Some(2), Some(200), Some(50.0)).unwrap();
    assert_tsne_result_finite(&result, n, 2);
}

/// t-SNE supports 3D output and respects input constraints.
#[test]
fn test_tsne_3d_and_validation() {
    let n = 20;
    let x = make_tsne_data(n, 9440);
    let result = TSNE::fit(&x, Some(5.0), Some(3), Some(200), Some(50.0)).unwrap();
    assert_tsne_result_finite(&result, n, 3);

    let x_short = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
    assert!(TSNE::fit(&x_short, None, None, None, None).is_err());
}
