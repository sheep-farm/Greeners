use greeners::{KMeans, KmeansResult};
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_kmeans_data(seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.4).unwrap();
    let mut data = Vec::new();
    let centers = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)];
    let n_per = 20;
    for (cx, cy) in centers {
        for _ in 0..n_per {
            data.push(cx + noise.sample(&mut rng));
            data.push(cy + noise.sample(&mut rng));
        }
    }
    Array2::from_shape_vec((n_per * centers.len(), 2), data).unwrap()
}

fn assert_kmeans_result_finite(result: &KmeansResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, 2);
    assert_eq!(result.n_clusters, k);
    assert_eq!(result.labels.len(), n);
    assert!(result.labels.iter().all(|&l| l < k));
    assert_eq!(result.centroids.nrows(), k);
    assert_eq!(result.centroids.ncols(), 2);
    assert!(result.centroids.iter().all(|v| v.is_finite()));
    assert_eq!(result.cluster_sizes.len(), k);
    assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n);
    assert!(result.cluster_sizes.iter().all(|&s| s > 0));
    assert!(result.inertia >= 0.0 && result.inertia.is_finite());
    assert!(result.between_ss >= 0.0 && result.between_ss.is_finite());
    assert!(result.total_ss >= 0.0 && result.total_ss.is_finite());
}

/// KMeans recovers three well-separated clusters with correct shapes.
#[test]
fn test_kmeans_cluster_recovery() {
    let x = make_kmeans_data(9425);
    let n = x.nrows();
    let result = KMeans::fit(&x, 3, None, None).unwrap();
    assert_kmeans_result_finite(&result, n, 3);
    for c in 0..3 {
        assert!(
            result.cluster_sizes[c] >= 15,
            "cluster {} size = {}",
            c,
            result.cluster_sizes[c]
        );
    }
}

/// KMeans centroids are near the true cluster centers.
#[test]
fn test_kmeans_centroids_near_truth() {
    let x = make_kmeans_data(9426);
    let result = KMeans::fit(&x, 3, None, None).unwrap();
    let mut centers: Vec<(f64, f64)> = Vec::new();
    for c in 0..3 {
        centers.push((result.centroids[(c, 0)], result.centroids[(c, 1)]));
    }
    let truth = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)];
    for (tx, ty) in truth {
        let mut found = false;
        for &(cx, cy) in &centers {
            if (cx - tx).abs() < 1.0 && (cy - ty).abs() < 1.0 {
                found = true;
                break;
            }
        }
        assert!(found, "no centroid near ({}, {})", tx, ty);
    }
}

/// Input validation catches impossible clustering requests.
#[test]
fn test_kmeans_input_validation() {
    let x = make_kmeans_data(9427);
    assert!(KMeans::fit(&x, 0, None, None).is_err());
    assert!(KMeans::fit(&x, 1000, None, None).is_err());
}
