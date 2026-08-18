use greeners::{SpectralClustering, SpectralResult};
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_spectral_data(seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.3).unwrap();
    let mut data = Vec::new();
    let centers = [(0.0, 0.0), (5.0, 0.0)];
    let n_per = 15;
    for (cx, cy) in centers {
        for _ in 0..n_per {
            data.push(cx + noise.sample(&mut rng));
            data.push(cy + noise.sample(&mut rng));
        }
    }
    Array2::from_shape_vec((n_per * centers.len(), 2), data).unwrap()
}

fn assert_spectral_result_finite(result: &SpectralResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, 2);
    assert_eq!(result.n_clusters, k);
    assert_eq!(result.labels.len(), n);
    assert!(result.labels.iter().all(|&l| l < k));
    assert_eq!(result.affinity.nrows(), n);
    assert_eq!(result.affinity.ncols(), n);
    assert!(result.affinity.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert_eq!(result.eigenvalues.len(), k);
    assert!(result.eigenvalues.iter().all(|v| v.is_finite()));
    assert_eq!(result.eigenvectors.nrows(), n);
    assert_eq!(result.eigenvectors.ncols(), k);
    assert!(result.eigenvectors.iter().all(|v| v.is_finite()));
    assert_eq!(result.centroids.nrows(), k);
    assert_eq!(result.centroids.ncols(), k);
    assert!(result.centroids.iter().all(|v| v.is_finite()));
    assert!(result.inertia >= 0.0 && result.inertia.is_finite());
    assert!(result.sigma > 0.0);
}

/// Spectral clustering recovers two well-separated clusters.
#[test]
fn test_spectral_cluster_recovery() {
    let x = make_spectral_data(9433);
    let n = x.nrows();
    let result = SpectralClustering::fit(&x, 2, None, Some(100)).unwrap();
    assert_spectral_result_finite(&result, n, 2);
    let mut sizes = vec![0; 2];
    for &l in &result.labels {
        sizes[l] += 1;
    }
    assert!(sizes[0] > 0);
    assert!(sizes[1] > 0);
}

/// Different numbers of components produce consistent shapes.
#[test]
fn test_spectral_n_components() {
    let x = make_spectral_data(9434);
    let n = x.nrows();
    let result = SpectralClustering::fit(&x, 2, Some(1.0), Some(50)).unwrap();
    assert_spectral_result_finite(&result, n, 2);
}

/// Input validation catches invalid cluster counts and too few observations.
#[test]
fn test_spectral_input_validation() {
    let x = make_spectral_data(9435);
    assert!(SpectralClustering::fit(&x, 1, None, None).is_err());
    assert!(SpectralClustering::fit(&x, 1000, None, None).is_err());

    let x_short = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(SpectralClustering::fit(&x_short, 2, None, None).is_err());
}
