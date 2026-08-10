use greeners::{GmmClustering, GmmClusteringResult};
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_gmm_data(seed: u64) -> Array2<f64> {
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

fn assert_gmm_result_finite(result: &GmmClusteringResult, n: usize, k: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, 2);
    assert_eq!(result.n_clusters, k);
    assert_eq!(result.labels.len(), n);
    assert!(result.labels.iter().all(|&l| l < k));
    assert_eq!(result.means.nrows(), k);
    assert_eq!(result.means.ncols(), 2);
    assert!(result.means.iter().all(|v| v.is_finite()));
    assert_eq!(result.covariances.len(), k);
    for cov in &result.covariances {
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
        assert!(cov.iter().all(|v| v.is_finite()));
    }
    assert_eq!(result.weights.len(), k);
    assert!((result.weights.sum() - 1.0).abs() < 1e-10);
    assert!(result.weights.iter().all(|&w| w >= 0.0 && w.is_finite()));
    assert_eq!(result.responsibilities.nrows(), n);
    assert_eq!(result.responsibilities.ncols(), k);
    assert!(result
        .responsibilities
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.log_likelihood.is_finite());
    assert!(result.bic.is_finite());
    assert!(result.aic.is_finite());
}

/// GMM recovers three well-separated Gaussian clusters.
#[test]
fn test_gmm_cluster_recovery() {
    let x = make_gmm_data(9436);
    let n = x.nrows();
    let result = GmmClustering::fit(&x, 3, None, None).unwrap();
    assert_gmm_result_finite(&result, n, 3);
    let mut sizes = vec![0; 3];
    for &l in &result.labels {
        sizes[l] += 1;
    }
    for c in 0..3 {
        assert!(sizes[c] >= 15, "cluster {} size = {}", c, sizes[c]);
    }
}

/// GMM centroids are near the true cluster means.
#[test]
fn test_gmm_centroids_near_truth() {
    let x = make_gmm_data(9437);
    let result = GmmClustering::fit(&x, 3, None, None).unwrap();
    let truth = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)];
    let mut found = [false; 3];
    for &(tx, ty) in &truth {
        for c in 0..3 {
            let (mx, my) = (result.means[(c, 0)], result.means[(c, 1)]);
            if (mx - tx).abs() < 1.0 && (my - ty).abs() < 1.0 {
                found[truth.iter().position(|&t| t == (tx, ty)).unwrap()] = true;
            }
        }
    }
    assert!(found.iter().all(|&f| f));
}

/// Input validation catches invalid cluster counts and too few observations.
#[test]
fn test_gmm_input_validation() {
    let x = make_gmm_data(9438);
    assert!(GmmClustering::fit(&x, 0, None, None).is_err());
    assert!(GmmClustering::fit(&x, 1000, None, None).is_err());

    let x_short = Array2::from_shape_vec((3, 2), vec![1.0; 6]).unwrap();
    assert!(GmmClustering::fit(&x_short, 2, None, None).is_err());
}
