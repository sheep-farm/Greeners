use greeners::{DbscanResult, DBSCAN};
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn make_dbscan_data(seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.2).unwrap();
    let mut data = Vec::new();
    let centers = [(0.0, 0.0), (5.0, 0.0)];
    let n_per = 20;
    for (cx, cy) in centers {
        for _ in 0..n_per {
            data.push(cx + noise.sample(&mut rng));
            data.push(cy + noise.sample(&mut rng));
        }
    }
    // Add an outlier far from the clusters
    data.push(20.0);
    data.push(20.0);
    Array2::from_shape_vec((n_per * centers.len() + 1, 2), data).unwrap()
}

fn assert_dbscan_result_finite(result: &DbscanResult, n: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, 2);
    assert_eq!(result.labels.len(), n);
    assert!(result.labels.iter().all(|&l| l >= -1));
    assert!(result.n_clusters > 0);

    assert!(result.cluster_sizes.iter().sum::<usize>() + result.n_noise == n);
    assert!(result.eps > 0.0);
    assert!(result.min_pts >= 2);
}

/// DBSCAN finds two clusters and flags the outlier as noise.
#[test]
fn test_dbscan_two_clusters_and_noise() {
    let x = make_dbscan_data(9428);
    let n = x.nrows();
    let result = DBSCAN::fit(&x, 1.5, 3).unwrap();
    assert_dbscan_result_finite(&result, n);
    assert_eq!(result.n_clusters, 2);
    assert_eq!(result.n_noise, 1);
    assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n - 1);
}

/// All noise: DBSCAN with too small eps returns only noise.
#[test]
fn test_dbscan_all_noise() {
    let x = make_dbscan_data(9429);
    let result = DBSCAN::fit(&x, 0.01, 3).unwrap();
    assert_eq!(result.n_clusters, 0);
    assert_eq!(result.n_noise, x.nrows());
}

/// Input validation catches invalid parameters.
#[test]
fn test_dbscan_input_validation() {
    let x = make_dbscan_data(9430);
    assert!(DBSCAN::fit(&x, 0.0, 3).is_err());
    assert!(DBSCAN::fit(&x, -1.0, 3).is_err());
    assert!(DBSCAN::fit(&x, 1.0, 1).is_err());
    assert!(DBSCAN::fit(&x, 1.0, 0).is_err());

    let x_short = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    assert!(DBSCAN::fit(&x_short, 1.0, 2).is_err());
}
