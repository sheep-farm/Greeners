use greeners_ml::hierarchical::HierarchicalClustering;
use greeners_ml::hierarchical::HierarchicalResult;
use greeners_ml::hierarchical::Linkage;
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_hierarchical_data(seed: u64) -> Array2<f64> {
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

fn assert_hierarchical_result_finite(result: &HierarchicalResult, n: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, 2);
    assert_eq!(result.labels.len(), n);
    assert!(result.labels.iter().all(|&l| l < result.n_clusters));
    assert!(result.n_clusters > 0);
    assert_eq!(result.merges.len(), n - 1);
    assert!(
        result.cophenetic_corr.is_nan() || result.cophenetic_corr.abs() <= 1.0,
        "cophenetic_corr = {}",
        result.cophenetic_corr
    );
    assert!(result.cut_height.is_finite());
    assert_eq!(result.cluster_sizes.iter().sum::<usize>(), n);
}

/// Hierarchical clustering with Ward linkage returns correct shapes and clusters.
#[test]
fn test_hierarchical_ward_fit() {
    let x = make_hierarchical_data(9431);
    let n = x.nrows();
    let result = HierarchicalClustering::fit(&x, Linkage::Ward, Some(5.0)).unwrap();
    assert_hierarchical_result_finite(&result, n);
    assert!(result.n_clusters <= n);
}

/// All four linkages produce finite, consistent results on the same data.
#[test]
fn test_hierarchical_linkages() {
    let x = make_hierarchical_data(9432);
    let n = x.nrows();
    for linkage in [
        Linkage::Ward,
        Linkage::Single,
        Linkage::Complete,
        Linkage::Average,
    ] {
        let result = HierarchicalClustering::fit(&x, linkage, Some(5.0)).unwrap();
        assert_hierarchical_result_finite(&result, n);
    }
}

/// Input validation catches insufficient observations.
#[test]
fn test_hierarchical_input_validation() {
    let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    assert!(HierarchicalClustering::fit(&x, Linkage::Ward, Some(1.0)).is_err());
}
