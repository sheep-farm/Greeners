use greeners::PCA;
use ndarray::Array2;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Principal components are orthonormal and reconstruct the standardized data.
#[test]
fn test_pca_components_orthonormal() {
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0],
    )
    .unwrap();

    let result = PCA::fit(&data, 2).unwrap();

    // Orthonormality: components' * components = I
    let ortho = result.components.t().dot(&result.components);
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            approx_zero((ortho[[i, j]] - expected).abs(), 1e-10);
        }
    }

    // Explained variance ratios sum to 1 for full components.
    approx_zero(result.explained_variance_ratio.sum() - 1.0, 1e-10);
    // Total explained variance equals the number of standardized variables.
    approx_zero(result.explained_variance.sum() - 2.0, 1e-10);

    // Scores are the standardized data projected onto components.
    let mut z = data.clone();
    for (j, mut col) in z.axis_iter_mut(ndarray::Axis(1)).enumerate() {
        col -= result.mean[j];
        col /= result.std[j];
    }
    let scores = z.dot(&result.components);
    for i in 0..scores.nrows() {
        for j in 0..scores.ncols() {
            approx_zero((scores[[i, j]] - result.scores[[i, j]]).abs(), 1e-10);
        }
    }

    // Loadings are components scaled by sqrt(eigenvalue).
    for i in 0..2 {
        for j in 0..2 {
            approx_zero(
                (result.loadings[[i, j]]
                    - result.components[[i, j]] * result.explained_variance[j].sqrt())
                .abs(),
                1e-10,
            );
        }
    }
}

/// n_components is capped at the number of columns.
#[test]
fn test_pca_n_components_cap() {
    let data = Array2::from_shape_vec((10, 3), (1..=30).map(|v| v as f64).collect()).unwrap();
    let result = PCA::fit(&data, 10).unwrap();
    assert_eq!(result.n_components, 3);
}

/// Perfectly correlated data has one PC that explains all the variance.
#[test]
fn test_pca_perfect_collinearity() {
    let data = Array2::from_shape_vec(
        (20, 2),
        (0..20)
            .flat_map(|i| vec![i as f64, 2.0 * i as f64])
            .collect(),
    )
    .unwrap();
    let result = PCA::fit(&data, 2).unwrap();
    approx_zero(result.explained_variance_ratio[0] - 1.0, 1e-10);
    approx_zero(result.explained_variance_ratio[1], 1e-10);
}

/// Input validation.
#[test]
fn test_pca_input_validation() {
    let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    assert!(PCA::fit(&data, 1).is_err());
}
