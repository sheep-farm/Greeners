use greeners::{Biplot, BiplotResult, BiplotType};
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_biplot_data(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut data = Vec::with_capacity(n * 3);
    for _ in 0..n {
        let x1 = rng.gen::<f64>() * 4.0 - 2.0;
        let x2 = 2.0 * x1 + noise.sample(&mut rng);
        let x3 = -1.0 * x1 + noise.sample(&mut rng);
        data.push(x1);
        data.push(x2);
        data.push(x3);
    }
    Array2::from_shape_vec((n, 3), data).unwrap()
}

fn assert_biplot_result_finite(result: &BiplotResult, n: usize, p: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_vars, p);
    assert_eq!(result.scores.nrows(), n);
    assert_eq!(result.scores.ncols(), 2);
    assert!(result.scores.iter().all(|v| v.is_finite()));
    assert_eq!(result.loadings.nrows(), p);
    assert_eq!(result.loadings.ncols(), 2);
    assert!(result.loadings.iter().all(|v| v.is_finite()));
    // Biplot always uses 2 components, so variance arrays have length 2.
    assert_eq!(result.explained_variance_ratio.len(), 2);
    assert!(result
        .explained_variance_ratio
        .iter()
        .all(|&v| v >= 0.0 && v.is_finite()));
    assert_eq!(result.cumulative_variance.len(), 2);
    assert!(result
        .cumulative_variance
        .iter()
        .all(|&v| v >= 0.0 && v <= 1.0));
    assert_eq!(result.variable_names.len(), p);
    assert_eq!(result.obs_labels.len(), n);
    assert!(!result.ascii_biplot.is_empty());
}

/// Biplot returns consistent shapes for all three biplot types.
#[test]
fn test_biplot_all_types() {
    let n = 20;
    let p = 3;
    let x = make_biplot_data(n, 9443);
    for biplot_type in [
        BiplotType::Form,
        BiplotType::Covariance,
        BiplotType::Symmetric,
    ] {
        let result = Biplot::fit(&x, biplot_type, None).unwrap();
        assert_biplot_result_finite(&result, n, p);
    }
}

/// Biplot handles custom variable names and validates input.
#[test]
fn test_biplot_names_and_validation() {
    let n = 20;
    let p = 3;
    let x = make_biplot_data(n, 9444);
    let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let result = Biplot::fit(&x, BiplotType::Form, Some(names)).unwrap();
    assert_biplot_result_finite(&result, n, p);
    assert_eq!(result.variable_names, vec!["a", "b", "c"]);

    let x_small = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();
    assert!(Biplot::fit(&x_small, BiplotType::Form, None).is_err());

    let x_one_var = Array2::from_shape_vec((5, 1), vec![1.0; 5]).unwrap();
    assert!(Biplot::fit(&x_one_var, BiplotType::Form, None).is_err());
}
