use greeners::{QrfResult, QRF};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_qrf_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut x = Array2::zeros((n, 2));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = rng.gen::<f64>() * 4.0 - 2.0;
        let x2 = rng.gen::<f64>() * 4.0 - 2.0;
        x[(i, 0)] = x1;
        x[(i, 1)] = x2;
        y.push(1.0 + 2.0 * x1 + 3.0 * x2 + noise.sample(&mut rng));
    }
    (Array1::from_vec(y), x)
}

fn assert_qrf_result_finite(result: &QrfResult, n: usize, k: usize, n_q: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, k);
    assert_eq!(result.quantile_predictions.nrows(), n);
    assert_eq!(result.quantile_predictions.ncols(), n_q);
    assert!(result.quantile_predictions.iter().all(|v| v.is_finite()));
    assert_eq!(result.quantiles.len(), n_q);
    assert_eq!(result.feature_importance.len(), k);
    assert!(result
        .feature_importance
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.oob_r_squared.is_finite());
    assert!(result.n_trees > 0);
}

/// QRF quantile predictions have the expected shape, are finite, and are monotonic.
#[test]
fn test_qrf_fit_finite_and_monotonic() {
    let n = 50;
    let (y, x) = make_qrf_data(n, 9410);
    let quantiles = vec![0.1, 0.5, 0.9];
    let result = QRF::fit(&y, &x, quantiles.clone(), 50, 5, None).unwrap();
    assert_qrf_result_finite(&result, n, 2, 3);

    for i in 0..n {
        for j in 1..3 {
            assert!(
                result.quantile_predictions[(i, j - 1)] <= result.quantile_predictions[(i, j)],
                "quantiles not monotonic at obs {}",
                i
            );
        }
    }
}

/// Input validation catches invalid dimensions, zero trees, and bad quantiles.
#[test]
fn test_qrf_input_validation() {
    let (y, x) = make_qrf_data(10, 9411);
    assert!(QRF::fit(&y, &x, vec![0.5], 0, 3, None).is_err());

    let y_short = Array1::from_vec(vec![1.0; 4]);
    let x_short = Array2::from_shape_vec((4, 1), vec![1.0; 4]).unwrap();
    assert!(QRF::fit(&y_short, &x_short, vec![0.5], 10, 3, None).is_err());

    let (y2, x2) = make_qrf_data(10, 9412);
    assert!(QRF::fit(&y2, &x2, vec![0.0], 10, 3, None).is_err());
    assert!(QRF::fit(&y2, &x2, vec![1.0], 10, 3, None).is_err());
    assert!(QRF::fit(&y2, &x2, vec![-0.1], 10, 3, None).is_err());
}
