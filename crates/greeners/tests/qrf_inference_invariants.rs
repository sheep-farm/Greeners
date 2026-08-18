use greeners::{QrfInference, QrfInferenceResult};
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Normal;

fn make_qrfi_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
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

fn assert_qrfi_result_finite(result: &QrfInferenceResult, n: usize, k: usize, n_q: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_features, k);
    assert_eq!(result.point_estimates.nrows(), n);
    assert_eq!(result.point_estimates.ncols(), n_q);
    assert_eq!(result.lower.nrows(), n);
    assert_eq!(result.lower.ncols(), n_q);
    assert_eq!(result.upper.nrows(), n);
    assert_eq!(result.upper.ncols(), n_q);
    assert!(result.point_estimates.iter().all(|v| v.is_finite()));
    assert!(result.lower.iter().all(|v| v.is_finite()));
    assert!(result.upper.iter().all(|v| v.is_finite()));
    assert!(result
        .feature_importance
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
    assert!(result.coverage >= 0.0 && result.coverage <= 1.0);
    assert!(result.confidence > 0.0 && result.confidence < 1.0);
    assert!(result.n_bootstrap > 0);
    assert!(result.n_trees > 0);
}

/// QRFInference returns shaped, finite point estimates and confidence bounds.
#[test]
fn test_qrf_inference_fit_finite_and_bounds() {
    let n = 40;
    let (y, x) = make_qrfi_data(n, 9413);
    let quantiles = vec![0.25, 0.75];
    let result = QrfInference::fit(
        &y,
        &x,
        quantiles,
        Some(10),
        Some(10),
        Some(3),
        Some(0.95),
        None,
    )
    .unwrap();
    assert_qrfi_result_finite(&result, n, 2, 2);

    for i in 0..n {
        for j in 0..2 {
            assert!(
                result.lower[(i, j)] <= result.upper[(i, j)],
                "lower > upper at ({}, {})",
                i,
                j
            );
        }
    }
}

/// Input validation catches insufficient observations and invalid quantiles.
#[test]
fn test_qrf_inference_input_validation() {
    let (y, x) = make_qrfi_data(9, 9414);
    assert!(QrfInference::fit(&y, &x, vec![0.5], None, None, None, None, None).is_err());

    let (y2, x2) = make_qrfi_data(12, 9415);
    assert!(QrfInference::fit(
        &y2,
        &x2,
        vec![0.0],
        Some(5),
        Some(5),
        Some(3),
        Some(0.95),
        None
    )
    .is_err());
    assert!(QrfInference::fit(
        &y2,
        &x2,
        vec![1.0],
        Some(5),
        Some(5),
        Some(3),
        Some(0.95),
        None
    )
    .is_err());
}
