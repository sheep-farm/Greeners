use greeners::{Transformer, TransformerResult};
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn make_transformer_series(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        y.push(0.1 * i as f64 + noise.sample(&mut rng));
    }
    Array1::from_vec(y)
}

fn assert_transformer_result_finite(result: &TransformerResult, n: usize) {
    assert_eq!(result.n_obs, n);
    assert_eq!(result.fitted.len(), n);
    assert!(result.fitted.iter().all(|v| v.is_finite()));
    assert!(result.forecast.len() > 0);
    assert!(result.forecast.iter().all(|v| v.is_finite()));
    assert!(result.d_model > 0);
    assert!(result.seq_len > 0);
    assert!(result.learning_rate > 0.0);
    assert!(result.n_epochs > 0);
    assert!(result.mse.is_finite());
    assert!(result.r_squared.is_finite());
    assert_eq!(result.n_samples, n - result.seq_len);
}

/// Transformer returns shaped, finite in-sample and forecast values.
#[test]
fn test_transformer_fit_finite_and_reasonable() {
    let n = 60;
    let y = make_transformer_series(n, 9445);
    let result = Transformer::fit(&y, Some(8), Some(10), Some(0.001), Some(100), Some(5)).unwrap();
    assert_transformer_result_finite(&result, n);
    assert!(result.r_squared > 0.0, "r2 = {}", result.r_squared);
    assert!(result.mse < 20.0, "mse = {}", result.mse);
}

/// Input validation catches short series and zero-variance input.
#[test]
fn test_transformer_input_validation() {
    let y_short = Array1::from_vec(vec![1.0; 10]);
    assert!(Transformer::fit(&y_short, None, None, None, None, None).is_err());

    let y_const = Array1::from_vec(vec![1.0; 25]);
    assert!(Transformer::fit(&y_const, None, None, None, None, None).is_err());
}
