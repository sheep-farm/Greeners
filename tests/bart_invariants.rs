use greeners::BART;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::{Distribution, StandardNormal, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_bart_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let x_vec: Vec<f64> = (0..n * 2).map(|_| dist.sample(&mut rng)).collect();
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let noise: f64 = StandardNormal.sample(&mut rng);
        y[i] = 1.0 + 2.0 * x[(i, 0)] - 1.5 * x[(i, 1)] + 0.3 * noise;
    }
    (y, x)
}

/// BART returns fitted values of the right length and finite diagnostics.
#[test]
fn test_bart_shape_and_finite() {
    let (y, x) = make_bart_data(50, 6001);
    let r = BART::fit(&y, &x, Some(10), Some(2), Some(50), Some(20), None).unwrap();
    assert_eq!(r.fitted.len(), y.len());
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.n_features, x.ncols());
    assert!(r.fitted.iter().all(|v| v.is_finite()));
    assert!(r.sigma2.is_finite() && r.sigma2 > 0.0);
    assert!(r.r_squared.is_finite());
    assert!(r.mse.is_finite());
    assert_eq!(r.variable_inclusion.len(), x.ncols());
    assert!(r.variable_inclusion.iter().all(|v| v.is_finite()));
    assert!((r.variable_inclusion.sum() - 1.0).abs() < 1e-10);
}

/// BART rejects too few observations or a zero-variance response.
#[test]
fn test_bart_input_validation() {
    let (y, x) = make_bart_data(5, 6002);
    assert!(BART::fit(&y, &x, None, None, None, None, None).is_err());

    let y_const = Array1::from_vec(vec![1.0; 20]);
    let x = Array2::from_shape_vec((20, 2), vec![0.0; 40]).unwrap();
    assert!(BART::fit(&y_const, &x, None, None, None, None, None).is_err());
}

/// Fitted values and in-sample MSE are self-consistent.
#[test]
fn test_bart_fitted_mse_consistency() {
    let (y, x) = make_bart_data(40, 6003);
    let r = BART::fit(&y, &x, Some(10), Some(2), Some(40), Some(15), None).unwrap();
    let sse: f64 = y
        .iter()
        .zip(r.fitted.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!((sse / (r.n_obs as f64) - r.mse).abs() < 1e-10);
}
