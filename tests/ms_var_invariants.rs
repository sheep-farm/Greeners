use greeners::MSVAR;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn generate_msvar_data(t: usize, n_vars: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.4).unwrap();

    // 2-regime process: intercept 0 vs 2, stable AR coefficients.
    let a = Array2::from_shape_vec((n_vars, n_vars), vec![0.3, 0.05, 0.05, 0.3]).unwrap();
    let mut data = Array2::zeros((t, n_vars));
    let mut regime = 0_usize;
    for i in 1..t {
        // Deterministic regime switch every 30 periods.
        if i % 30 == 0 {
            regime = 1 - regime;
        }
        let intercept = if regime == 0 {
            Array1::zeros(n_vars)
        } else {
            Array1::from(vec![2.0; n_vars])
        };
        let y_prev = data.row(i - 1).to_owned();
        let y_t = &intercept + a.dot(&y_prev);
        data.row_mut(i).assign(&y_t);
        for j in 0..n_vars {
            data[[i, j]] += noise.sample(&mut rng);
        }
    }
    data
}

#[test]
fn test_msvar_runs_and_produces_finite_output() {
    let data = generate_msvar_data(120, 2, 11001);

    let result = MSVAR::fit(&data, 2, 1, None).unwrap();

    assert_eq!(result.n_regimes, 2);
    assert_eq!(result.n_vars, 2);
    assert_eq!(result.lags, 1);
    assert_eq!(result.n_obs, data.nrows() - 1);
    assert_eq!(result.regime_intercepts.shape(), &[2, 2]);
    assert_eq!(result.ar_coeffs.shape(), &[2, 2]); // (n_vars * lags) x n_vars
    assert_eq!(result.regime_covariances.shape(), &[2, 2, 2]);
    assert_eq!(result.transition_matrix.shape(), &[2, 2]);
    assert_eq!(result.filtered_probs.shape(), &[result.n_obs, 2]);
    assert_eq!(result.smoothed_probs.shape(), &[result.n_obs, 2]);
    assert!(result.regime_intercepts.iter().all(|&v| v.is_finite()));
    assert!(result.ar_coeffs.iter().all(|&v| v.is_finite()));
    assert!(result.regime_covariances.iter().all(|&v| v.is_finite()));
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());

    // Transition matrix rows should be probabilities.
    for i in 0..2 {
        let s: f64 = result.transition_matrix.row(i).sum();
        assert!((s - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_msvar_probabilities_are_valid_and_ar_stable() {
    let data = generate_msvar_data(150, 2, 11002);

    let result = MSVAR::fit(&data, 2, 1, Some(vec!["y1".into(), "y2".into()])).unwrap();

    // Smoothed and filtered probabilities should sum to one and be in [0, 1].
    for probs in &[&result.filtered_probs, &result.smoothed_probs] {
        for i in 0..probs.nrows() {
            let s: f64 = probs.row(i).sum();
            assert!((s - 1.0).abs() < 1e-5);
            assert!(probs.row(i).iter().all(|&v| v >= 0.0 && v <= 1.0));
        }
    }

    // AR coefficients should be stable (diagonal own-lags smaller than 1).
    assert!(result.ar_coeffs[[0, 0]].abs() < 1.0);
    assert!(result.ar_coeffs[[1, 1]].abs() < 1.0);
    assert!(result.log_likelihood.is_finite());
}

#[test]
fn test_msvar_input_validation() {
    let data = Array2::from_shape_vec((12, 2), vec![1.0; 24]).unwrap();

    // Too few regimes.
    assert!(MSVAR::fit(&data, 1, 1, None).is_err());

    // Too few observations.
    let short = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    assert!(MSVAR::fit(&short, 2, 1, None).is_err());
}
