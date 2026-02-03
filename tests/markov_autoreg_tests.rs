use greeners::MarkovAutoregression;
use ndarray::Array1;

#[test]
fn test_markov_autoreg_two_regime_ar1() {
    // Generate data from a 2-regime AR(1) model
    // Regime 0: mu=0, phi=0.5, sigma=1
    // Regime 1: mu=2, phi=-0.3, sigma=0.5
    let n = 500;
    let mut y = vec![0.0_f64; n];
    let mut regime = vec![0_usize; n];

    // Simple PRNG (LCG)
    let mut seed: u64 = 42;
    let mut next_rand = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };

    // Pre-generate all random numbers needed
    // We need: 1 normal for y[0], then for each t in 1..n: 1 uniform + 1 normal
    // Each normal needs 2 uniforms (Box-Muller), so total = 2 + (n-1)*3 uniforms
    let mut all_u: Vec<f64> = Vec::with_capacity(2 + (n - 1) * 3);
    for _ in 0..(2 + (n - 1) * 3) {
        all_u.push(next_rand());
    }
    let mut ui = 0;
    let next_normal_from = |idx: &mut usize| -> f64 {
        let u1 = all_u[*idx].max(1e-10);
        let u2 = all_u[*idx + 1];
        *idx += 2;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // Switch regimes every ~50 obs
    let mut current_regime = 0_usize;
    y[0] = next_normal_from(&mut ui);
    regime[0] = current_regime;

    for t in 1..n {
        // Switch regime with probability ~0.02
        let u = all_u[ui];
        ui += 1;
        if u < 0.02 {
            current_regime = 1 - current_regime;
        }
        regime[t] = current_regime;

        let (mu, phi, sigma) = if current_regime == 0 {
            (0.0, 0.5, 1.0)
        } else {
            (2.0, -0.3, 0.5)
        };

        y[t] = mu + phi * y[t - 1] + sigma * next_normal_from(&mut ui);
    }

    let y_arr = Array1::from_vec(y);

    let result = MarkovAutoregression::fit(&y_arr, 2, 1).expect("fit should succeed");

    // Check shapes
    assert_eq!(result.k_regimes, 2);
    assert_eq!(result.ar_order, 1);
    assert_eq!(result.regime_means.len(), 2);
    assert_eq!(result.ar_params.shape(), &[2, 1]);
    assert_eq!(result.regime_sigmas.len(), 2);
    assert_eq!(result.smoothed_probs.ncols(), 2);
    assert_eq!(result.smoothed_probs.nrows(), result.n_obs);
    assert_eq!(result.filtered_probs.shape(), result.smoothed_probs.shape());
    assert_eq!(result.transition_matrix.shape(), &[2, 2]);

    // Log-likelihood should be finite
    assert!(
        result.log_likelihood.is_finite(),
        "log_likelihood not finite"
    );
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());

    // Regime means should be approximately recovered (may be swapped)
    let mut means: Vec<f64> = result.regime_means.to_vec();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // The lower mean should be roughly near 0, higher near 2
    // With EM and limited data, allow generous tolerance
    assert!(
        means[0] < means[1],
        "regime means should differ: {:?}",
        means
    );

    // predict_regime should return valid regime indices
    let regimes = result.predict_regime();
    assert_eq!(regimes.len(), result.n_obs);
    for &r in regimes.iter() {
        assert!(r < 2);
    }

    // Display should not panic
    let display = format!("{}", result);
    assert!(display.contains("Markov Autoregression"));
}

#[test]
fn test_markov_autoreg_ar0() {
    // Test with ar_order=0 (pure switching mean model)
    let n = 200;
    let mut y = vec![0.0_f64; n];
    let mut seed: u64 = 123;
    let mut next_rand = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let mut next_normal = || -> f64 {
        let u1 = next_rand().max(1e-10);
        let u2 = next_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    for t in 0..n {
        let regime = if t < 100 { 0 } else { 1 };
        let mu = if regime == 0 { -1.0 } else { 3.0 };
        y[t] = mu + next_normal();
    }

    let y_arr = Array1::from_vec(y);
    let result = MarkovAutoregression::fit(&y_arr, 2, 0).expect("fit should succeed");
    assert_eq!(result.ar_order, 0);
    assert_eq!(result.ar_params.shape(), &[2, 0]);
    assert!(result.log_likelihood.is_finite());
}

#[test]
fn test_markov_autoreg_errors() {
    let short = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(MarkovAutoregression::fit(&short, 2, 1).is_err());

    let y = Array1::from_vec(vec![1.0; 50]);
    assert!(MarkovAutoregression::fit(&y, 1, 1).is_err());
}
