use greeners::NARDL;
use ndarray::Array1;
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn generate_nardl_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.3).unwrap();

    // Generate x with asymmetric positive/negative increments.
    let mut x = vec![0.0; n];
    for i in 1..n {
        let dx = if i % 2 == 0 { 0.4 } else { -0.2 } + noise.sample(&mut rng) * 0.2;
        x[i] = x[i - 1] + dx;
    }

    // y = 0.5 + 0.6 * x + noise (long-run multiplier ~ 0.6 for both signs).
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = 0.5 + 0.6 * x[i] + noise.sample(&mut rng);
    }

    (Array1::from_vec(y), Array1::from_vec(x))
}

#[test]
fn test_nardl_runs_and_produces_finite_output() {
    let (y, x) = generate_nardl_data(80, 15001);

    let result = NARDL::fit(&y, &x, 2).unwrap();

    assert_eq!(result.lags, 2);
    assert!(result.n_obs > 0);
    assert!(result.coefficients.iter().all(|&v| v.is_finite()));
    assert!(result.std_errors.iter().all(|&v| v.is_finite()));
    assert!(result.t_values.iter().all(|&v| v.is_finite()));
    assert!(result
        .p_values
        .iter()
        .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)));
    assert_eq!(result.theta_pos.len(), 2);
    assert_eq!(result.theta_neg.len(), 2);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert!(result.lr_asym_f.is_finite());
    assert!(result.sr_asym_f.is_finite());
    assert_eq!(result.coef_names.len(), result.coefficients.len());
}

#[test]
fn test_nardl_recovers_long_run_multiplier() {
    let (y, x) = generate_nardl_data(120, 15002);

    let result = NARDL::fit(&y, &x, 2).unwrap();

    // For this symmetric long-run DGP, both long-run multipliers should be
    // positive and around 0.6.
    assert!(result.beta_pos > 0.0 && result.beta_pos < 1.5);
    assert!(result.beta_neg > 0.0 && result.beta_neg < 1.5);
    assert!((result.beta_pos - result.beta_neg).abs() < 0.8);
    assert!(result.r_squared > 0.5);
}

#[test]
fn test_nardl_input_validation() {
    let y = Array1::from(vec![1.0; 10]);
    let x = Array1::from(vec![1.0; 9]);

    // Mismatched lengths.
    assert!(NARDL::fit(&y, &x, 1).is_err());

    // lags = 0.
    assert!(NARDL::fit(&y, &y, 0).is_err());

    // Too few observations.
    let short = Array1::from(vec![1.0; 8]);
    assert!(NARDL::fit(&short, &short, 1).is_err());
}
