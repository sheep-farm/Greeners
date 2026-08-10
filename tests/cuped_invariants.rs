use greeners::{CupedResult, CUPED};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn make_cuped_data(seed: u64, n: usize) -> (Array1<f64>, Array1<f64>, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut treated = Vec::with_capacity(n);

    for i in 0..n {
        let treated_i = i % 2 == 0;
        treated.push(treated_i);
        let x_val = noise.sample(&mut rng) * 2.0 + if treated_i { 0.5 } else { 0.0 };
        let y_val = 1.0 + 0.8 * x_val + if treated_i { 1.5 } else { 0.0 } + noise.sample(&mut rng);
        x.push(x_val);
        y.push(y_val);
    }

    (Array1::from_vec(y), Array1::from_vec(x), treated)
}

fn assert_finite(result: &CupedResult) {
    assert!(result.treatment_effect.is_finite());
    assert!(result.se >= 0.0 && result.se.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(result.ci[0].is_finite() && result.ci[1].is_finite());
    assert!(result.theta.is_finite());
}

#[test]
fn cuped_univariate_runs_and_recovers() {
    let (y, x, treated) = make_cuped_data(11, 200);

    let result = CUPED::fit(&y, &x, &treated).unwrap();
    assert_finite(&result);

    assert!(
        (result.treatment_effect - 1.5).abs() < 0.3,
        "effect out of range: {}",
        result.treatment_effect
    );
    assert!(
        result.variance_reduction >= 0.0,
        "variance reduction should be non-negative"
    );
    assert!(result.n_treatment > 0);
    assert!(result.n_control > 0);
}

#[test]
fn cuped_multivariate_runs_and_recovers() {
    let n = 200;
    let mut rng = StdRng::seed_from_u64(22);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n * 2);
    let mut treated = Vec::with_capacity(n);

    for i in 0..n {
        let treated_i = i % 2 == 0;
        treated.push(treated_i);
        let x1 = noise.sample(&mut rng) + if treated_i { 0.5 } else { 0.0 };
        let x2 = 0.5 * x1 + noise.sample(&mut rng);
        x.push(x1);
        x.push(x2);
        let y_val =
            1.0 + 0.7 * x1 + 0.3 * x2 + if treated_i { 1.5 } else { 0.0 } + noise.sample(&mut rng);
        y.push(y_val);
    }

    let y = Array1::from_vec(y);
    let x = Array2::from_shape_vec((n, 2), x).unwrap();

    let result = CUPED::fit_multivariate(&y, &x, &treated).unwrap();
    assert_finite(&result);

    assert!(
        (result.treatment_effect - 1.5).abs() < 0.4,
        "effect out of range: {}",
        result.treatment_effect
    );
    assert!(result.n_treatment > 0);
    assert!(result.n_control > 0);
}

#[test]
fn cuped_input_validation() {
    let n = 20;
    let y = Array1::from_vec(vec![0.0; n]);
    let x = Array1::from_vec(vec![0.0; n - 1]);
    let treated = vec![true; n];

    assert!(CUPED::fit(&y, &x, &treated).is_err());

    let treated_short = vec![true; n - 1];
    assert!(CUPED::fit(&y, &Array1::from_vec(vec![0.0; n]), &treated_short).is_err());

    // All same group
    let treated_all = vec![true; n];
    assert!(CUPED::fit(&y, &Array1::from_vec(vec![0.0; n]), &treated_all).is_err());
}
