use greeners_core::isotonic::IsotonicRegression;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

/// Isotonic regression preserves monotonicity and fits increasing data.
#[test]
fn test_isotonic_increasing() {
    let n = 100;
    let mut rng = StdRng::seed_from_u64(9701);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 10.0;
        x_vec.push(x);
        y_vec.push(x + noise.sample(&mut rng));
    }
    let x = Array1::from_vec(x_vec);
    let y = Array1::from_vec(y_vec);

    let result = IsotonicRegression::fit(&x, &y, true, None).unwrap();

    // Fitted values are non-decreasing.
    for i in 1..n {
        assert!(result.fitted[i] >= result.fitted[i - 1]);
    }

    let ss_res: f64 = y
        .iter()
        .zip(result.fitted.iter())
        .map(|(&y, &f)| (y - f).powi(2))
        .sum();
    let y_mean = y.mean().unwrap();
    let ss_tot: f64 = y.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    assert!(r2 > 0.85, "r2 = {}", r2);
    assert!(result.x_steps.len() > 0);
    assert!(result.y_steps.len() == result.x_steps.len());
    assert!(result.r_squared > 0.85);
}

/// Isotonic regression fits decreasing data when requested.
#[test]
fn test_isotonic_decreasing() {
    let n = 100;
    let mut rng = StdRng::seed_from_u64(9702);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / 10.0;
        x_vec.push(x);
        y_vec.push(10.0 - x + noise.sample(&mut rng));
    }
    let x = Array1::from_vec(x_vec);
    let y = Array1::from_vec(y_vec);

    let result = IsotonicRegression::fit(&x, &y, false, None).unwrap();

    for i in 1..n {
        assert!(result.fitted[i] <= result.fitted[i - 1]);
    }

    let ss_res: f64 = y
        .iter()
        .zip(result.fitted.iter())
        .map(|(&y, &f)| (y - f).powi(2))
        .sum();
    let y_mean = y.mean().unwrap();
    let ss_tot: f64 = y.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    assert!(r2 > 0.85, "r2 = {}", r2);
    assert!(result.increasing == false);
}

/// Weighted isotonic regression and input validation.
#[test]
fn test_isotonic_weighted_and_validation() {
    let y = Array1::from_vec(vec![1.0; 10]);
    let x_short = Array1::from_vec(vec![1.0; 5]);
    assert!(IsotonicRegression::fit(&x_short, &y, true, None).is_err());

    let x = Array1::from_vec(vec![1.0; 2]);
    let y2 = Array1::from_vec(vec![2.0, 1.0]);
    let w = Array1::from_vec(vec![1.0, 1.0]);
    let result = IsotonicRegression::fit(&x, &y2, true, Some(&w)).unwrap();
    assert!(result.fitted[0] <= result.fitted[1]);
}
