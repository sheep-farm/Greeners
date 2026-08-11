use greeners::ConformalPrediction;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_conformal_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let x_vec: Vec<f64> = (0..n * 2).map(|_| dist.sample(&mut rng)).collect();
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let noise: f64 = StandardNormal.sample(&mut rng);
        y[i] = 1.0 + 2.0 * x[(i, 0)] - 1.5 * x[(i, 1)] + 0.4 * noise;
    }
    let x_test_vec: Vec<f64> = (0..10 * 2).map(|_| dist.sample(&mut rng)).collect();
    let x_test = Array2::from_shape_vec((10, 2), x_test_vec).unwrap();
    (y, x, x_test)
}

/// Conformal prediction returns intervals that contain the point prediction.
#[test]
fn test_conformal_shape_and_finite() {
    let (y, x, x_test) = make_conformal_data(60, 20001);
    let r = ConformalPrediction::fit(&y, &x, &x_test, Some(0.1), Some(0.3), None).unwrap();
    assert_eq!(r.n_test, x_test.nrows());
    assert_eq!(r.n_train + r.n_calib, y.len());
    assert_eq!(r.predictions.len(), x_test.nrows());
    assert_eq!(r.lower.len(), x_test.nrows());
    assert_eq!(r.upper.len(), x_test.nrows());
    assert!(r.predictions.iter().all(|v| v.is_finite()));
    assert!(r.lower.iter().all(|v| v.is_finite()));
    assert!(r.upper.iter().all(|v| v.is_finite()));
    assert!(r.lower.iter().zip(r.upper.iter()).all(|(l, u)| l <= u));
    assert!(r
        .lower
        .iter()
        .zip(r.predictions.iter())
        .all(|(l, p)| l <= p));
    assert!(r
        .upper
        .iter()
        .zip(r.predictions.iter())
        .all(|(u, p)| p <= u));
    assert!(r.alpha > 0.0 && r.alpha < 1.0);
    assert!(r.empirical_coverage >= 0.0 && r.empirical_coverage <= 1.0);
}

/// Coverage level is the complement of miscoverage.
#[test]
fn test_conformal_coverage_identity() {
    let (y, x, x_test) = make_conformal_data(60, 20002);
    let r = ConformalPrediction::fit(&y, &x, &x_test, Some(0.1), None, None).unwrap();
    assert!((r.coverage - (1.0 - r.alpha)).abs() < 1e-12);
}

/// Input validation rejects mismatched shapes and too few observations.
#[test]
fn test_conformal_input_validation() {
    let (y, x, x_test) = make_conformal_data(60, 20003);
    let x_bad = Array2::from_shape_vec((10, 3), vec![0.0; 30]).unwrap();
    assert!(ConformalPrediction::fit(&y, &x, &x_bad, None, None, None).is_err());
    let y_short = Array1::from_vec(vec![0.0; 5]);
    assert!(ConformalPrediction::fit(&y_short, &x, &x_test, None, None, None).is_err());
}
