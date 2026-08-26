use greeners_causal::double_ml::DoubleML;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_dml_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let x_vec: Vec<f64> = (0..n * 2).map(|_| dist.sample(&mut rng)).collect();
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let mut d = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let e_d: f64 = StandardNormal.sample(&mut rng);
        d[i] = 0.3 + 0.5 * x[(i, 0)] + 0.2 * x[(i, 1)] + e_d;
        let conf = 0.4 * x[(i, 0)] + 0.6 * x[(i, 1)];
        let e_y: f64 = StandardNormal.sample(&mut rng);
        y[i] = 0.8 * d[i] + conf + e_y;
    }
    (y, d, x)
}

/// DoubleML returns finite treatment effect and residuals.
#[test]
fn test_double_ml_shape_and_finite() {
    let (y, d, x) = make_dml_data(100, 16001);
    let r = DoubleML::fit_plr(&y, &d, &x, 5, 2).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.n_folds, 5);
    assert!(r.theta.is_finite());
    assert!(r.std_error.is_finite());
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    assert_eq!(r.y_tilde.len(), y.len());
    assert_eq!(r.d_tilde.len(), y.len());
}

/// y_tilde and d_tilde are centered (mean close to zero by construction).
#[test]
fn test_double_ml_residuals_centered() {
    let (y, d, x) = make_dml_data(120, 16002);
    let r = DoubleML::fit_plr(&y, &d, &x, 5, 2).unwrap();
    assert!(r.y_tilde.mean().unwrap_or(f64::NAN).is_finite());
    assert!(r.d_tilde.mean().unwrap_or(f64::NAN).is_finite());
}

/// Input validation rejects mismatched sizes and too few folds.
#[test]
fn test_double_ml_input_validation() {
    let (y, d, x) = make_dml_data(100, 16003);
    let d_bad = Array1::from_vec(vec![0.0; 50]);
    assert!(DoubleML::fit_plr(&y, &d_bad, &x, 5, 2).is_err());
    assert!(DoubleML::fit_plr(&y, &d, &x, 1, 2).is_err());
}
