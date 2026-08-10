use greeners::DMLCrossfit;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::{Distribution, StandardNormal, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

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

/// DMLCrossfit returns a finite treatment effect and MSEs.
#[test]
fn test_dml_crossfit_shape_and_finite() {
    let (y, d, x) = make_dml_data(50, 17001);
    let r = DMLCrossfit::fit(&y, &d, &x, Some(3)).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.n_confounders, x.ncols());
    assert!(r.theta.is_finite());
    assert!(r.se.is_finite());
    assert!(r.g_mse >= 0.0);
    assert!(r.m_mse >= 0.0);
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
}

/// The confidence interval contains the point estimate.
#[test]
fn test_dml_crossfit_ci_contains_theta() {
    let (y, d, x) = make_dml_data(80, 17002);
    let r = DMLCrossfit::fit(&y, &d, &x, Some(4)).unwrap();
    assert!(r.ci[0] <= r.theta && r.theta <= r.ci[1]);
}

/// Input validation rejects mismatched sizes and too few observations.
#[test]
fn test_dml_crossfit_input_validation() {
    let (y, d, x) = make_dml_data(50, 17003);
    let d_bad = Array1::from_vec(vec![0.0; 10]);
    assert!(DMLCrossfit::fit(&y, &d_bad, &x, None).is_err());
    let y_short = Array1::from_vec(vec![0.0; 10]);
    assert!(DMLCrossfit::fit(&y_short, &d, &x, None).is_err());
}
