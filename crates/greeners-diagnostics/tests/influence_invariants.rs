use greeners_core::types::CovarianceType;
use greeners_diagnostics::influence::CUSUMTest;
use greeners_diagnostics::influence::Influence;
use greeners_ols::ols::OLS;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_ols_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let dist = Uniform::new(-1.0, 1.0);
    let mut x = Array2::zeros((n, 2));
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = dist.sample(&mut rng);
    }
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let noise: f64 = StandardNormal.sample(&mut rng);
        y[i] = 1.0 + 2.0 * x[(i, 1)] + 0.4 * noise;
    }
    (y, x)
}

/// Influence diagnostics return the expected shapes on OLS residuals.
#[test]
fn test_influence_shape_and_finite() {
    let (y, x) = make_ols_data(30, 14001);
    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let residuals = ols.residuals(&y, &x);
    let mse = ols.sigma * ols.sigma;
    let r = Influence::compute(&residuals, &x, mse).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.n_params, x.ncols());
    assert_eq!(r.dfbetas.shape(), &[y.len(), x.ncols()]);
    assert_eq!(r.dffits.len(), y.len());
    assert_eq!(r.leverage.len(), y.len());
    assert_eq!(r.student_resid.len(), y.len());
    assert!(r.dffits.iter().all(|v| v.is_finite()));
    assert!(r.leverage.iter().all(|v| v.is_finite()));
    assert!(r.student_resid.iter().all(|v| v.is_finite()));
    assert!(r.dfbetas.iter().all(|v| v.is_finite()));
    assert!(r.dffits_threshold() > 0.0);
    assert!(r.dfbetas_threshold() > 0.0);
}

/// CUSUM test returns expected shapes and a boolean stability flag.
#[test]
fn test_cusum_shape_and_finite() {
    let (y, x) = make_ols_data(50, 14002);
    let r = CUSUMTest::test(&y, &x).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.cusum.len(), r.upper_bound.len());
    assert_eq!(r.cusum.len(), r.lower_bound.len());
    assert!(r.cusum.iter().all(|v| v.is_finite()));
    assert!(r.upper_bound.iter().all(|v| v.is_finite()));
    assert!(r.lower_bound.iter().all(|v| v.is_finite()));
    assert!(r
        .upper_bound
        .iter()
        .zip(r.lower_bound.iter())
        .all(|(u, l)| u >= l));
}

/// Input validation rejects mismatched sizes and too few obs.
#[test]
fn test_influence_input_validation() {
    let (y, x) = make_ols_data(30, 14003);
    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let _residuals = ols.residuals(&y, &x);
    let mse = ols.sigma * ols.sigma;
    let residuals_bad = Array1::from_vec(vec![0.0; 5]);
    assert!(Influence::compute(&residuals_bad, &x, mse).is_err());

    let (y_small, x_small) = make_ols_data(3, 14004);
    assert!(CUSUMTest::test(&y_small, &x_small).is_err());
}
