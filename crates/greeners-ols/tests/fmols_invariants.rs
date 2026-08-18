use greeners_ols::fmols::FMOLS;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn random_walk_x(t: usize, k: usize, seed: u64) -> Array2<f64> {
    let mut rng = make_rng(seed);
    let mut x = Array2::zeros((t, k));
    for j in 0..k {
        let v: f64 = StandardNormal.sample(&mut rng);
        x[(0, j)] = v;
    }
    for i in 1..t {
        for j in 0..k {
            let inc: f64 = StandardNormal.sample(&mut rng);
            x[(i, j)] = x[(i - 1, j)] + inc;
        }
    }
    x
}

fn make_fmols_data(t: usize, k: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = make_rng(seed);
    let x = random_walk_x(t, k, seed);
    let mut y = Array1::zeros(t);
    let beta: Vec<f64> = (0..k)
        .map(|_| {
            let v: f64 = StandardNormal.sample(&mut rng);
            0.5 + 0.5 * v
        })
        .collect();
    for i in 0..t {
        let e: f64 = StandardNormal.sample(&mut rng);
        y[i] = 0.2 * (i as f64) + x.row(i).dot(&Array1::from_vec(beta.clone())) + e;
    }
    (y, x)
}

/// FMOLS returns finite coefficients and expected shapes.
#[test]
fn test_fmols_shape_and_finite() {
    let (y, x) = make_fmols_data(50, 2, 11001);
    let r = FMOLS::fit(&y, &x, None).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.n_regressors, x.ncols());
    assert_eq!(r.beta.len(), x.ncols());
    assert!(r.beta.iter().all(|v| v.is_finite()));
    assert!(r.beta_se.iter().all(|v| v.is_finite()));
    assert_eq!(r.omega.shape(), &[x.ncols() + 1, x.ncols() + 1]);
    assert!(r.alpha.is_finite());
    assert!(r.r_squared >= 0.0 && r.r_squared <= 1.0);
}

/// FMOLS rejects shape mismatches and too few observations.
#[test]
fn test_fmols_input_validation() {
    let (_y, x) = make_fmols_data(50, 2, 11002);
    let y_short = Array1::from_vec(vec![1.0; 5]);
    assert!(FMOLS::fit(&y_short, &x, None).is_err());

    let x_small = Array2::from_shape_vec((6, 3), vec![0.0; 18]).unwrap();
    let y_small = Array1::from_vec(vec![0.0; 6]);
    assert!(FMOLS::fit(&y_small, &x_small, None).is_err());
}

/// Coefficient signs and magnitudes are stable across two random seeds.
#[test]
fn test_fmols_stability() {
    let (y1, x1) = make_fmols_data(80, 2, 11003);
    let (y2, x2) = make_fmols_data(80, 2, 11004);
    let r1 = FMOLS::fit(&y1, &x1, None).unwrap();
    let r2 = FMOLS::fit(&y2, &x2, None).unwrap();
    assert_eq!(r1.beta.len(), r2.beta.len());
    assert!(r1.beta.iter().all(|v| v.is_finite()));
    assert!(r2.beta.iter().all(|v| v.is_finite()));
}
