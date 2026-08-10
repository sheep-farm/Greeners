use greeners::JohansenBreak;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn random_walk_y(t: usize, k: usize, seed: u64) -> Array2<f64> {
    let mut rng = make_rng(seed);
    let mut y = Array2::zeros((t, k));
    for j in 0..k {
        let v: f64 = StandardNormal.sample(&mut rng);
        y[(0, j)] = v;
    }
    for i in 1..t {
        for j in 0..k {
            let inc: f64 = StandardNormal.sample(&mut rng);
            y[(i, j)] = y[(i - 1, j)] + inc;
        }
    }
    y
}

/// JohansenBreak returns the expected output shapes and finite trace stats.
#[test]
fn test_johansen_shape_and_finite() {
    let y = random_walk_y(60, 2, 12001);
    let r = JohansenBreak::fit(&y, 1, &[]).unwrap();
    assert_eq!(r.n_obs, y.nrows() - r.lags - 1);
    assert_eq!(r.n_vars, y.ncols());
    assert_eq!(r.lags, 1);
    assert_eq!(r.trace_stats.len(), y.ncols());
    assert_eq!(r.lambda_max_stats.len(), y.ncols());
    assert_eq!(r.eigenvalues.len(), y.ncols());
    assert!(r.trace_stats.iter().all(|v| v.is_finite()));
    assert!(r.lambda_max_stats.iter().all(|v| v.is_finite()));
    assert!(r.cointegration_rank <= y.ncols());
}

/// Including break points produces the same rank space and records them.
#[test]
fn test_johansen_with_breaks() {
    let y = random_walk_y(60, 2, 12002);
    let r = JohansenBreak::fit(&y, 1, &[30]).unwrap();
    assert_eq!(r.n_breaks, 1);
    assert_eq!(r.break_points, vec![30]);
    assert_eq!(r.n_obs, y.nrows() - r.lags - 1);
    assert!(r.cointegration_rank <= y.ncols());
}

/// Input validation rejects insufficient observations or zero lags.
#[test]
fn test_johansen_input_validation() {
    let y = random_walk_y(6, 2, 12003);
    assert!(JohansenBreak::fit(&y, 1, &[]).is_err());
    let y2 = random_walk_y(60, 2, 12004);
    assert!(JohansenBreak::fit(&y2, 0, &[]).is_err());
}
