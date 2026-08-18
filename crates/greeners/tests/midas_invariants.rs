use greeners::Midas;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn make_midas_data(seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = make_rng(seed);
    let n_low = 20;
    let freq = 3;
    let n_high = n_low * freq;
    let dist = Uniform::new(0.0, 1.0);
    let x: Vec<f64> = (0..n_high).map(|_| dist.sample(&mut rng)).collect();
    let mut y = Array1::zeros(n_low);
    for t in 0..n_low {
        let base = t * freq + (freq - 1);
        let midas = 0.6 * x[base] + 0.3 * x[base.saturating_sub(1)];
        let e: f64 = StandardNormal.sample(&mut rng);
        y[t] = 0.1 + 2.0 * midas + 0.2 * e;
    }
    (y, Array1::from_vec(x))
}

/// MIDAS returns finite coefficients and expected shapes.
#[test]
fn test_midas_shape_and_finite() {
    let (y, x) = make_midas_data(13001);
    let r = Midas::fit(&y, &x, 3, 2, 1).unwrap();
    assert_eq!(r.n_obs, y.len());
    assert_eq!(r.n_lags, 2);
    assert_eq!(r.freq_ratio, 3);
    assert_eq!(r.weights.len(), 2);
    assert_eq!(r.gamma.len(), 1);
    assert!(r.alpha.is_finite());
    assert!(r.beta.is_finite());
    assert!(r.weights.iter().all(|v| v.is_finite()));
    assert!(r.weights.iter().all(|v| *v >= 0.0));
    assert!((r.weights.sum() - 1.0).abs() < 1e-6);
    assert!(r.r_squared >= 0.0 && r.r_squared <= 1.0);
}

/// MIDAS recovers positive beta on a known positive relationship.
#[test]
fn test_midas_positive_beta() {
    let (y, x) = make_midas_data(13002);
    let r = Midas::fit(&y, &x, 3, 2, 1).unwrap();
    assert!(
        r.beta > 0.0,
        "beta should be positive for positively correlated data"
    );
}

/// Input validation rejects insufficient high-frequency data and invalid parameters.
#[test]
fn test_midas_input_validation() {
    let y = Array1::from_vec(vec![0.0; 5]);
    let x = Array1::from_vec(vec![0.0; 10]);
    assert!(Midas::fit(&y, &x, 3, 2, 1).is_err());

    let y2 = Array1::from_vec(vec![0.0; 20]);
    let x2 = Array1::from_vec(vec![0.0; 60]);
    assert!(Midas::fit(&y2, &x2, 0, 2, 1).is_err());
    assert!(Midas::fit(&y2, &x2, 3, 0, 1).is_err());
    assert!(Midas::fit(&y2, &x2, 3, 2, 0).is_err());
}
