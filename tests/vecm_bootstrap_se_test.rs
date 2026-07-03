use greeners::VECM;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::prelude::*;

#[test]
fn test_vecm_bootstrap_standard_errors() {
    let t = 300;
    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();

    let mut data = Array2::<f64>::zeros((t, 2));

    // Cointegrated system: y1 = cumsum(e1), y2 = 0.5*y1 + e2
    // True long-run relationship: y1 - 2*y2 = 0  => beta proportional to [1, -2]
    let mut y1 = 0.0;
    for tt in 0..t {
        y1 += norm.sample(&mut rng);
        let y2 = 0.5 * y1 + norm.sample(&mut rng);
        data[[tt, 0]] = y1;
        data[[tt, 1]] = y2;
    }

    // VECM(1) in differences => 2 lags in levels
    let model = VECM::fit(&data, 2, 1).unwrap();
    let inferred = model.with_inference(100).unwrap();

    println!("{}", inferred);

    // Standard errors must be finite and positive
    for &se in inferred.std_errors_beta.iter() {
        assert!(se.is_finite());
        assert!(se > 0.0);
    }

    // Normalize estimated beta so that the first element is positive and equal to 1
    let b0 = inferred.beta[[0, 0]];
    let b1 = inferred.beta[[1, 0]];
    assert!(b0.abs() > 1e-6, "beta element too small to normalize");
    let ratio = b1 / b0;
    let se_ratio = (inferred.std_errors_beta[[1, 0]].abs() / b0.abs()
        + inferred.std_errors_beta[[0, 0]].abs() * b1.abs() / (b0 * b0))
        .max(1e-6);

    // True normalized cointegration vector is [1, -2]
    assert!(
        (ratio - (-2.0)).abs() < 2.0 * se_ratio,
        "normalized beta {} not within ~2 SE of -2 (se approx {})",
        ratio,
        se_ratio
    );
}
