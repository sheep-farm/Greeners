use greeners_glm::mnlogit::MNLogit;
use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Multinomial logit recovers the coefficients used to generate three
/// categories.
#[test]
fn test_mnlogit_recovery() {
    let n = 600;
    let mut rng = StdRng::seed_from_u64(9801);

    let beta0_0 = 0.5; // const for category 0 vs base
    let beta1_0 = 1.5; // x effect for category 0
    let beta0_1 = -0.5; // const for category 1 vs base
    let beta1_1 = -1.0; // x effect for category 1

    let mut x_vec = Vec::with_capacity(n * 2);
    let mut y_vec = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.gen::<f64>() * 2.0 - 1.0;
        x_vec.push(1.0);
        x_vec.push(x);

        let eta0 = beta0_0 + beta1_0 * x;
        let eta1 = beta0_1 + beta1_1 * x;
        let eta2 = 0.0; // base category
        let max_eta = eta0.max(eta1).max(eta2);
        let e0 = (eta0 - max_eta).exp();
        let e1 = (eta1 - max_eta).exp();
        let e2 = (eta2 - max_eta).exp();
        let sum = e0 + e1 + e2;
        let p = [e0 / sum, e1 / sum, e2 / sum];
        let dist = WeightedIndex::new(&p).unwrap();
        y_vec.push(dist.sample(&mut rng) as f64);
    }

    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);

    let result = MNLogit::fit(&y, &x).unwrap();
    assert!(result.converged);
    assert_eq!(result.n_categories, 3);
    assert_eq!(result.params.shape(), &[2, 2]);

    // Column 0 = category 0, Column 1 = category 1.
    approx_zero((result.params[[0, 0]] - beta0_0).abs(), 0.3);
    approx_zero((result.params[[1, 0]] - beta1_0).abs(), 0.35);
    approx_zero((result.params[[0, 1]] - beta0_1).abs(), 0.35);
    approx_zero((result.params[[1, 1]] - beta1_1).abs(), 0.35);
    assert!(result.log_likelihood.is_finite());
    assert!(result.pseudo_r2.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

/// MNLogit input validation.
#[test]
fn test_mnlogit_input_validation() {
    let y = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0]);
    let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
    assert!(MNLogit::fit(&y, &x).is_err()); // only 2 categories

    let y_nan = Array1::from_vec(vec![0.0, 1.0, f64::NAN]);
    assert!(MNLogit::fit(&y_nan, &x).is_err());
}
