use greeners_glm::conditional::ConditionalLogit;
use greeners_glm::conditional::ConditionalMNLogit;
use greeners_glm::conditional::ConditionalPoisson;
use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Normal, Poisson};

/// Conditional logit returns finite coefficients and group diagnostics.
#[test]
fn test_conditional_logit_invariants() {
    let mut rng = StdRng::seed_from_u64(12345);
    let noise = Normal::new(0.0, 0.1).unwrap();

    let n_groups = 60;
    let size = 4;
    let n = n_groups * size;
    let k = 1;
    let true_beta = 1.0;

    let mut x = Vec::with_capacity(n * k);
    let mut y = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);

    for g in 0..n_groups {
        let group_x: Vec<f64> = (0..size)
            .map(|i| (i as f64) * 0.5 + noise.sample(&mut rng))
            .collect();
        let exp_xb: Vec<f64> = group_x.iter().map(|&xi| (true_beta * xi).exp()).collect();
        let sum = exp_xb.iter().sum::<f64>();
        let probs: Vec<f64> = exp_xb.iter().map(|&v| v / sum).collect();
        let dist = WeightedIndex::new(&probs).unwrap();
        let chosen = dist.sample(&mut rng);

        for (i, &xi) in group_x.iter().enumerate() {
            groups.push(g);
            x.push(xi);
            y.push(if i == chosen { 1.0 } else { 0.0 });
        }
    }

    let y_arr = Array1::from_vec(y);
    let x_arr = Array2::from_shape_vec((n, k), x).unwrap();
    let result =
        ConditionalLogit::fit_with_names(&y_arr, &x_arr, &groups, Some(vec!["x1".into()])).unwrap();

    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_groups, n_groups);
    assert_eq!(result.params.len(), k);
    assert!(result.params.iter().all(|v| v.is_finite()));
    assert!(result.std_errors.iter().all(|v| v.is_finite()));
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

/// Conditional Poisson returns finite coefficients for panel count data.
#[test]
fn test_conditional_poisson_invariants() {
    let mut rng = StdRng::seed_from_u64(23456);
    let noise = Normal::new(0.0, 0.2).unwrap();

    let n_groups = 50;
    let size = 4;
    let n = n_groups * size;
    let k = 1;

    let mut x = Vec::with_capacity(n * k);
    let mut y = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);

    for g in 0..n_groups {
        let base = (g as f64) * 0.02;
        for i in 0..size {
            groups.push(g);
            let x1 = base + (i as f64) * 0.1 + noise.sample(&mut rng);
            x.push(x1);
            let lambda = (0.5 + 0.1 * x1).exp();
            let pois = Poisson::new(lambda.max(0.5)).unwrap();
            let count = pois.sample(&mut rng) as f64;
            y.push(count);
        }
    }

    let y_arr = Array1::from_vec(y);
    let x_arr = Array2::from_shape_vec((n, k), x).unwrap();
    let result = ConditionalPoisson::fit(&y_arr, &x_arr, &groups).unwrap();

    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_groups, n_groups);
    assert_eq!(result.params.len(), k);
    assert!(result.params.iter().all(|v| v.is_finite()));
    assert!(result.log_likelihood.is_finite());
}

/// Conditional multinomial logit returns coefficients for repeated choice sets.
#[test]
fn test_conditional_mnlogit_invariants() {
    let mut rng = StdRng::seed_from_u64(34567);
    let noise = Normal::new(0.0, 0.05).unwrap();

    let n_occasions = 40;
    let n_alts = 3;
    let k = 1;
    let true_beta = 1.5;
    let n_rows = n_occasions * n_alts;

    let mut x = Vec::with_capacity(n_rows * k);
    let mut y = Vec::with_capacity(n_occasions);
    let mut groups = Vec::with_capacity(n_rows);

    for occ in 0..n_occasions {
        let alt_x: Vec<f64> = (0..n_alts)
            .map(|a| a as f64 * 0.5 + noise.sample(&mut rng))
            .collect();
        let exp_xb: Vec<f64> = alt_x.iter().map(|&xi| (true_beta * xi).exp()).collect();
        let sum = exp_xb.iter().sum::<f64>();
        let probs: Vec<f64> = exp_xb.iter().map(|&v| v / sum).collect();
        let dist = WeightedIndex::new(&probs).unwrap();
        let chosen = dist.sample(&mut rng);

        y.push(chosen as f64);
        for &xi in &alt_x {
            groups.push(occ);
            x.push(xi);
        }
    }

    let y_arr = Array1::from_vec(y);
    let x_arr = Array2::from_shape_vec((n_rows, k), x).unwrap();
    let result = ConditionalMNLogit::fit(&y_arr, &x_arr, &groups, n_alts).unwrap();

    assert_eq!(result.n_obs, n_rows);
    assert_eq!(result.n_groups, n_occasions);
    assert_eq!(result.params.len(), k);
    assert!(result.params.iter().all(|v| v.is_finite()));
    assert!(result.log_likelihood.is_finite());
}

/// Input validation rejects mismatched dimensions, empty groups and degenerate choices.
#[test]
fn test_conditional_input_validation() {
    let y = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
    let x = Array2::from_shape_vec((4, 1), vec![0.5, 1.5, -0.5, 0.0]).unwrap();
    let groups = vec![0, 0, 1, 1];

    let short_groups = vec![0, 0, 1];
    assert!(ConditionalLogit::fit(&y, &x, &short_groups).is_err());

    // No groups with variation in y for logit
    let y_const = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
    assert!(ConditionalLogit::fit(&y_const, &x, &groups).is_err());

    // Poisson with all zero counts
    let y_zero = Array1::from_vec(vec![0.0; 4]);
    assert!(ConditionalPoisson::fit(&y_zero, &x, &groups).is_err());

    // MNLogit with mismatched number of groups and y length
    let y_mn = Array1::from_vec(vec![0.0; 2]);
    assert!(ConditionalMNLogit::fit(&y_mn, &x, &groups, 2).is_err());
}
