use greeners::{BayesMixedGLM, MixedLM};
use ndarray::{Array1, Array2};

/// MixedLM returns finite fixed and random effects with the expected shapes.
#[test]
fn test_mixed_lm_invariants() {
    let n = 20;
    let y: Array1<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * (i as f64) + (if i < 10 { 0.5 } else { -0.5 }))
        .collect();
    let x_fixed =
        Array2::from_shape_vec((n, 2), (0..n).flat_map(|i| vec![1.0, i as f64]).collect()).unwrap();
    let groups = Array1::from_vec((0..n).map(|i| if i < 10 { 0 } else { 1 }).collect());
    let x_random = Array2::ones((n, 1));

    let r = MixedLM::fit_with_names(
        &y,
        &x_fixed,
        &groups,
        &x_random,
        Some(vec!["intercept".into(), "x".into()]),
    )
    .unwrap();

    assert_eq!(r.n_obs, n);
    assert_eq!(r.n_groups, 2);
    assert_eq!(r.fixed_effects.len(), 2);
    assert!(r.fixed_effects.iter().all(|v| v.is_finite()));
    assert!(r.fixed_se.iter().all(|v| v.is_finite()));
    assert!(r.z_values.iter().all(|v| v.is_finite()));
    assert!(r
        .p_values
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    assert_eq!(r.var_random.shape(), [1, 1]);
    assert!(r.var_random.iter().all(|v| v.is_finite()));
    assert!(r.var_resid.is_finite() && r.var_resid > 0.0);
    assert!(r.random_effects.len() == 2);
    assert!(r.log_likelihood.is_finite());
    assert!(r.aic.is_finite());
    assert!(r.bic.is_finite());
}

/// BayesMixedGLM returns finite posterior summaries for binomial data.
#[test]
fn test_bayes_mixed_glm_invariants() {
    let n = 40;
    let x_fixed = Array2::from_shape_vec(
        (n, 2),
        (0..n).flat_map(|i| vec![1.0, i as f64 / 10.0]).collect(),
    )
    .unwrap();
    let groups = Array1::from_vec((0..n).map(|i| i / 20).collect());
    let group_effect: Array1<f64> = (0..n).map(|i| if i < 20 { 0.3 } else { -0.3 }).collect();

    let base =
        &x_fixed.column(0).to_owned() * 0.5 + &x_fixed.column(1).to_owned() * 2.0 + &group_effect;
    let y = base.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

    let r = BayesMixedGLM::fit(&y, &x_fixed, &groups, "binomial").unwrap();

    assert_eq!(r.n_obs, n);
    assert_eq!(r.n_groups, 2);
    assert_eq!(r.posterior_mean.len(), 2);
    assert_eq!(r.posterior_sd.len(), 2);
    assert!(r.posterior_mean.iter().all(|v| v.is_finite()));
    assert!(r.posterior_sd.iter().all(|v| v.is_finite()));
    assert!(r.random_effects.len() == 2);
    assert!(r.random_effects_sd.len() == 2);
    assert!(r.log_likelihood.is_finite());
}

/// Input validation rejects dimension mismatches for MixedLM.
#[test]
fn test_mixed_input_validation() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x_fixed = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
    let groups = Array1::from_vec(vec![0; 5]);
    let x_random = Array2::ones((5, 1));
    assert!(MixedLM::fit(&y, &x_fixed, &groups, &x_random).is_err());

    let x_good = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
    let groups_bad = Array1::from_vec(vec![0; 4]);
    assert!(MixedLM::fit(&y, &x_good, &groups_bad, &x_random).is_err());

    // x_random rows do not match y
    let x_random_bad = Array2::ones((4, 1));
    assert!(MixedLM::fit(&y, &x_good, &groups, &x_random_bad).is_err());
}
