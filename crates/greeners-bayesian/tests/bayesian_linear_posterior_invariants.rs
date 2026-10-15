use greeners_bayesian::BayesianLinear;
use ndarray::{array, Array1, Array2};

fn reference_fit() -> greeners_bayesian::BayesianLinearResult {
    BayesianLinear::fit_with_prior(
        &array![-2.8, -1.2, 1.1, 2.7, 5.3, 6.8],
        &array![[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]],
        Some(&array![0.3, -0.2]),
        Some(&array![[2.0, 0.4], [0.4, 1.5]]),
        Some(2.3),
        Some(0.7),
        Some(vec!["x".into()]),
    )
    .unwrap()
}

// Independent NumPy/SciPy conjugate reference, with correlated prior and nonzero
// prior mean. Constants use scipy.special.gammaln and scipy.stats.t, not Greeners.
#[test]
fn posterior_covariance_includes_marginal_noise_scale() {
    let r = reference_fit();
    let expected = array![
        [0.0903073601816001, -0.01310392717431367],
        [-0.01310392717431367, 0.029919557858593036]
    ];
    for (actual, expected) in r.beta_cov.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
    assert!((r.beta[0] - 1.020152002338497).abs() < 1e-12);
    assert!((r.beta[1] - 1.8980765857936277).abs() < 1e-12);
    assert!((r.sigma2_scale - 2.3739248757673197).abs() < 1e-12);
    assert!((r.sigma2_shape - 5.3).abs() < 1e-12);
    assert!((r.sigma2 - 0.5520755525040278).abs() < 1e-12);
    assert!((r.beta_ci[(0, 0)] - 0.42163258544313176).abs() < 1e-10);
    assert!((r.beta_ci[(1, 1)] - 2.2425805757423722).abs() < 1e-10);
}

#[test]
fn positive_probability_matches_the_credible_interval() {
    let r = reference_fit();
    assert!((r.p_positive[0] - 0.9983436498141873).abs() < 1e-10);
    assert!((r.p_positive[1] - 0.999999928044161).abs() < 1e-10);
    assert!(r.beta_ci[(1, 0)] > 0.0 && r.p_positive[1] > 0.975);
}

#[test]
fn log_evidence_uses_exact_gamma_and_prior_posterior_determinant_ratio() {
    assert!((reference_fit().log_marginal - -10.34817515303829).abs() < 1e-10);
}

#[test]
fn invalid_priors_inputs_and_name_counts_are_rejected() {
    let y = array![1.0, 2.0, 2.8, 4.1];
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    for v in [
        array![[1.0, 2.0], [2.0, 1.0]],
        array![[1.0, 0.5], [0.0, 1.0]],
        array![[0.0, 0.0], [0.0, 1.0]],
    ] {
        assert!(BayesianLinear::fit_with_prior(&y, &x, None, Some(&v), None, None, None).is_err());
    }
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(BayesianLinear::fit_with_prior(&y, &x, None, None, Some(bad), None, None).is_err());
        assert!(BayesianLinear::fit_with_prior(&y, &x, None, None, None, Some(bad), None).is_err());
    }
    let mut bad_y = y.clone();
    bad_y[0] = f64::NAN;
    assert!(BayesianLinear::fit(&bad_y, &x, None).is_err());
    assert!(BayesianLinear::fit(&y, &x, Some(vec![])).is_err());
}

#[test]
fn proper_prior_handles_collinear_predictors_and_tiny_scale() {
    let x = array![
        [-2.0, -2.0],
        [-1.0, -1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ];
    let r = BayesianLinear::fit(&array![-3.0, -1.0, 1.0, 3.0, 5.0], &x, None).unwrap();
    assert!(r.beta_cov.iter().all(|v| v.is_finite()));
    let x = Array2::zeros((4, 0));
    for sign in [-1.0, 1.0] {
        let r = BayesianLinear::fit_with_prior(
            &Array1::from_elem(4, sign * 1e-12),
            &x,
            Some(&array![sign * 1e-12]),
            Some(&array![[1.0]]),
            Some(2.0),
            Some(1e-30),
            None,
        )
        .unwrap();
        assert!(if sign > 0.0 {
            r.p_positive[0] > 0.999
        } else {
            r.p_positive[0] < 0.001
        });
    }
    let r = BayesianLinear::fit(&Array1::zeros(4), &x, None).unwrap();
    assert_eq!(r.p_positive[0], 0.5);
}

#[test]
fn large_signal_does_not_cancel_the_posterior_residual_scale() {
    let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0]];
    let y = array![
        999999998.1,
        999999998.8,
        1000000000.1,
        1000000001.1,
        1000000001.9
    ];
    let r = BayesianLinear::fit_with_prior(
        &y,
        &x,
        Some(&array![1e9, 1.0]),
        Some(&Array2::eye(2)),
        Some(2.0),
        Some(0.5),
        None,
    )
    .unwrap();
    assert!(
        r.sigma2_scale >= 0.5 && r.sigma2_scale < 0.6,
        "{}",
        r.sigma2_scale
    );
    assert!(r.log_marginal.is_finite());
}

#[test]
fn posterior_sign_probabilities_and_intervals_reflect_response_sign() {
    let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0]];
    let y = array![-1.8, -1.1, 0.2, 1.3, 1.9];
    let positive = BayesianLinear::fit(&y, &x, None).unwrap();
    let negative = BayesianLinear::fit(&(-&y), &x, None).unwrap();
    for j in 0..2 {
        assert!((positive.p_positive[j] + negative.p_positive[j] - 1.0).abs() < 1e-12);
        assert!((positive.beta[j] + negative.beta[j]).abs() < 1e-12);
        assert!((positive.beta_ci[(j, 0)] + negative.beta_ci[(j, 1)]).abs() < 1e-12);
    }
    assert!((positive.log_marginal - negative.log_marginal).abs() < 1e-12);
}

#[test]
fn log_evidence_does_not_form_overflowing_or_underflowing_determinants() {
    let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0]];
    let y = array![-1.8, -1.1, 0.2, 1.3, 1.9];
    // The 2x2 prior determinant would be 1e400 or 1e-400 in ordinary arithmetic.
    for prior_scale in [1e200, 1e-200] {
        let r = BayesianLinear::fit_with_prior(
            &y,
            &x,
            None,
            Some(&(Array2::eye(2) * prior_scale)),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(r.log_marginal.is_finite());
        assert!(r.beta_cov.iter().all(|v| v.is_finite()));
        assert!(r.beta_ci.iter().all(|v| v.is_finite()));
        assert!(r.beta_cov.diag().iter().all(|&v| v > 0.0));
    }
}

#[test]
fn prior_symmetry_is_checked_in_each_covariance_block() {
    let x = array![[-2.0, 4.0], [-1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 4.0]];
    let y = array![-1.8, -1.1, 0.2, 1.3, 1.9];
    let prior = array![[1e20, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]];
    assert!(BayesianLinear::fit_with_prior(&y, &x, None, Some(&prior), None, None, None).is_err());
}

#[test]
fn marginal_covariance_underflow_is_not_reported_as_zero_uncertainty() {
    let x = Array2::zeros((4, 0));
    // The t scale is representable (~1e-200), but its squared marginal moment
    // is not (~1e-400). The covariance/SD API must report this numerical limit.
    assert!(BayesianLinear::fit_with_prior(
        &Array1::zeros(4),
        &x,
        None,
        Some(&array![[1e-200]]),
        Some(2.0),
        Some(1e-200),
        None,
    )
    .is_err());
}

fn relative_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} differs from reference {expected}"
    );
}

// F1: an almost uninformative, distant proper prior must not erase the response.
// Independent 300-digit conjugate reference; intervals use scipy.stats.t(df=8).
#[test]
fn distant_diffuse_prior_does_not_discard_the_response() {
    for sign in [-1.0, 1.0] {
        let y = array![0.5, 1.5, 0.5, 1.5] * sign;
        let r = BayesianLinear::fit_with_prior(
            &y,
            &Array2::zeros((4, 0)),
            Some(&array![sign * 1e16]),
            Some(&array![[1e200]]),
            Some(2.0),
            Some(0.5),
            None,
        )
        .unwrap();
        relative_close(r.beta[0], sign, 1e-12);
        for value in &r.fitted {
            relative_close(*value, sign, 1e-12);
        }
        relative_close(r.sigma2_scale, 1.0, 1e-12);
        relative_close(r.sigma2, 1.0 / 3.0, 1e-12);
        relative_close(r.beta_cov[[0, 0]], 1.0 / 12.0, 1e-12);
        relative_close(r.beta_ci[[0, 0]], sign - 0.5765010338010415, 1e-11);
        relative_close(r.beta_ci[[0, 1]], sign + 0.5765010338010415, 1e-11);
        relative_close(
            r.p_positive[0],
            if sign > 0.0 {
                0.9980251135982773
            } else {
                0.001974886401722661
            },
            1e-12,
        );
        relative_close(r.log_marginal, -234.22194550467506, 1e-12);
    }
}

// F4: offsets change coefficient coordinates, not the resolved local variation.
// The proper 1e200*I prior changes these limiting OLS expressions by <1e-170;
// 300-digit reference arithmetic establishes the same displayed f64 targets.
#[test]
fn posterior_resolves_predictor_variation_across_large_offsets() {
    let y = array![-1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
    let expected_probabilities = [
        0.9999921983340261,
        5.8162573812892446e-8,
        5.8107863080694525e-8,
        5.810732922382052e-8,
        5.8107316337705075e-8,
    ];
    for (offset, intercept_probability) in [0.0, 1e4, 1e6, 3e7, 1e8]
        .into_iter()
        .zip(expected_probabilities)
    {
        let x = Array2::from_shape_fn((6, 1), |(i, _)| offset + i as f64 - 2.5);
        let r = BayesianLinear::fit_with_prior(
            &y,
            &x,
            None,
            Some(&(Array2::eye(2) * 1e200)),
            Some(2.0),
            Some(0.5),
            None,
        )
        .unwrap();
        relative_close(r.beta[0], 1.0 - offset, 1e-10);
        relative_close(r.beta[1], 1.0, 1e-10);
        for (actual, expected) in r.fitted.iter().zip(y.iter()) {
            relative_close(*actual, *expected, 1e-9);
        }
        relative_close(r.sigma2_scale, 0.5, 1e-11);
        relative_close(r.sigma2, 0.125, 1e-11);
        let v00 = 0.125 * (1.0 / 6.0 + offset * offset / 17.5);
        relative_close(r.beta_cov[[0, 0]], v00, 1e-10);
        relative_close(r.beta_cov[[0, 1]], -offset / 140.0, 1e-10);
        relative_close(r.beta_cov[[1, 0]], -offset / 140.0, 1e-10);
        relative_close(r.beta_cov[[1, 1]], 1.0 / 140.0, 1e-11);
        // t(10) 0.975 quantile, independently calculated by SciPy.
        let intercept_margin = 2.2281388519649385 * (v00 * 0.8).sqrt();
        relative_close(r.beta_ci[[0, 0]], 1.0 - offset - intercept_margin, 1e-10);
        relative_close(r.beta_ci[[0, 1]], 1.0 - offset + intercept_margin, 1e-10);
        relative_close(r.beta_ci[[1, 0]], 0.8315685346051375, 1e-10);
        relative_close(r.beta_ci[[1, 1]], 1.1684314653948624, 1e-10);
        relative_close(r.p_positive[0], intercept_probability, 1e-12);
        relative_close(r.p_positive[1], 0.9999999418926891, 1e-12);
        relative_close(r.log_marginal, -463.10013460108814, 1e-12);
    }
}

#[test]
fn correlated_proper_prior_identifies_rank_deficient_likelihood() {
    let x = Array2::from_shape_fn((5, 2), |(i, _)| i as f64 - 2.0);
    let y = array![-3.0, -1.0, 1.0, 3.0, 5.0];
    let r = BayesianLinear::fit_with_prior(
        &y,
        &x,
        Some(&array![0.4, 0.1, -0.2]),
        Some(&array![[2.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.5]]),
        Some(2.0),
        Some(0.5),
        None,
    )
    .unwrap();
    // Independent mpmath(300 digits) conjugate reference; SciPy t(9) summaries.
    let mean = [0.9513513513513514, 0.9918918918918919, 0.9432432432432433];
    let covariance = array![
        [
            0.06513639833696733,
            0.0051986796794762655,
            -0.00509674478380026
        ],
        [
            0.0051986796794762655,
            0.16493066120377642,
            -0.1504559060177837
        ],
        [
            -0.00509674478380026,
            -0.1504559060177837,
            0.17068998280947073
        ]
    ];
    let ci = [
        [0.4421818173266341, 1.4605208853760687],
        [0.18167510057722924, 1.8021086832065545],
        [0.11900159218520212, 1.7674848943012844],
    ];
    let probability = [0.9988913017912566, 0.9891125913332047, 0.985363542642521];
    for j in 0..3 {
        relative_close(r.beta[j], mean[j], 1e-12);
        relative_close(r.p_positive[j], probability[j], 1e-11);
        for (end, expected) in ci[j].iter().enumerate() {
            relative_close(r.beta_ci[[j, end]], *expected, 1e-10);
        }
    }
    for (actual, expected) in r.beta_cov.iter().zip(covariance.iter()) {
        relative_close(*actual, *expected, 1e-12);
    }
    let fitted = [
        -2.918918918918919,
        -0.9837837837837838,
        0.9513513513513514,
        2.8864864864864863,
        4.821621621621621,
    ];
    for (actual, expected) in r.fitted.iter().zip(fitted) {
        relative_close(*actual, expected, 1e-12);
    }
    relative_close(r.sigma2_scale, 1.2540540540540541, 1e-12);
    relative_close(r.sigma2, 0.3583011583011583, 1e-12);
    relative_close(r.log_marginal, -7.477072271572105, 1e-12);
}

#[test]
fn unresolved_augmented_posterior_returns_a_precision_error() {
    let x = Array2::from_shape_fn((5, 2), |(i, _)| i as f64 - 2.0);
    // A proper prior mathematically identifies the duplicated slopes, but its
    // 1e-100 precision factor is below double-precision resolution here.
    let error = BayesianLinear::fit_with_prior(
        &array![-3., -1., 1., 3., 5.],
        &x,
        None,
        Some(&(Array2::eye(3) * 1e200)),
        Some(2.0),
        Some(0.5),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("numerically unresolved"));
}

#[test]
fn predictor_offsets_do_not_degrade_a_concentrated_prior() {
    let x = Array2::from_shape_fn((6, 1), |(i, _)| 1e8 + i as f64 - 2.5);
    let y = array![-1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
    let r = BayesianLinear::fit_with_prior(
        &y,
        &x,
        Some(&array![1.0, 1.0]),
        Some(&(Array2::eye(2) * 1e-200)),
        Some(2.0),
        Some(0.5),
        None,
    )
    .unwrap();
    for j in 0..2 {
        relative_close(r.beta[j], 1.0, 1e-12);
        relative_close(r.beta_cov[[j, j]] / 7.5e-185, 1.0, 1e-12);
    }
    relative_close(r.sigma2_scale, 3e16, 1e-12);
}

#[test]
fn concentrated_nonbinary_prior_preserves_residual_scale_across_offsets() {
    let y = array![-1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
    // Independent 300-digit conjugate algebra uses the exact represented
    // inputs. The correction to beta0 is below binary64 precision here.
    for prior_shape in [Array2::eye(2), array![[1.0, 0.4], [0.4, 2.0]]] {
        for (offset, bn, evidence) in [
            (0.0, 14.57, -17.11669483103555),
            (1e4, 12008414.570000002, -85.22746261091577),
            (1e8, 1200000084000014.8, -177.32736183852322),
        ] {
            let x = Array2::from_shape_fn((6, 1), |(i, _)| offset + i as f64 - 2.5);
            let r = BayesianLinear::fit_with_prior(
                &y,
                &x,
                Some(&array![0.3, -0.2]),
                Some(&(&prior_shape * 1e-200)),
                Some(2.0),
                Some(0.5),
                None,
            )
            .unwrap();
            for (j, mean) in [0.3, -0.2].iter().enumerate() {
                assert_eq!(r.beta[j], *mean);
                for end in 0..2 {
                    assert_eq!(r.beta_ci[[j, end]], *mean);
                }
                relative_close(r.p_positive[j], if j == 0 { 1.0 } else { 0.0 }, 1e-12);
            }
            for i in 0..6 {
                relative_close(r.fitted[i], 0.3 - 0.2 * x[[i, 0]], 1e-12);
            }
            relative_close(r.sigma2_shape, 5.0, 1e-12);
            relative_close(r.sigma2_scale / bn, 1.0, 1e-12);
            relative_close(r.sigma2 / (bn / 4.0), 1.0, 1e-12);
            relative_close(r.log_marginal, evidence, 1e-12);
            for i in 0..2 {
                for j in 0..2 {
                    relative_close(
                        r.beta_cov[[i, j]] / (bn / 4.0 * 1e-200),
                        prior_shape[[i, j]],
                        1e-12,
                    );
                }
            }
        }
    }
}

#[test]
fn concentrated_prior_residual_is_not_subtracted_from_rounded_fitted_values() {
    let r = BayesianLinear::fit_with_prior(
        &Array1::from_elem(6, 1e16),
        &Array2::from_elem((6, 1), 1e16),
        Some(&array![1.0, 1.0]),
        Some(&(Array2::eye(2) * 1e-200)),
        Some(2.0),
        Some(0.5),
        None,
    )
    .unwrap();
    // Each exact represented-input residual is -1 although the fitted value
    // 1e16 + 1 rounds to 1e16. The tiny posterior correction is negligible.
    assert_eq!(r.beta, array![1.0, 1.0]);
    relative_close(r.sigma2_scale, 3.5, 1e-12);
    relative_close(r.sigma2, 0.875, 1e-12);
}

#[test]
fn concentrated_correlated_prior_resolves_a_zero_mean_coordinate() {
    let r = BayesianLinear::fit_with_prior(
        &array![-1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
        &Array2::from_shape_fn((6, 1), |(i, _)| 1e8 + i as f64 - 2.5),
        Some(&array![0.0, 0.3]),
        Some(&(array![[1.0, 0.4], [0.4, 2.0]] * 1e-200)),
        Some(2.0),
        Some(0.5),
        None,
    )
    .unwrap();
    // 300-digit conjugate reference for exact represented input values.
    relative_close(r.beta[0] / -7.199999939999989e-185, 1.0, 1e-12);
    assert_eq!(r.beta[1], 0.3);
    relative_close(r.sigma2_scale, 2699999820000007.5, 1e-12);
    relative_close(r.beta_cov[[0, 0]] / 6.749999550000019e-186, 1.0, 1e-12);
    relative_close(r.beta_ci[[0, 0]] / -5.177726627427863e-93, 1.0, 1e-10);
    relative_close(r.beta_ci[[0, 1]] / 5.177726627427863e-93, 1.0, 1e-10);
    relative_close(r.p_positive[0], 0.5, 1e-12);
    relative_close(r.log_marginal, -181.3820122362715, 1e-12);
}
