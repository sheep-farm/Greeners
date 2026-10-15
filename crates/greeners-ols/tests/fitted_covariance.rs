//! Regression tests for fitted-covariance post-estimation, using explicit sandwich algebra.
use greeners_core::linalg::LinalgInverse as _;
use greeners_core::{CovarianceType, InferenceType};
use greeners_ols::ols::OLS;
use greeners_ols::wls::WLS;
use ndarray::{array, Array1, Array2};

fn fixture() -> (Array1<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
    let y = Array1::from_shape_fn(20, |i| {
        let z = i as f64;
        1.0 + 0.75 * z
            + if i % 2 == 0 {
                0.1 + 0.04 * z * z
            } else {
                -0.1 - 0.04 * z * z
            }
    });
    (y, x)
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-8 * (1.0 + b.abs()), "{a} != {b}");
}
fn hc1(y: &Array1<f64>, x: &Array2<f64>, beta: &Array1<f64>) -> Array2<f64> {
    let bread = x.t().dot(x).inv().unwrap();
    let resid = y - &x.dot(beta);
    let mut meat = Array2::zeros((x.ncols(), x.ncols()));
    for i in 0..x.nrows() {
        for a in 0..x.ncols() {
            for b in 0..x.ncols() {
                meat[[a, b]] += x[[i, a]] * x[[i, b]] * resid[i] * resid[i];
            }
        }
    }
    bread.dot(&meat).dot(&bread) * (x.nrows() as f64 / (x.nrows() - x.ncols()) as f64)
}
#[test]
fn hc1_scalar_and_global_use_the_coefficient_table() {
    let (y, x) = fixture();
    let fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    let (t, p) = fit.t_test(&array![0., 1.], 0., &x).unwrap();
    close(t, fit.t_values[1]);
    close(p, fit.p_values[1]);
    assert!(p > 0.05); // The incorrect classical reconstruction reverses this inference.
    close(fit.f_statistic, t * t);
    close(fit.prob_f, p);
}
#[test]
fn correlated_contrast_and_prediction_use_off_diagonal_covariance() {
    let (y, x) = fixture();
    let fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    let cov = hc1(&y, &x, &fit.params);
    let r = array![1., 2.];
    let expected_se = r.dot(&cov.dot(&r)).sqrt();
    let (t, _) = fit.t_test(&r, 0., &x).unwrap();
    close(t, r.dot(&fit.params) / expected_se);
    let pred = fit.get_prediction(&array![[1., 2.]], &x, 0.05).unwrap();
    close(pred.se[0], expected_se);
    let (_, se, _, p) = fit.nlcom(|b| b[0] + 2. * b[1], &x).unwrap();
    close(se, expected_se);
    assert!(p.is_finite());
}
#[test]
fn normal_joint_probability_matches_scalar_normal_probability() {
    let (y, x) = fixture();
    let fit = OLS::fit(&y, &x, CovarianceType::HC1)
        .unwrap()
        .with_inference(InferenceType::Normal)
        .unwrap();
    let (stat, p) = fit.f_test(&[1], &x).unwrap();
    close(stat, fit.t_values[1].powi(2));
    close(p, fit.p_values[1]);
    close(fit.prob_f, p);
    assert!(fit.to_string().contains("Scaled Wald"));
}
#[test]
fn weighted_predictions_preserve_weighted_fit_covariance() {
    let (y, x) = fixture();
    let w = Array1::from_shape_fn(20, |i| 1. + i as f64 / 10.);
    let fit = WLS::fit(&y, &x, &w, CovarianceType::HC1).unwrap();
    let mut wx = x.clone();
    for i in 0..20 {
        wx.row_mut(i).mapv_inplace(|a| a * w[i].sqrt());
    }
    let wy = &y * &w.mapv(f64::sqrt);
    let cov = hc1(&wy, &wx, &fit.params);
    let r = array![1., 2.];
    let pred = fit.get_prediction(&array![[1., 2.]], &x, 0.05).unwrap();
    close(pred.se[0], r.dot(&cov.dot(&r)).sqrt());
}
#[test]
fn invalid_restrictions_are_errors() {
    let (y, x) = fixture();
    let fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    assert!(fit
        .wald_test(&Array2::zeros((0, 2)), &array![], &x)
        .is_err());
    assert!(fit.f_test(&[1, 1], &x).is_err());
}

#[test]
fn scalar_joint_agreement_for_all_covariance_modes_and_reference_distributions() {
    let (y, x) = fixture();
    let covariances = vec![
        CovarianceType::NonRobust,
        CovarianceType::HC1,
        CovarianceType::HC2,
        CovarianceType::HC3,
        CovarianceType::HC4,
        CovarianceType::NeweyWest(2),
        CovarianceType::Clustered((0..20).map(|i| i / 4).collect()),
    ];
    for covariance in covariances {
        let fitted = OLS::fit(&y, &x, covariance).unwrap();
        for distribution in [InferenceType::StudentT, InferenceType::Normal] {
            let fit = fitted.clone().with_inference(distribution.clone()).unwrap();
            for j in 0..2 {
                let (stat, p) = fit.f_test(&[j], &x).unwrap();
                close(stat, fit.t_values[j].powi(2));
                close(p, fit.p_values[j]);
            }
        }
    }
}

#[test]
fn joint_test_uses_correlated_covariance_with_correct_distribution() {
    use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};
    let (y, x) = fixture();
    let fitted = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    let cov = hc1(&y, &x, &fitted.params);
    let q_stat = fitted.params.dot(&cov.inv().unwrap().dot(&fitted.params));
    for distribution in [InferenceType::StudentT, InferenceType::Normal] {
        let fit = fitted.clone().with_inference(distribution.clone()).unwrap();
        let (stat, p) = fit.wald_test(&Array2::eye(2), &array![0., 0.], &x).unwrap();
        close(stat, q_stat / 2.);
        let expected = match distribution {
            InferenceType::StudentT => FisherSnedecor::new(2., 18.).unwrap().sf(q_stat / 2.),
            InferenceType::Normal => ChiSquared::new(2.).unwrap().sf(q_stat),
        };
        close(p, expected);
        let (_, _, t, p_nl) = fit.nlcom(|b| b[0] + 2. * b[1], &x).unwrap();
        let (t_ref, p_ref) = fit.t_test(&array![1., 2.], 0., &x).unwrap();
        close(t, t_ref);
        close(p_nl, p_ref);
    }
}

#[test]
fn actual_intercept_position_and_intercept_only_are_respected() {
    let (y, x) = fixture();
    let reordered = Array2::from_shape_fn((20, 2), |(i, j)| x[[i, 1 - j]]);
    let fit = OLS::fit(&y, &reordered, CovarianceType::HC1).unwrap();
    assert_eq!(fit.intercept_index, Some(1));
    close(fit.f_statistic, fit.t_values[0].powi(2));
    let w = Array1::from_shape_fn(20, |i| 1. + i as f64 / 10.);
    let weighted = WLS::fit(&y, &reordered, &w, CovarianceType::HC1).unwrap();
    assert_eq!(weighted.intercept_index, Some(1));
    close(weighted.f_statistic, weighted.t_values[0].powi(2));
    let no_intercept = x.slice(ndarray::s![.., 1..]).to_owned();
    let fit = OLS::fit(&y, &no_intercept, CovarianceType::HC1).unwrap();
    assert_eq!(fit.df_model, 1);
    close(fit.f_statistic, fit.t_values[0].powi(2));
    let fit = OLS::fit(&y, &Array2::ones((20, 1)), CovarianceType::HC1).unwrap();
    assert_eq!(fit.df_model, 0);
    assert!(fit.f_statistic.is_nan() && fit.prob_f.is_nan());
    assert!(fit.to_string().contains("Unavailable"));
}

#[test]
fn singular_covariance_valid_scalar_contrast_and_zero_variance_are_distinct() {
    let (y, x) = fixture();
    let mut fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    fit.covariance = array![[0., 0.], [0., 4.]];
    let (stat, p) = fit.t_test(&array![0., 1.], 0., &x).unwrap();
    close(stat, fit.params[1] / 2.);
    assert!(p.is_finite());
    let (stat, p) = fit.t_test(&array![1., 0.], fit.params[0], &x).unwrap();
    close(stat, 0.);
    close(p, 1.);
    let (stat, p) = fit.t_test(&array![1., 0.], fit.params[0] - 1., &x).unwrap();
    assert!(stat.is_infinite() && stat > 0.);
    close(p, 0.);
    assert!(fit.f_test(&[0, 1], &x).is_err());
    assert!(fit.nlcom(|_| 0., &x).is_err()); // A black-box flat derivative is unresolved.
    fit.covariance = array![[1e-40, 0.], [0., 1e-40]];
    let (stat, p) = fit
        .t_test(&array![1., 0.], fit.params[0] - 1e-10, &x)
        .unwrap();
    assert!(stat.is_finite() && stat > 1e5);
    assert!(p.is_finite());
}

#[test]
fn invalid_covariance_dimensions_targets_and_significance_fail_without_panicking() {
    let (y, x) = fixture();
    let mut fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    assert!(fit.t_test(&array![1.], 0., &x).is_err());
    assert!(fit.t_test(&array![1., 0.], f64::NAN, &x).is_err());
    assert!(fit
        .t_test(&array![1., 0.], 0., &Array2::ones((20, 3)))
        .is_err());
    assert!(fit.f_test(&[3], &x).is_err());
    assert!(fit
        .wald_test(&array![[1., 0.]], &array![0., 0.], &x)
        .is_err());
    assert!(fit.get_prediction(&array![[1., 2.]], &x, 0.).is_err());
    assert!(fit.conf_int(f64::NAN).is_err());
    fit.covariance = Array2::eye(3);
    assert!(fit.t_test(&array![1., 0.], 0., &x).is_err());
    assert!(fit.clone().with_inference(InferenceType::Normal).is_err());
    fit.covariance = Array2::eye(2);
    fit.params[0] = f64::NAN;
    assert!(fit.t_test(&array![1., 0.], 0., &x).is_err());
    fit.params[0] = 0.0;
    fit.covariance = array![[1., 2.], [2., 1.]];
    assert!(fit.t_test(&array![1., 0.], 0., &x).is_err());
    fit.covariance = array![[1., 0.], [0., -1e-40]];
    assert!(fit.t_test(&array![1., 0.], 0., &x).is_err());
}

#[test]
fn covariance_and_contrast_units_do_not_change_inference() {
    let (y, x) = fixture();
    let fitted = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    let (expected_stat, expected_p) = fitted
        .wald_test(&Array2::eye(2), &array![0., 0.], &x)
        .unwrap();
    for factors in [array![1e-10, 1e10], array![1e10, 1e-10]] {
        let mut fit = fitted.clone();
        fit.params = &fit.params * &factors;
        fit.covariance = Array2::from_shape_fn((2, 2), |(i, j)| {
            fitted.covariance[[i, j]] * factors[i] * factors[j]
        });
        let (stat, p) = fit.wald_test(&Array2::eye(2), &array![0., 0.], &x).unwrap();
        close(stat, expected_stat);
        close(p, expected_p);
    }
}

#[test]
fn restriction_row_rescaling_preserves_joint_inference() {
    let (y, x) = fixture();
    let fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    let (expected_stat, expected_p) = fit.wald_test(&Array2::eye(2), &array![0., 0.], &x).unwrap();
    for scale in [1e-200, 1e200] {
        let (stat, p) = fit
            .wald_test(&array![[scale, 0.], [0., 1. / scale]], &array![0., 0.], &x)
            .unwrap();
        close(stat, expected_stat);
        close(p, expected_p);
    }
}

#[test]
fn weighted_duplicate_constant_retains_intercept_after_collinearity_removal() {
    let (y, x) = fixture();
    let design = Array2::from_shape_fn((20, 3), |(i, j)| match j {
        0 => 1.,
        1 => 2.,
        _ => x[[i, 1]],
    });
    let weights = Array1::from_shape_fn(20, |i| 1. + i as f64 / 10.);
    let fit = WLS::fit_with_names(
        &y,
        &design,
        &weights,
        CovarianceType::HC1,
        Some(vec!["const".into(), "constant2".into(), "x".into()]),
    )
    .unwrap();
    assert_eq!(fit.omitted_vars.len(), 1);
    assert!(fit.intercept_index.is_some());
    assert_eq!(fit.df_model, 1);
    let slope = if fit.intercept_index == Some(0) { 1 } else { 0 };
    close(fit.f_statistic, fit.t_values[slope].powi(2));
}

#[test]
fn tiny_nonzero_residuals_cannot_be_reported_as_exact_zero_uncertainty() {
    let (y, x) = fixture();
    for covariance in [CovarianceType::NonRobust, CovarianceType::HC1] {
        assert!(OLS::fit(&y.mapv(|v| v * 1e-200), &x, covariance).is_err());
    }
}

#[test]
fn scalar_and_prediction_uncertainty_preserve_extreme_row_units() {
    let (y, x) = fixture();
    let fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    let (expected_t, expected_p) = fit.t_test(&array![0., 1.], 0., &x).unwrap();
    for scale in [1e-200, 1e200] {
        let (t, p) = fit.t_test(&array![0., scale], 0., &x).unwrap();
        close(t, expected_t);
        close(p, expected_p);
        let pred = fit.get_prediction(&array![[0., scale]], &x, 0.05).unwrap();
        close(pred.se[0] / scale, fit.std_errors[1]);
        assert!(pred.ci_lower[0].is_finite() && pred.ci_upper[0].is_finite());
    }
}
#[test]
fn robust_nonzero_influence_cannot_underflow_to_exact_zero_variance() {
    let (y, _) = fixture();
    let x = Array2::from_elem((20, 1), 1e100);
    assert!(OLS::fit(&y.mapv(|v| v * 1e-100), &x, CovarianceType::HC1).is_err());
}
#[test]
fn extreme_alpha_has_finite_symmetric_intervals() {
    let (y, x) = fixture();
    let fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    for distribution in [InferenceType::StudentT, InferenceType::Normal] {
        let fit = fit.clone().with_inference(distribution).unwrap();
        for (lo, hi) in fit.conf_int(1e-20).unwrap() {
            assert!(lo.is_finite() && hi.is_finite());
        }
        let pred = fit.get_prediction(&array![[1., 2.]], &x, 1e-20).unwrap();
        assert!(pred.ci_lower[0].is_finite() && pred.ci_upper[0].is_finite());
    }
}
#[test]
fn pooled_panel_hc3_covariance_has_intrinsic_sandwich_symmetry() {
    use greeners_core::{DataFrame, Formula};
    let data = DataFrame::from_csv(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/panel_covariance.csv"
    ))
    .unwrap();
    let formula = Formula::parse("lucro ~ alavancagem + tamanho + C(setor)").unwrap();
    for covariance in [CovarianceType::HC3, CovarianceType::NeweyWest(2)] {
        let fit = OLS::from_formula(&formula, &data, covariance).unwrap();
        assert!(fit.covariance.iter().all(|v| v.is_finite()));
        for j in 0..fit.params.len() {
            close(fit.covariance[[j, j]], fit.std_errors[j].powi(2));
        }
    }
}
#[test]
fn quantile_var_weighted_initialisation_remains_identified() {
    use greeners_ols::quantile::QuantileReg;
    let first =
        array![10., 12., 15., 18., 20., 22., 25., 28., 30., 32., 35., 38., 40., 42., 45., 48.];
    let second = Array1::from_shape_fn(16, |i| 5. + i as f64);
    let design = Array2::from_shape_fn((15, 3), |(i, j)| match j {
        0 => 1.,
        1 => first[i],
        _ => second[i],
    });
    for y in [
        first.slice(ndarray::s![1..]).to_owned(),
        second.slice(ndarray::s![1..]).to_owned(),
    ] {
        let result = QuantileReg::fit(&y, &design, 0.5, 0).unwrap();
        assert!(result.params.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn zero_variance_coordinates_do_not_hide_estimable_contrasts() {
    let (y, x) = fixture();
    let mut fit = OLS::fit(&y, &x, CovarianceType::HC1).unwrap();
    fit.params = array![0., 1.];
    fit.covariance = array![[0., 0.], [0., 1.]];
    let (t, p) = fit.t_test(&array![1e200, 1.], 0., &x).unwrap();
    close(t, 1.);
    assert!(p > 0.05);
    let (f, p_joint) = fit
        .wald_test(&array![[1e200, 1.]], &array![0.], &x)
        .unwrap();
    close(f, t * t);
    close(p, p_joint);
    let prediction = fit.get_prediction(&array![[1e200, 1.]], &x, 0.05).unwrap();
    close(prediction.mean[0], 1.);
    close(prediction.se[0], 1.);
}

#[test]
fn six_row_cluster_covariance_preserves_intrinsic_symmetry() {
    let y = array![1., 3., 4., 6., 7., 9.];
    let x = Array2::from_shape_fn((6, 2), |(i, j)| if j == 0 { 1. } else { i as f64 + 1. });
    let fit = OLS::fit(&y, &x, CovarianceType::Clustered(vec![0, 0, 0, 1, 1, 1])).unwrap();
    let bread = x.t().dot(&x).inv().unwrap();
    let residuals = &y - &x.dot(&fit.params);
    let mut meat = Array2::zeros((2, 2));
    for range in [0..3, 3..6] {
        let score = x
            .slice(ndarray::s![range.clone(), ..])
            .t()
            .dot(&residuals.slice(ndarray::s![range]));
        for i in 0..2 {
            for j in 0..2 {
                meat[[i, j]] += score[i] * score[j];
            }
        }
    }
    let expected = bread.dot(&meat).dot(&bread) * 2. * 5. / 4.;
    for i in 0..2 {
        for j in 0..2 {
            close(fit.covariance[[i, j]], expected[[i, j]]);
        }
    }
}

#[test]
fn two_way_cluster_uses_label_pairs_without_integer_overflow() {
    let (y, x) = fixture();
    let labels: Vec<_> = (0..20).map(|i| usize::MAX - i / 4).collect();
    let first = OLS::fit(&y, &x, CovarianceType::Clustered(labels.clone())).unwrap();
    let second = OLS::fit(
        &y,
        &x,
        CovarianceType::ClusteredTwoWay(labels.clone(), labels),
    )
    .unwrap();
    for i in 0..2 {
        for j in 0..2 {
            close(first.covariance[[i, j]], second.covariance[[i, j]]);
        }
    }
}

#[test]
fn bartlett_hac_score_windows_match_the_lag_sandwich() {
    let (y, x) = fixture();
    for lags in [0, 2, 19, 25] {
        let fit = OLS::fit(&y, &x, CovarianceType::NeweyWest(lags)).unwrap();
        let bread = x.t().dot(&x).inv().unwrap();
        let residuals = &y - &x.dot(&fit.params);
        let scores = Array2::from_shape_fn((20, 2), |(i, j)| x[[i, j]] * residuals[i]);
        let mut meat = scores.t().dot(&scores);
        for lag in 1..=lags.min(19) {
            let cross = scores
                .slice(ndarray::s![lag.., ..])
                .t()
                .dot(&scores.slice(ndarray::s![..20 - lag, ..]));
            meat += &((&cross + &cross.t()) * (1. - lag as f64 / (lags + 1) as f64));
        }
        let expected = bread.dot(&meat).dot(&bread) * (20. / 18.);
        for i in 0..2 {
            for j in 0..2 {
                close(fit.covariance[[i, j]], expected[[i, j]]);
            }
        }
    }
}
