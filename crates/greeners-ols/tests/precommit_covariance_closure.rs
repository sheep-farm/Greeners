//! Counterexamples from independent review, with analytical delta-method references.
use greeners_core::{CovarianceType, GreenersError, InferenceType};
use greeners_ols::ols::{OlsResult, OLS};
use ndarray::{array, Array1, Array2};
use statrs::distribution::{ContinuousCDF, StudentsT};

fn fixture() -> (OlsResult, Array2<f64>) {
    let x = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { 1. } else { i as f64 });
    let y = Array1::from_shape_fn(20, |i| 1. + i as f64 + if i % 2 == 0 { 0.1 } else { -0.1 });
    (OLS::fit(&y, &x, CovarianceType::HC1).unwrap(), x)
}
fn relative(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual / expected - 1.).abs() <= tolerance,
        "{actual:e} != {expected:e}"
    );
}
#[test]
fn exact_rank_one_nearly_null_contrasts_preserve_positive_uncertainty() {
    let (mut fit, x) = fixture();
    fit.params = array![0., 1e-9];
    fit.covariance = array![[1., 1.], [1., 1.]];
    for scale in [1e-100, 1., 1e100] {
        let r: Array1<f64> = array![scale, (-1. + 1e-9) * scale];
        let expected = (r[0] + r[1]).abs();
        let prediction = fit
            .get_prediction(&r.clone().insert_axis(ndarray::Axis(0)), &x, 0.05)
            .unwrap();
        relative(prediction.se[0], expected, 2e-6);
        let q = r.dot(&fit.params) - expected;
        let (t, p) = fit.t_test(&r, q, &x).unwrap();
        relative(t, 1., 2e-6);
        relative(p, 2. * StudentsT::new(0., 1., 18.).unwrap().sf(1.), 2e-6);
        let (f, p_joint) = fit
            .wald_test(&r.clone().insert_axis(ndarray::Axis(0)), &array![q], &x)
            .unwrap();
        relative(f, 1., 4e-6);
        relative(p_joint, p, 2e-6);
    }
    let (_, se, _, _) = fit.nlcom(|b| b[0] + (-1. + 1e-9) * b[1], &x).unwrap();
    relative(se, (1.0_f64 - (1.0 - 1e-9)).abs(), 2e-5);
}
fn accurate_or_precision_error<T>(result: Result<T, GreenersError>, check: impl FnOnce(T)) {
    match result {
        Ok(value) => check(value),
        Err(GreenersError::InvalidOperation(message)) => assert!(
            message.contains("precision") || message.contains("unresolved"),
            "{message}"
        ),
        Err(error) => panic!("Unexpected rejection: {error}"),
    }
}
#[test]
fn actual_cluster_nearly_null_uncertainty_is_accurate_or_explicitly_unresolved() {
    let x = array![[1., -1.], [1., -1.], [1., 1.], [1., 1.]];
    let y = array![0., 0., 1., -1.];
    let fit = OLS::fit(&y, &x, CovarianceType::Clustered(vec![0, 1, 0, 1])).unwrap();
    let r: Array1<f64> = array![1., -1. + 1e-9];
    let expected = (r[0] + r[1]).abs() * 0.375_f64.sqrt();
    let q = r.dot(&fit.params) - expected;
    accurate_or_precision_error(
        fit.get_prediction(&r.clone().insert_axis(ndarray::Axis(0)), &x, 0.05),
        |pred| relative(pred.se[0], expected, 2e-6),
    );
    accurate_or_precision_error(fit.t_test(&r, q, &x), |(t, p)| {
        relative(t, 1., 2e-6);
        relative(p, 0.42264973081037427, 2e-6);
    });
    accurate_or_precision_error(
        fit.wald_test(&r.clone().insert_axis(ndarray::Axis(0)), &array![q], &x),
        |(f, p)| {
            relative(f, 1., 4e-6);
            relative(p, 0.42264973081037427, 2e-6);
        },
    );
    accurate_or_precision_error(
        fit.nlcom(|b| b[0] + (-1. + 1e-9) * b[1], &x),
        |(_, se, _, _)| relative(se, expected, 2e-5),
    );
}
#[test]
fn certified_exact_null_direction_retains_the_scalar_boundary_contract() {
    let (mut fit, x) = fixture();
    fit.params = array![0., 0.];
    fit.covariance = array![[1., 1.], [1., 1.]];
    let r = array![1., -1.];
    let pred = fit
        .get_prediction(&r.clone().insert_axis(ndarray::Axis(0)), &x, 0.05)
        .unwrap();
    assert_eq!(pred.se[0], 0.);
    assert_eq!(fit.t_test(&r, 0., &x).unwrap(), (0., 1.));
    let (t, p) = fit.t_test(&r, -1., &x).unwrap();
    assert!(t.is_infinite() && t > 0.);
    assert_eq!(p, 0.);
}
#[test]
fn cubic_delta_inference_does_not_depend_on_response_units() {
    let x = Array2::ones((20, 1));
    let y = Array1::from_shape_fn(20, |i| {
        1e-5 + if i % 2 == 0 {
            1.5e-6 * 19_f64.sqrt()
        } else {
            -1.5e-6 * 19_f64.sqrt()
        }
    });
    for scale in [1e-5, 1., 1e5, 1e10] {
        let fit = OLS::fit(&y.mapv(|v| v * scale), &x, CovarianceType::HC1).unwrap();
        let (_, se, t, p) = fit.nlcom(|b| b[0].powi(3), &x).unwrap();
        let expected = 3. * fit.params[0].powi(2) * fit.std_errors[0];
        relative(se, expected, 1e-7);
        relative(t, fit.params[0] / (3. * fit.std_errors[0]), 1e-7);
        relative(p, 0.038608561669242855, 1e-7);
    }
}
#[test]
fn log_gradient_stays_in_the_local_positive_domain() {
    let x = Array2::ones((20, 1));
    let y = Array1::from_shape_fn(20, |i| 1e-6 + if i % 2 == 0 { 1e-9 } else { -1e-9 });
    for scale in [1e-8, 1., 1e8] {
        let fit = OLS::fit(&y.mapv(|v| v * scale), &x, CovarianceType::HC1)
            .unwrap()
            .with_inference(InferenceType::Normal)
            .unwrap();
        let (estimate, se, t, p) = fit.nlcom(|b| b[0].ln(), &x).unwrap();
        relative(se, fit.std_errors[0] / fit.params[0], 1e-7);
        relative(t, estimate / se, 1e-7);
        assert!(p.is_finite());
    }
}
#[test]
fn unresolved_flat_or_nonsmooth_derivatives_are_errors() {
    let (mut fit, x) = fixture();
    fit.params = array![0., 0.];
    fit.covariance = Array2::eye(2);
    assert!(fit.nlcom(|b| 1e16 + b[0] + b[1], &x).is_err());
    assert!(fit.nlcom(|b| b[0].abs() + b[1], &x).is_err());
    assert!(fit.nlcom(|b| b[0].sqrt() + b[1], &x).is_err());
    assert!(fit.nlcom(|_| 3., &x).is_err());
    // A stationary smooth function has resolvable variation despite zero gradient.
    let (estimate, se, t, p) = fit.nlcom(|b| b[0] * b[0] + b[1] * b[1], &x).unwrap();
    assert_eq!((estimate, se, t, p), (0., 0., 0., 1.));
    fit.covariance = Array2::zeros((2, 2));
    assert_eq!(fit.nlcom(|_| 3., &x).unwrap().1, 0.);
}

#[test]
fn opaque_unused_uncertain_coordinates_have_an_explicit_availability_limit() {
    let (mut fit, x) = fixture();
    fit.params = array![0., 1.];
    fit.covariance = Array2::eye(2);
    assert!(fit.nlcom(|b| b[1], &x).is_err());
    assert!(fit.nlcom(|b| b[1].ln(), &x).is_err());
    // No numerical derivative is needed for an exactly deterministic coordinate.
    fit.covariance = array![[0., 0.], [0., 1.]];
    let (_, se, _, _) = fit.nlcom(|b| b[1], &x).unwrap();
    relative(se, 1., 1e-8);
}

#[test]
fn factor_projection_does_not_drop_small_cancellation_remainders() {
    let (mut fit, _) = fixture();
    fit.params = Array1::zeros(6);
    fit.covariance = Array2::ones((6, 6));
    let x = Array2::ones((20, 6));
    for tail in [0.0, 3e-200] {
        let r = array![1e100, 1., 1e-200, -1e100, -1., tail];
        accurate_or_precision_error(
            fit.get_prediction(&r.insert_axis(ndarray::Axis(0)), &x, 0.05),
            |pred| relative(pred.se[0], 1e-200 + tail, 1e-6),
        );
    }
}

#[test]
fn low_exponent_fma_residuals_cannot_certify_false_rank_one_covariance() {
    let (mut fit, x) = fixture();
    fit.params = array![0., 0.];
    // Exact Fraction arithmetic on the stored covariance/contrast gives this
    // positive variance in units of d. It includes covariance rounding effects.
    let expected_se_units = 6.66059300196825e-9;
    let v = 1.3333;
    for multiplier in [1., 2., 8., 128.] {
        let d = f64::MIN_POSITIVE * multiplier;
        fit.covariance = array![[d, d * v], [d * v, d * v * v]];
        let r = array![v, -1. + 1e-9];
        let expected = expected_se_units * d.sqrt();
        accurate_or_precision_error(
            fit.get_prediction(&r.clone().insert_axis(ndarray::Axis(0)), &x, 0.05),
            |pred| relative(pred.se[0], expected, 1e-6),
        );
        accurate_or_precision_error(fit.t_test(&r, -expected, &x), |(t, _)| {
            relative(t, 1., 1e-6)
        });
        accurate_or_precision_error(
            fit.wald_test(
                &r.clone().insert_axis(ndarray::Axis(0)),
                &array![-expected],
                &x,
            ),
            |(stat, _)| relative(stat, 1., 2e-6),
        );
        accurate_or_precision_error(
            fit.nlcom(|b| v * b[0] + (-1. + 1e-9) * b[1], &x),
            |(_, se, _, _)| relative(se, expected, 1e-6),
        );
        // Ordinary uncertainty is still resolved without exact-factor certification.
        let ordinary = fit.get_prediction(&array![[1., 0.]], &x, 0.05).unwrap();
        relative(ordinary.se[0], d.sqrt(), 1e-12);
    }
}
