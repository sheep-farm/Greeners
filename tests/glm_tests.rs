use greeners::{CovarianceType, Family, Formula, InferenceType, Link, GLM, OLS};
use ndarray::{Array1, Array2};

/// Helper: build design matrix with intercept column.
fn with_intercept(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let mut xm = Array2::<f64>::ones((n, k + 1));
    for j in 0..k {
        xm.column_mut(j + 1).assign(&x.column(j));
    }
    xm
}

// ============================================================
// 1. Gaussian GLM ≈ OLS
// ============================================================
#[test]
fn test_gaussian_glm_matches_ols() {
    let y = Array1::from(vec![1.0, 2.1, 3.0, 3.9, 5.1, 6.0, 7.2, 7.9, 9.1, 10.0]);
    let x_raw = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let x = with_intercept(&x_raw);

    let ols_res = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let glm_res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();

    // Coefficients should be very close
    for i in 0..ols_res.params.len() {
        assert!(
            (ols_res.params[i] - glm_res.params[i]).abs() < 1e-4,
            "Param {} differs: OLS={} GLM={}",
            i,
            ols_res.params[i],
            glm_res.params[i]
        );
    }

    assert!(glm_res.converged);
}

// ============================================================
// 2. Binomial GLM ≈ Logit
// ============================================================
#[test]
fn test_binomial_glm_matches_logit() {
    let y = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let x_raw = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let x = with_intercept(&x_raw);

    let logit_res = greeners::Logit::fit(&y, &x).unwrap();
    let glm_res = GLM::fit(&y, &x, Family::Binomial, CovarianceType::NonRobust).unwrap();

    // Coefficients should be close (IRLS vs Newton-Raphson may differ slightly)
    for i in 0..logit_res.params.len() {
        assert!(
            (logit_res.params[i] - glm_res.params[i]).abs() < 0.1,
            "Param {} differs: Logit={} GLM={}",
            i,
            logit_res.params[i],
            glm_res.params[i]
        );
    }

    assert!(glm_res.converged);
    assert_eq!(glm_res.link, Link::Logit);
}

// ============================================================
// 3. Poisson with count data
// ============================================================
#[test]
fn test_poisson_glm() {
    // Simple count data
    let y = Array1::from(vec![2.0, 3.0, 5.0, 7.0, 11.0, 14.0, 18.0, 22.0, 28.0, 35.0]);
    let x_raw = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let x = with_intercept(&x_raw);

    let res = GLM::fit(&y, &x, Family::Poisson, CovarianceType::NonRobust).unwrap();

    assert!(res.converged);
    assert!(res.n_iter < 50);
    // Log link means exp(b0 + b1*x) ≈ y; b1 should be positive
    assert!(res.params[1] > 0.0, "Slope should be positive");
    assert!(
        res.dispersion == 1.0,
        "Poisson dispersion should be fixed at 1"
    );
}

// ============================================================
// 4. Gamma with positive continuous data
// ============================================================
#[test]
fn test_gamma_glm() {
    let y = Array1::from(vec![0.5, 1.2, 2.3, 3.1, 4.5, 5.0, 6.8, 7.2, 8.9, 10.1]);
    let x_raw = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let x = with_intercept(&x_raw);

    let res = GLM::fit(&y, &x, Family::Gamma, CovarianceType::NonRobust).unwrap();

    assert!(res.converged);
    assert!(res.dispersion > 0.0, "Gamma dispersion should be estimated");
    assert!(res.deviance >= 0.0);
}

// ============================================================
// 5. Non-canonical link: Poisson + Identity
// ============================================================
#[test]
fn test_non_canonical_link() {
    let y = Array1::from(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    let x_raw = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let x = with_intercept(&x_raw);

    let res = GLM::fit_with_link(
        &y,
        &x,
        Family::Poisson,
        Link::Identity,
        CovarianceType::NonRobust,
    )
    .unwrap();

    assert!(res.converged);
    // With identity link on linear count data, slope ≈ 2
    assert!((res.params[1] - 2.0).abs() < 0.5);
}

// ============================================================
// 6. Convergence and iterations
// ============================================================
#[test]
fn test_convergence_info() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = with_intercept(&Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap());

    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();

    assert!(res.converged);
    assert!(res.n_iter > 0);
    assert!(res.n_iter <= 100);
}

// ============================================================
// 7. Predict / fitted_values / residuals
// ============================================================
#[test]
fn test_predict_and_residuals() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let x_raw = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let x = with_intercept(&x_raw);

    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();

    let fitted = res.fitted_values();
    assert_eq!(fitted.len(), 10);

    let dev_resid = res.residuals();
    assert_eq!(dev_resid.len(), 10);

    let pearson_resid = res.pearson_residuals();
    assert_eq!(pearson_resid.len(), 10);

    let working_resid = res.working_residuals();
    assert_eq!(working_resid.len(), 10);

    // predict_mean for new data
    let x_new = Array2::from_shape_vec((2, 2), vec![1.0, 11.0, 1.0, 12.0]).unwrap();
    let pred = res.predict_mean(&x_new);
    assert_eq!(pred.len(), 2);
    assert!(pred[0] > 10.0);
}

// ============================================================
// 8. NaN/Inf validation
// ============================================================
#[test]
fn test_nan_input_rejected() {
    let y = Array1::from(vec![1.0, f64::NAN, 3.0]);
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust);
    assert!(res.is_err());
}

#[test]
fn test_inf_input_rejected() {
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    let x = Array2::from_shape_vec((3, 2), vec![1.0, f64::INFINITY, 1.0, 2.0, 1.0, 3.0]).unwrap();

    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust);
    assert!(res.is_err());
}

// ============================================================
// 9. Collinearity detection
// ============================================================
#[test]
fn test_collinearity_detection() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    // x2 = 2*x1 (perfectly collinear)
    let x = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 3.0, 6.0, 1.0, 4.0, 8.0, 1.0, 5.0, 10.0,
        ],
    )
    .unwrap();

    let names = vec!["const".into(), "x1".into(), "x2".into()];
    let res = GLM::fit_with_names(
        &y,
        &x,
        Family::Gaussian,
        CovarianceType::NonRobust,
        Some(names),
    )
    .unwrap();

    assert!(!res.omitted_vars.is_empty());
}

// ============================================================
// 10. from_formula with categoricals
// ============================================================
#[test]
fn test_from_formula() {
    use greeners::DataFrame;

    let csv = "y,x\n1.0,1.0\n2.0,2.0\n3.0,3.0\n4.0,4.0\n5.0,5.0\n6.0,6.0\n7.0,7.0\n8.0,8.0\n9.0,9.0\n10.0,10.0";
    // Write CSV to temp file
    let tmp = "/tmp/glm_test_data.csv";
    std::fs::write(tmp, csv).unwrap();
    let df = DataFrame::from_csv(tmp).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();

    let res =
        GLM::from_formula(&formula, &df, Family::Gaussian, CovarianceType::NonRobust).unwrap();

    assert!(res.converged);
    assert!(res.variable_names.is_some());
    let names = res.variable_names.as_ref().unwrap();
    assert!(names.contains(&"const".to_string()));
    assert!(names.contains(&"x".to_string()));
}

// ============================================================
// 11. Display output
// ============================================================
#[test]
fn test_display() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let x = with_intercept(
        &Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap(),
    );

    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();
    let output = format!("{}", res);
    assert!(output.contains("Generalized Linear Model"));
    assert!(output.contains("Gaussian"));
}

// ============================================================
// 12. with_inference
// ============================================================
#[test]
fn test_with_inference() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let x = with_intercept(
        &Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap(),
    );

    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();
    let res_t = res.with_inference(InferenceType::StudentT).unwrap();

    assert_eq!(res_t.inference_type, InferenceType::StudentT);
}

// ============================================================
// 13. HC1 robust standard errors
// ============================================================
#[test]
fn test_hc1_standard_errors() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let x = with_intercept(
        &Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap(),
    );

    let res_nr = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();
    let res_hc = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::HC1).unwrap();

    // Coefficients should be identical
    for i in 0..res_nr.params.len() {
        assert!((res_nr.params[i] - res_hc.params[i]).abs() < 1e-6);
    }
    // Standard errors differ
}

// ============================================================
// 14. Negative Binomial
// ============================================================
#[test]
fn test_negative_binomial() {
    let y = Array1::from(vec![
        2.0, 5.0, 8.0, 12.0, 18.0, 22.0, 30.0, 35.0, 45.0, 55.0,
    ]);
    let x = with_intercept(
        &Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap(),
    );

    let res = GLM::fit(
        &y,
        &x,
        Family::NegativeBinomial(1.0),
        CovarianceType::NonRobust,
    )
    .unwrap();
    assert!(res.converged);
    assert!(res.params[1] > 0.0);
}

// ============================================================
// 15. InverseGaussian
// ============================================================
#[test]
fn test_inverse_gaussian() {
    // y ≈ 1/sqrt(0.1 + 0.1*x): decreasing series compatible with InverseGaussian canonical link
    // (canonical link is η = 1/μ², which is increasing when y is decreasing)
    let y = Array1::from(vec![
        2.24, 1.83, 1.58, 1.41, 1.29, 1.20, 1.12, 1.05, 1.00, 0.95,
    ]);
    let x = with_intercept(
        &Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap(),
    );

    let res = GLM::fit(&y, &x, Family::InverseGaussian, CovarianceType::NonRobust).unwrap();
    assert!(res.converged);
}

// ============================================================
// 16. Model stats
// ============================================================
#[test]
fn test_model_stats() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let x = with_intercept(
        &Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap(),
    );

    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();
    let (aic, bic, ll, pr2) = res.model_stats();
    assert!(aic.is_finite());
    assert!(bic.is_finite());
    assert!(ll.is_finite());
    assert!(pr2.is_finite());
}
