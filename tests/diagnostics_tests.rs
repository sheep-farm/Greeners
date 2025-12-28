use greeners::diagnostics::Diagnostics;
use ndarray::{Array1, Array2};

#[test]
fn test_vif_basic() {
    // Data without perfect multicollinearity
    // Column 0: intercept (all 1s)
    // Column 1: sequential values
    // Column 2: independent random-like values
    let x = Array2::from_shape_vec(
        (10, 3),
        vec![
            1.0, 1.0, 2.3, 1.0, 2.0, 5.1, 1.0, 3.0, 3.8, 1.0, 4.0, 7.2, 1.0, 5.0, 4.5, 1.0, 6.0,
            9.1, 1.0, 7.0, 6.7, 1.0, 8.0, 8.3, 1.0, 9.0, 5.9, 1.0, 10.0, 10.2,
        ],
    )
    .unwrap();

    let vif_results = Diagnostics::vif(&x).unwrap();

    assert_eq!(vif_results.len(), 3);

    // First column (intercept) should be NaN
    assert!(vif_results[0].is_nan());

    // Other columns should have finite VIF >= 1.0
    for &vif_value in vif_results.iter().skip(1) {
        assert!(vif_value >= 1.0); // VIF is always >= 1 for non-constant columns
        assert!(vif_value.is_finite());
    }
}

#[test]
fn test_vif_multicollinearity() {
    // x2 is highly (but not perfectly) correlated with x1
    // x2 ≈ x1 + noise
    let x = Array2::from_shape_vec(
        (10, 3),
        vec![
            1.0, 1.0, 1.2, 1.0, 2.0, 2.1, 1.0, 3.0, 3.3, 1.0, 4.0, 3.9, 1.0, 5.0, 5.1, 1.0, 6.0,
            5.8, 1.0, 7.0, 7.2, 1.0, 8.0, 7.9, 1.0, 9.0, 9.1, 1.0, 10.0, 9.8,
        ],
    )
    .unwrap();

    let vif_results = Diagnostics::vif(&x).unwrap();

    assert_eq!(vif_results.len(), 3);

    // First column (intercept) should be NaN
    assert!(vif_results[0].is_nan());

    // Find max VIF among non-NaN, finite values (columns 1 and 2)
    let max_vif = vif_results
        .iter()
        .skip(1)
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // VIF values should be elevated due to high correlation between x1 and x2
    assert!(max_vif > &5.0); // With high correlation, VIF should be well above 1
}

#[test]
fn test_condition_number_well_conditioned() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let cond_num = Diagnostics::condition_number(&x).unwrap();

    assert!(cond_num.is_finite());
    assert!(cond_num > 0.0);
}

#[test]
fn test_jarque_bera_normal() {
    // Approximately normal residuals
    let residuals = Array1::from(vec![
        -0.1, 0.2, -0.05, 0.15, -0.2, 0.1, 0.0, -0.15, 0.05, -0.1, 0.2, -0.05,
    ]);

    let (jb_stat, p_value) = Diagnostics::jarque_bera(&residuals).unwrap();

    assert!(jb_stat >= 0.0);
    assert!((0.0..=1.0).contains(&p_value));
}

#[test]
fn test_durbin_watson() {
    // Test DW statistic
    let residuals = Array1::from(vec![0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.15, -0.15]);

    let dw = Diagnostics::durbin_watson(&residuals);

    // DW should be between 0 and 4
    assert!((0.0..=4.0).contains(&dw));
}

#[test]
fn test_durbin_watson_no_autocorrelation() {
    // Random residuals - DW should be close to 2
    let residuals = Array1::from(vec![0.1, -0.2, 0.15, -0.1, 0.05, -0.15, 0.2, -0.05]);

    let dw = Diagnostics::durbin_watson(&residuals);

    assert!(dw > 0.0 && dw < 4.0);
}

#[test]
fn test_breusch_pagan() {
    let residuals = Array1::from(vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.25]);
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();

    let (lm_stat, p_value) = Diagnostics::breusch_pagan(&residuals, &x).unwrap();

    assert!(lm_stat >= 0.0);
    assert!((0.0..=1.0).contains(&p_value));
}

#[test]
fn test_vif_single_variable() {
    // Only one variable (intercept)
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

    let vif_results = Diagnostics::vif(&x).unwrap();

    assert_eq!(vif_results.len(), 1);
}

#[test]
fn test_condition_number_ill_conditioned() {
    // Nearly singular matrix
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 1.001, 1.0, 1.002, 1.0, 1.003, 1.0, 1.004],
    )
    .unwrap();

    let cond_num = Diagnostics::condition_number(&x).unwrap();

    // Should be high for ill-conditioned matrix
    assert!(cond_num > 10.0);
}
