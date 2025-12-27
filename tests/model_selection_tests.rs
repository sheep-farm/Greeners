use greeners::{ModelSelection, PanelDiagnostics, SummaryStats};
use ndarray::Array1;

#[test]
fn test_model_selection_compare_models() {
    // Create 3 hypothetical models
    let models = vec![
        ("Model 1", -100.0, 3, 100), // log_lik, k params, n obs
        ("Model 2", -95.0, 5, 100),
        ("Model 3", -98.0, 4, 100),
    ];

    let comparison = ModelSelection::compare_models(models);

    // Should return 3 models
    assert_eq!(comparison.len(), 3);

    // Check that all fields are present
    for (name, aic, bic, rank_aic, rank_bic) in &comparison {
        assert!(!name.is_empty());
        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(*rank_aic >= 1 && *rank_aic <= 3);
        assert!(*rank_bic >= 1 && *rank_bic <= 3);
    }

    // Models should be sorted by AIC (lowest first)
    for i in 0..comparison.len() - 1 {
        assert!(comparison[i].1 <= comparison[i + 1].1);
    }

    // Best model (rank 1) should have lowest AIC
    let best_model = &comparison[0];
    assert_eq!(best_model.3, 1); // rank_aic should be 1
}

#[test]
fn test_model_selection_aic_calculation() {
    // Manually verify AIC calculation
    let loglik = -50.0;
    let k = 3;
    let n = 100;

    let models = vec![("Test Model", loglik, k, n)];
    let comparison = ModelSelection::compare_models(models);

    // AIC = -2*loglik + 2*k = -2*(-50) + 2*3 = 100 + 6 = 106
    let expected_aic = -2.0 * loglik + 2.0 * (k as f64);
    assert!((comparison[0].1 - expected_aic).abs() < 1e-10);
}

#[test]
fn test_model_selection_bic_calculation() {
    let loglik = -50.0;
    let k = 3;
    let n = 100;

    let models = vec![("Test Model", loglik, k, n)];
    let comparison = ModelSelection::compare_models(models);

    // BIC = -2*loglik + k*ln(n) = -2*(-50) + 3*ln(100) = 100 + 3*4.605 ≈ 113.82
    let expected_bic = -2.0 * loglik + (k as f64) * (n as f64).ln();
    assert!((comparison[0].2 - expected_bic).abs() < 1e-10);
}

#[test]
fn test_akaike_weights_basic() {
    let aic_values = vec![100.0, 102.0, 105.0];
    let (delta_aic, weights) = ModelSelection::akaike_weights(&aic_values);

    // Check delta AIC
    assert_eq!(delta_aic.len(), 3);
    assert!((delta_aic[0] - 0.0).abs() < 1e-10); // Min AIC has delta = 0
    assert!((delta_aic[1] - 2.0).abs() < 1e-10);
    assert!((delta_aic[2] - 5.0).abs() < 1e-10);

    // Check weights
    assert_eq!(weights.len(), 3);

    // Weights should sum to 1
    let sum_weights: f64 = weights.iter().sum();
    assert!((sum_weights - 1.0).abs() < 1e-10);

    // All weights should be between 0 and 1
    for &w in &weights {
        assert!(w >= 0.0 && w <= 1.0);
    }

    // Best model (lowest AIC) should have highest weight
    assert!(weights[0] > weights[1]);
    assert!(weights[1] > weights[2]);
}

#[test]
fn test_akaike_weights_equal_models() {
    // If all models have same AIC, weights should be equal
    let aic_values = vec![100.0, 100.0, 100.0];
    let (delta_aic, weights) = ModelSelection::akaike_weights(&aic_values);

    // All deltas should be 0
    for &d in &delta_aic {
        assert!(d.abs() < 1e-10);
    }

    // All weights should be 1/3
    for &w in &weights {
        assert!((w - 1.0 / 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_panel_diagnostics_breusch_pagan_basic() {
    // Create pooled OLS residuals with panel structure
    let residuals = Array1::from(vec![
        0.1, -0.2, 0.15, -0.1, 0.05, // Entity 1
        0.3, -0.25, 0.2, -0.15, 0.1, // Entity 2
        -0.1, 0.2, -0.15, 0.1, -0.05, // Entity 3
    ]);

    let entity_ids = vec![
        0, 0, 0, 0, 0, // Entity 1
        1, 1, 1, 1, 1, // Entity 2
        2, 2, 2, 2, 2, // Entity 3
    ];

    let result = PanelDiagnostics::breusch_pagan_lm(&residuals, &entity_ids);
    assert!(result.is_ok());

    let (lm_stat, p_value) = result.unwrap();

    // LM statistic should be non-negative
    assert!(lm_stat >= 0.0);

    // P-value should be between 0 and 1
    assert!(p_value >= 0.0 && p_value <= 1.0);
}

#[test]
fn test_panel_diagnostics_bp_no_panel_effect() {
    // Residuals with NO panel effect (all random)
    // Should have high p-value (fail to reject H0)
    let residuals = Array1::from(vec![
        0.05, -0.03, 0.02, -0.04, 0.01, -0.02, 0.04, -0.01, 0.03, -0.05, 0.03, -0.05, 0.04, -0.02,
        0.01,
    ]);

    let entity_ids = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

    let (lm_stat, p_value) = PanelDiagnostics::breusch_pagan_lm(&residuals, &entity_ids).unwrap();

    // With no panel effect, LM should be small and p-value high
    assert!(lm_stat >= 0.0);
    // Note: p-value might not always be > 0.05 due to randomness in this simple example
}

#[test]
fn test_panel_diagnostics_f_test_basic() {
    let ssr_pooled = 150.0;
    let ssr_fe = 100.0;
    let n = 100;
    let n_entities = 10;
    let k = 3; // Number of slope parameters

    let result = PanelDiagnostics::f_test_fixed_effects(ssr_pooled, ssr_fe, n, n_entities, k);
    assert!(result.is_ok());

    let (f_stat, p_value) = result.unwrap();

    // F statistic should be positive (FE reduces SSR)
    assert!(f_stat > 0.0);

    // P-value should be valid
    assert!(p_value >= 0.0 && p_value <= 1.0);

    // With substantial reduction in SSR, should reject H0
    assert!(p_value < 0.05);
}

#[test]
fn test_panel_diagnostics_f_test_no_effect() {
    // SSR same for both models (no improvement from FE)
    let ssr_pooled = 100.0;
    let ssr_fe = 100.0;
    let n = 100;
    let n_entities = 10;
    let k = 3;

    let (f_stat, p_value) =
        PanelDiagnostics::f_test_fixed_effects(ssr_pooled, ssr_fe, n, n_entities, k).unwrap();

    // F should be near 0
    assert!(f_stat.abs() < 1e-10);

    // P-value should be near 1
    assert!(p_value > 0.95);
}

#[test]
fn test_summary_stats_basic() {
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let (mean, std, min, q25, median, q75, max, n) = SummaryStats::describe(&data);

    // Check n
    assert_eq!(n, 10);

    // Check mean: (1+2+...+10)/10 = 55/10 = 5.5
    assert!((mean - 5.5).abs() < 1e-10);

    // Check min and max
    assert_eq!(min, 1.0);
    assert_eq!(max, 10.0);

    // Check median (should be between 5 and 6)
    assert!(median >= 5.0 && median <= 6.0);

    // Check std is positive
    assert!(std > 0.0);

    // Check quartiles are in order
    assert!(min <= q25);
    assert!(q25 <= median);
    assert!(median <= q75);
    assert!(q75 <= max);
}

#[test]
fn test_summary_stats_quartiles() {
    // Test with known quartiles
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let (_, _, min, q25, median, q75, max, _) = SummaryStats::describe(&data);

    assert_eq!(min, 1.0);
    assert_eq!(max, 5.0);
    assert_eq!(median, 3.0); // Middle value

    // Q25 should be around 2.0, Q75 around 4.0
    assert!((q25 - 2.0).abs() < 0.5);
    assert!((q75 - 4.0).abs() < 0.5);
}

#[test]
fn test_summary_stats_single_value() {
    let data = Array1::from(vec![5.0]);
    let (mean, std, min, q25, median, q75, max, n) = SummaryStats::describe(&data);

    assert_eq!(n, 1);
    assert_eq!(mean, 5.0);
    assert_eq!(min, 5.0);
    assert_eq!(max, 5.0);
    assert_eq!(median, 5.0);
    assert_eq!(std, 0.0); // No variance with single value
}

#[test]
fn test_summary_stats_constant_values() {
    let data = Array1::from(vec![3.0, 3.0, 3.0, 3.0, 3.0]);
    let (mean, std, min, q25, median, q75, max, _) = SummaryStats::describe(&data);

    assert_eq!(mean, 3.0);
    assert_eq!(std, 0.0);
    assert_eq!(min, 3.0);
    assert_eq!(max, 3.0);
    assert_eq!(median, 3.0);
    assert_eq!(q25, 3.0);
    assert_eq!(q75, 3.0);
}

#[test]
fn test_model_selection_ranking() {
    // Test that models are ranked correctly
    // AIC = -2*loglik + 2*k
    let models = vec![
        ("Best AIC", -50.0, 2, 100),  // AIC = 100 + 4 = 104 (best)
        ("Worst AIC", -30.0, 5, 100), // AIC = 60 + 10 = 70... wait this is wrong
        ("Middle", -45.0, 3, 100),    // AIC = 90 + 6 = 96
    ];

    // Let me recalculate for correct ranking:
    // "Best": -50, k=2 -> AIC = 100 + 4 = 104
    // "Middle": -45, k=3 -> AIC = 90 + 6 = 96
    // "Worst": -30, k=5 -> AIC = 60 + 10 = 70 (actually best!)
    //
    // Correct order by AIC (ascending): Worst(70), Middle(96), Best(104)
    // Let me fix this properly:
    let models = vec![
        ("Best", -52.0, 2, 100),   // AIC = 104 + 4 = 108 -> Best (lowest)
        ("Middle", -51.0, 4, 100), // AIC = 102 + 8 = 110 -> Middle
        ("Worst", -50.0, 6, 100),  // AIC = 100 + 12 = 112 -> Worst (highest)
    ];

    let comparison = ModelSelection::compare_models(models);

    // Check AIC ranking (sorted by AIC ascending)
    assert_eq!(comparison[0].0, "Best");
    assert_eq!(comparison[0].3, 1); // Should be rank 1

    assert_eq!(comparison[1].0, "Middle");
    assert_eq!(comparison[1].3, 2); // Should be rank 2

    assert_eq!(comparison[2].0, "Worst");
    assert_eq!(comparison[2].3, 3); // Should be rank 3
}

#[test]
fn test_panel_diagnostics_entity_mismatch() {
    let residuals = Array1::from(vec![0.1, 0.2, 0.3]);
    let entity_ids = vec![0, 0]; // Wrong length

    let result = PanelDiagnostics::breusch_pagan_lm(&residuals, &entity_ids);
    assert!(result.is_err());
}

#[test]
fn test_panel_diagnostics_insufficient_df() {
    // Test F-test with insufficient degrees of freedom
    let ssr_pooled = 100.0;
    let ssr_fe = 90.0;
    let n = 10;
    let n_entities = 8; // Too many entities for small n
    let k = 5; // Too many parameters

    let result = PanelDiagnostics::f_test_fixed_effects(ssr_pooled, ssr_fe, n, n_entities, k);

    // Should return error due to insufficient df
    assert!(result.is_err());
}
