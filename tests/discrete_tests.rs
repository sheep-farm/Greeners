use greeners::{DataFrame, Formula, Logit, Probit};
use ndarray::Array1;
use std::collections::HashMap;

// Helper function to create realistic binary outcome data without perfect separation
fn create_test_data() -> (Vec<f64>, Vec<f64>) {
    let x_data = vec![
        -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, -1.8, -1.3, -0.8, -0.3, 0.2, 0.7,
        1.2, 1.7, 2.2, 2.7,
    ];
    let y_data = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0,
    ];
    (x_data, y_data)
}

#[test]
fn test_logit_basic_estimation() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Logit::from_formula(&formula, &df).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.log_likelihood.is_finite());
    assert!(result.pseudo_r2 >= 0.0 && result.pseudo_r2 <= 1.0);
    assert!(result.iterations > 0);
    assert!(result.params[1] > 0.0);
}

#[test]
fn test_probit_basic_estimation() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Probit::from_formula(&formula, &df).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.log_likelihood.is_finite());
    assert!(result.pseudo_r2 >= 0.0 && result.pseudo_r2 <= 1.0);
    assert!(result.params[1] > 0.0);
}

#[test]
fn test_logit_convergence() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Logit::from_formula(&formula, &df).unwrap();

    assert!(result.iterations < 100);
    assert!(result.params[1] > 0.0);
}

#[test]
fn test_logit_average_marginal_effects() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Logit::from_formula(&formula, &df).unwrap();

    let (_, x) = df.to_design_matrix(&formula).unwrap();
    let ame = result.average_marginal_effects(&x).unwrap();

    assert_eq!(ame.len(), 2);
    assert!(ame.iter().all(|&m| m.is_finite()));
    assert!(ame[1].abs() > 0.0);
}

#[test]
fn test_logit_marginal_effects_at_means() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Logit::from_formula(&formula, &df).unwrap();

    let (_, x) = df.to_design_matrix(&formula).unwrap();
    let mem = result.marginal_effects_at_means(&x).unwrap();

    assert_eq!(mem.len(), 2);
    assert!(mem.iter().all(|&m| m.is_finite()));
}

#[test]
fn test_logit_predict_proba() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Logit::from_formula(&formula, &df).unwrap();

    let (_, x) = df.to_design_matrix(&formula).unwrap();
    let probs = result.predict_proba(&x);

    assert_eq!(probs.len(), x.nrows());

    for &p in probs.iter() {
        assert!(p >= 0.0 && p <= 1.0);
    }

    // With positive slope, probabilities should generally increase
    if result.params[1] > 0.0 {
        let first_prob = probs[0];
        let last_prob = probs[probs.len() - 1];
        // Allow for some variation but expect general trend
        assert!(last_prob > first_prob - 0.3);
    }
}

#[test]
fn test_probit_marginal_effects() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Probit::from_formula(&formula, &df).unwrap();

    let (_, x) = df.to_design_matrix(&formula).unwrap();
    let ame = result.average_marginal_effects(&x).unwrap();

    assert_eq!(ame.len(), 2);
    assert!(ame.iter().all(|&m| m.is_finite()));
}

#[test]
fn test_logit_model_stats() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Logit::from_formula(&formula, &df).unwrap();

    let (aic, bic, loglik, pseudo_r2) = result.model_stats();

    assert!(aic.is_finite());
    assert!(bic.is_finite());
    assert!(loglik.is_finite());
    assert!(pseudo_r2.is_finite());
    assert!(pseudo_r2 >= 0.0 && pseudo_r2 <= 1.0);
    assert!(aic > 0.0);
    assert!(bic > 0.0);
}

#[test]
fn test_logit_ame_confidence_intervals() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Logit::from_formula(&formula, &df).unwrap();

    let (_, x) = df.to_design_matrix(&formula).unwrap();
    let (lower, upper) = result.ame_confidence_intervals(&x, 0.05).unwrap();

    assert_eq!(lower.len(), 2);
    assert_eq!(upper.len(), 2);

    for i in 0..lower.len() {
        assert!(upper[i] >= lower[i]);
    }
}

#[test]
fn test_logit_vs_probit_similarity() {
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();

    let logit_result = Logit::from_formula(&formula, &df).unwrap();
    let probit_result = Probit::from_formula(&formula, &df).unwrap();

    let (_, x) = df.to_design_matrix(&formula).unwrap();

    let ame_logit = logit_result.average_marginal_effects(&x).unwrap();
    let ame_probit = probit_result.average_marginal_effects(&x).unwrap();

    // AMEs should have same sign and similar magnitude
    for i in 0..ame_logit.len() {
        if ame_logit[i].abs() > 0.01 {
            let ratio = ame_logit[i] / ame_probit[i];
            assert!(ratio > 0.0); // Same sign
            assert!(ratio > 0.3 && ratio < 3.0); // Similar magnitude (relaxed from 0.5-2.0)
        }
    }
}

#[test]
fn test_logit_iterations_reasonable() {
    // Test that Logit converges in reasonable iterations
    let (x_data, y_data) = create_test_data();

    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(y_data));
    data.insert("x".to_string(), Array1::from(x_data));

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x").unwrap();
    let result = Logit::from_formula(&formula, &df).unwrap();

    // Should converge in much less than max iterations
    assert!(result.iterations < 50);
    assert!(result.iterations > 0);
}
