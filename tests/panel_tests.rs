use greeners::{BetweenEstimator, DataFrame, FixedEffects, Formula, RandomEffects};
use ndarray::Array1;
use std::collections::HashMap;

#[test]
fn test_fixed_effects_basic() {
    // Panel data: 3 entities, 2 time periods each
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = ndarray::Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let entity_ids = vec![1, 1, 2, 2, 3, 3];

    let result = FixedEffects::fit(&y, &x, &entity_ids).unwrap();

    assert_eq!(result.params.len(), 1);
    assert_eq!(result.n_entities, 3);
    assert_eq!(result.n_obs, 6);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
}

#[test]
fn test_fixed_effects_from_formula() {
    let mut data = HashMap::new();
    data.insert(
        "y".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    data.insert(
        "x2".to_string(),
        Array1::from(vec![2.0, 3.5, 4.2, 5.1, 6.3, 7.5]), // Not perfectly correlated
    );

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1 + x2 - 1").unwrap(); // No intercept in FE
    let entity_ids = vec![1, 1, 2, 2, 3, 3];

    let result = FixedEffects::from_formula(&formula, &df, &entity_ids).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.variable_names.is_some());
    let names = result.variable_names.unwrap();
    assert_eq!(names, vec!["x1", "x2"]);
}

#[test]
fn test_fixed_effects_demeaning() {
    // Test that FE removes entity-specific effects
    let y = Array1::from(vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0]); // Different entity means
    let x = ndarray::Array2::from_shape_vec(
        (6, 1),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0], // Same pattern across entities
    )
    .unwrap();
    let entity_ids = vec![1, 1, 2, 2, 3, 3];

    let result = FixedEffects::fit(&y, &x, &entity_ids).unwrap();

    // Coefficient should be close to 1.0 (difference within entities)
    assert!((result.params[0] - 1.0).abs() < 0.1);
}

#[test]
fn test_random_effects_basic() {
    // Add some variation to avoid perfect fit
    let y = Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9]);
    let x = ndarray::Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0,
        ],
    )
    .unwrap();
    let entity_ids = Array1::from(vec![1, 1, 2, 2, 3, 3, 4, 4]);

    let result = RandomEffects::fit(&y, &x, &entity_ids).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.r_squared_overall >= 0.0 && result.r_squared_overall <= 1.0);
    assert!(result.theta > 0.0 && result.theta < 1.0);
    assert!(result.sigma_u > 0.0);
    assert!(result.sigma_e >= 0.0);
}

#[test]
fn test_random_effects_from_formula() {
    let mut data = HashMap::new();
    data.insert(
        "y".to_string(),
        Array1::from(vec![1.2, 2.1, 3.3, 3.9, 5.1, 5.8, 7.2, 7.9]),
    );
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1").unwrap();
    let entity_ids = Array1::from(vec![1, 1, 2, 2, 3, 3, 4, 4]);

    let result = RandomEffects::from_formula(&formula, &df, &entity_ids).unwrap();

    assert_eq!(result.params.len(), 2);
}

#[test]
fn test_between_estimator_basic() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = ndarray::Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    )
    .unwrap();
    let entity_ids = Array1::from(vec![1, 1, 2, 2, 3, 3]);

    let result = BetweenEstimator::fit(&y, &x, &entity_ids).unwrap();

    assert_eq!(result.params.len(), 2);
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
}

#[test]
fn test_between_estimator_from_formula() {
    let mut data = HashMap::new();
    data.insert(
        "y".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );

    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ x1").unwrap();
    let entity_ids = Array1::from(vec![1, 1, 2, 2, 3, 3]);

    let result = BetweenEstimator::from_formula(&formula, &df, &entity_ids).unwrap();

    assert_eq!(result.params.len(), 2);
}

#[test]
fn test_panel_degrees_of_freedom() {
    // Test that FE correctly adjusts degrees of freedom
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let x = ndarray::Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let entity_ids = vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

    let result = FixedEffects::fit(&y, &x, &entity_ids).unwrap();

    // n=10, k=1, n_entities=2
    // df_resid = n - k - (n_entities - 1) = 10 - 1 - 1 = 8
    assert_eq!(result.df_resid, 8);
}

#[test]
fn test_fixed_effects_string_ids() {
    // Test with string entity IDs
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let x = ndarray::Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let entity_ids = vec!["firm_a", "firm_a", "firm_b", "firm_b"];

    let result = FixedEffects::fit(&y, &x, &entity_ids).unwrap();

    assert_eq!(result.n_entities, 2);
    assert!(result.params[0].is_finite());
}
