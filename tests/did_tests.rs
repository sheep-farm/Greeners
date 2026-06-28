use greeners::{DataFrame, Formula, DiffInDiff, CovarianceType};

#[test]
fn test_did_dummy_types() {
    let y_vec = vec![10.0, 12.0, 11.0, 13.0, 10.5, 12.5, 12.0, 16.0];
    let treated_int = vec![0i64, 0, 0, 0, 1, 1, 1, 1];
    let post_bool = vec![false, false, true, true, false, false, true, true];

    let df = DataFrame::builder()
        .add_column("y", y_vec)
        .add_int("treated", treated_int)
        .add_bool("post", post_bool)
        .build()
        .unwrap();

    let formula = Formula::parse("y ~ treated").unwrap();
    let res = DiffInDiff::from_formula(&formula, &df, "treated", "post", CovarianceType::HC1).unwrap();

    // ATT:
    // control_pre = (10 + 12) / 2 = 11
    // control_post = (11 + 13) / 2 = 12 -> diff = 1
    // treated_pre = (10.5 + 12.5) / 2 = 11.5
    // treated_post = (12 + 16) / 2 = 14 -> diff = 2.5
    // ATT = 2.5 - 1.0 = 1.5
    assert!((res.att - 1.5).abs() < 1e-9);
    assert!(res.std_error > 0.0);
    assert!((0.0..=1.0).contains(&res.p_value));
}
