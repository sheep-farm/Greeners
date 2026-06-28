use greeners::{DataFrame, Formula, FamaMacBeth};

#[test]
fn test_fama_macbeth_basic() {
    let mut y_vec = Vec::new();
    let mut x1_vec = Vec::new();
    let mut time_vec = Vec::new();

    let mut state = 42u64;
    let mut rand_double = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state as f64) / (u64::MAX as f64)
    };

    for t in 1..=10 {
        for _asset in 1..=5 {
            let x1 = rand_double() * 4.0 - 2.0;
            let err = rand_double() * 0.1 - 0.05;
            let y = 1.0 + 2.0 * x1 + err;

            y_vec.push(y);
            x1_vec.push(x1);
            time_vec.push(t as i64);
        }
    }

    // Build DataFrame using DataFrameBuilder to support mix of Float and Int columns
    let df = DataFrame::builder()
        .add_column("y", y_vec)
        .add_column("x1", x1_vec)
        .add_int("time", time_vec)
        .build()
        .unwrap();

    let formula = Formula::parse("y ~ x1").unwrap();

    let res = FamaMacBeth::fit(&formula, &df, "time", 0).unwrap();

    assert_eq!(res.n_periods, 10);
    assert_eq!(res.n_obs_total, 50);
    assert_eq!(res.params.len(), 2);

    assert!((res.params[0] - 1.0).abs() < 0.1);
    assert!((res.params[1] - 2.0).abs() < 0.1);

    assert!(res.std_errors[0] > 0.0);
    assert!(res.std_errors[1] > 0.0);
    assert!((0.0..=1.0).contains(&res.p_values[0]));

    let res_nw = FamaMacBeth::fit(&formula, &df, "time", 2).unwrap();
    assert_eq!(res_nw.n_periods, 10);
    assert_eq!(res_nw.n_obs_total, 50);
    assert!(res_nw.std_errors[0] > 0.0);
}
