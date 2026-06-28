use greeners::PSM;
use ndarray::{Array1, Array2};

#[test]
fn test_psm_basic() {
    let n = 100;
    let mut y_vec = Vec::with_capacity(n);
    let mut d_vec = Vec::with_capacity(n);
    let mut x_vec = Vec::with_capacity(n * 2);

    let mut state = 42u64;
    let mut rand_double = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state as f64) / (u64::MAX as f64)
    };

    for _ in 0..n {
        let x1 = rand_double() * 2.0 - 1.0;
        let x2 = rand_double() * 2.0 - 1.0;
        x_vec.push(x1);
        x_vec.push(x2);

        let p_treat = 1.0 / (1.0 + (-(x1 + x2)).exp());
        let d = if rand_double() < p_treat { 1.0 } else { 0.0 };
        d_vec.push(d);

        let err = rand_double() * 0.2 - 0.1;
        let y = 2.0 * d + 1.5 * x1 + 0.5 * x2 + err;
        y_vec.push(y);
    }

    let y = Array1::from(y_vec);
    let d = Array1::from(d_vec);
    let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();

    let res = PSM::fit(&y, &d, &x, 1, None, true, 50, None).unwrap();

    assert!(res.att > 0.0);
    assert!(res.se > 0.0);
    assert!((0.0..=1.0).contains(&res.p_value));
    assert_eq!(res.n_treated + res.n_control, n);
    assert_eq!(res.balance.len(), 2);

    for row in &res.balance {
        assert!(row.smd_before.is_finite());
        assert!(row.smd_after.is_finite());
    }
}
