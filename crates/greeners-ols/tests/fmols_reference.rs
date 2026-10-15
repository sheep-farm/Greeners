#![cfg(feature = "experimental")]
use greeners_ols::FMOLS;
use ndarray::{Array1, Array2};
use serde_json::Value;
use std::path::Path;

fn fixture(name: &str) -> (Array1<f64>, Array2<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fmols")
        .join(format!("{name}.csv"));
    let mut reader = csv::Reader::from_path(path).unwrap();
    let k = reader.headers().unwrap().len() - 1;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for row in reader.records() {
        let row = row.unwrap();
        xs.extend((0..k).map(|j| row[j].parse::<f64>().unwrap()));
        ys.push(row[k].parse::<f64>().unwrap());
    }
    let n = ys.len();
    (
        Array1::from_vec(ys),
        Array2::from_shape_vec((n, k), xs).unwrap(),
    )
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-10 + 1e-9 * expected.abs(),
        "{actual} != {expected}"
    );
}

#[test]
fn matches_native_arch_scalar_and_multivariate_fixtures() {
    let references: Value =
        serde_json::from_str(include_str!("fixtures/fmols/reference.json")).unwrap();
    for case in references["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (y, x) = fixture(name);
        let r = FMOLS::fit(&y, &x, None).unwrap();
        close(r.alpha, case["params"][0].as_f64().unwrap());
        close(r.alpha_se, case["se"][0].as_f64().unwrap());
        for j in 0..x.ncols() {
            close(r.beta[j], case["params"][j + 1].as_f64().unwrap());
            close(r.beta_se[j], case["se"][j + 1].as_f64().unwrap());
            let expected = case["normal_probabilities"][j + 1].as_f64().unwrap();
            if expected > 0.0 {
                assert!(
                    (r.beta_p[j] / expected - 1.0).abs() < 1e-8,
                    "{name}: probability {} != {expected}",
                    r.beta_p[j]
                );
            } else {
                assert_eq!(r.beta_p[j], 0.0);
            }
        }
        for j in 0..=x.ncols() {
            close(
                case["native_R"]["params"][j].as_f64().unwrap(),
                case["predicted_R_params"][j].as_f64().unwrap(),
            );
            close(
                case["native_R"]["se"][j].as_f64().unwrap(),
                case["se"][j].as_f64().unwrap(),
            );
        }
        assert_eq!(r.n_obs, case["n"].as_u64().unwrap() as usize);
        assert_eq!(r.bandwidth, case["bandwidth"].as_u64().unwrap() as usize);
        for ((i, j), v) in r.omega.indexed_iter() {
            close(*v, case["omega"][i][j].as_f64().unwrap());
        }
    }
}

#[test]
fn predictor_units_shifts_and_permutations_restore_estimates_and_uncertainty() {
    let (y, x) = fixture("multivariate");
    let r = FMOLS::fit(&y, &x, None).unwrap();
    let factors = [1e-6, -3e4];
    let shifts = [25.0, -10.0];
    let changed = Array2::from_shape_fn(x.dim(), |(i, j)| {
        (x[[i, 1 - j]] + shifts[1 - j]) * factors[1 - j]
    });
    let s = FMOLS::fit(&y, &changed, None).unwrap();
    close(
        s.alpha,
        r.alpha - r.beta[0] * shifts[0] - r.beta[1] * shifts[1],
    );
    for j in 0..2 {
        close(s.beta[j] * factors[1 - j], r.beta[1 - j]);
        close(s.beta_se[j] * factors[1 - j].abs(), r.beta_se[1 - j]);
    }
    // Shift to the unchanged sample centre: its intercept uncertainty must match.
    let back = Array2::from_shape_fn(x.dim(), |(i, j)| {
        changed[[i, 1 - j]] / factors[j] - shifts[j]
    });
    let restored = FMOLS::fit(&y, &back, None).unwrap();
    close(restored.alpha_se, r.alpha_se);
    let mean = x.mean_axis(ndarray::Axis(0)).unwrap();
    let centered = &x - &mean;
    let center_result = FMOLS::fit(&y, &centered, None).unwrap();
    assert!((center_result.alpha_se - r.alpha_se).abs() > 1e-4);
    close(center_result.alpha, r.alpha + r.beta.dot(&mean));
}

#[test]
fn outcome_shift_and_units_transform_all_reported_covariance_units() {
    let (y, x) = fixture("endogeneity");
    let r = FMOLS::fit(&y, &x, None).unwrap();
    let s = FMOLS::fit(&y.mapv(|v| -7.0 * v + 123.0), &x, None).unwrap();
    close(s.alpha, -7.0 * r.alpha + 123.0);
    close(s.alpha_se, 7.0 * r.alpha_se);
    close(s.beta[0], -7.0 * r.beta[0]);
    close(s.beta_se[0], 7.0 * r.beta_se[0]);
    close(s.omega[[0, 0]], 49.0 * r.omega[[0, 0]]);
    close(s.omega[[0, 1]], -7.0 * r.omega[[0, 1]]);
    close(s.omega[[1, 1]], r.omega[[1, 1]]);
}

#[test]
fn unit_intercepts_and_names_follow_supplied_columns() {
    let (y, x) = fixture("multivariate");
    let r = FMOLS::fit(&y, &x, None).unwrap();
    for intercept in 0..=2 {
        let z = Array2::from_shape_fn((x.nrows(), 3), |(i, j)| {
            if j == intercept {
                1.0
            } else {
                x[[i, if j < intercept { j } else { j - 1 }]]
            }
        });
        let mut names = vec!["a".into(), "b".into()];
        names.insert(intercept, "constant".into());
        let s = FMOLS::fit(&y, &z, Some(names)).unwrap();
        assert_eq!(s.variable_names, vec!["a", "b"]);
        close(s.alpha, r.alpha);
        close(s.alpha_se, r.alpha_se);
        assert!(FMOLS::fit(&y, &z, Some(vec!["a".into(), "b".into()])).is_ok());
    }
    for names in [
        vec!["a".into()],
        vec!["a".into(), "a".into()],
        vec!["".into(), "b".into()],
    ] {
        assert!(FMOLS::fit(&y, &x, Some(names)).is_err());
    }
}

#[test]
fn invalid_and_unresolved_inputs_fail_instead_of_fabricating_precision() {
    let (y, x) = fixture("endogeneity");
    assert!(FMOLS::fit(&Array1::zeros(0), &Array2::zeros((0, 0)), None).is_err());
    assert!(FMOLS::fit(&y, &Array2::zeros((y.len(), 0)), None).is_err());
    assert!(FMOLS::fit(&y, &Array2::ones((y.len(), 2)), None).is_err());
    assert!(FMOLS::fit(&y, &Array2::from_elem((y.len(), 1), 2.0), None).is_err());
    let duplicate = Array2::from_shape_fn((y.len(), 2), |(i, _)| x[[i, 0]]);
    assert!(FMOLS::fit(&y, &duplicate, None).is_err());
    let almost_duplicate = Array2::from_shape_fn((y.len(), 2), |(i, j)| {
        x[[i, 0]]
            + if j == 1 {
                1e-13 * (i as f64).sin()
            } else {
                0.0
            }
    });
    assert!(FMOLS::fit(&y, &almost_duplicate, None).is_err());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut xx = x.clone();
        xx[[5, 0]] = bad;
        assert!(FMOLS::fit(&y, &xx, None).is_err());
        let mut yy = y.clone();
        yy[5] = bad;
        assert!(FMOLS::fit(&yy, &x, None).is_err());
    }
    assert!(FMOLS::fit(&Array1::zeros(y.len()), &x, None).is_err());
    let exact = x.column(0).mapv(|v| 3.0 * v + 2.0);
    assert!(FMOLS::fit(&exact, &x, None).is_err());
    let huge = Array2::from_shape_fn(
        x.dim(),
        |(i, _)| if i % 2 == 0 { f64::MAX } else { -f64::MAX },
    );
    assert!(FMOLS::fit(&y, &huge, None).is_err());
    let tiny = x.mapv(|v| v * 1e-200);
    assert!(FMOLS::fit(&y, &tiny, None).is_err());
    let large = x.mapv(|v| v * 1e200);
    assert!(FMOLS::fit(&y, &large, None).is_err());
}

#[test]
fn displayed_intercept_uses_normal_probability_and_small_tails_survive() {
    let (y, x) = fixture("iid");
    let r = FMOLS::fit(&y, &x, None).unwrap();
    let table = r.to_string();
    assert!(table.contains("0.9446"));
    assert!(table.contains("P>|z|"));
    let (y, x) = fixture("endogeneity");
    let r = FMOLS::fit(&y, &x, None).unwrap();
    assert!(r.beta_p[0] > 0.0 && r.beta_p[0] < 1e-300);
}
