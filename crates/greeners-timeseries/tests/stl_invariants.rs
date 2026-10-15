use greeners_timeseries::Decomposition;
use ndarray::Array1;

fn fixture(text: &str) -> Vec<[f64; 6]> {
    csv::Reader::from_reader(text.as_bytes())
        .records()
        .enumerate()
        .map(|(i, row)| {
            let row = row.unwrap();
            assert_eq!(row.len(), 6);
            let values = std::array::from_fn(|j| row[j].parse::<f64>().unwrap());
            assert_eq!(values[0], i as f64);
            values
        })
        .collect()
}

fn golden(text: &str, n: usize, period: usize, sw: usize, tw: usize) {
    let rows = fixture(text);
    assert_eq!(rows.len(), n);
    let y = Array1::from_iter(rows.iter().map(|r| r[1]));
    let fit = Decomposition::stl(&y, period, sw, tw).unwrap();
    assert_eq!(fit.model, "STL");
    assert_eq!(fit.observed, y);
    for (name, actual, col) in [
        ("trend", &fit.trend, 2),
        ("seasonal", &fit.seasonal, 3),
        ("remainder", &fit.residual, 4),
    ] {
        assert_eq!(actual.len(), n);
        for i in 0..n {
            assert!(actual[i].is_finite());
            assert!(
                (actual[i] - rows[i][col]).abs() <= 1e-8,
                "{name}[{i}]: actual={} expected={} absolute_error={}",
                actual[i],
                rows[i][col],
                (actual[i] - rows[i][col]).abs()
            );
        }
    }
    for i in 0..n {
        assert!((y[i] - (fit.trend[i] + fit.seasonal[i] + fit.residual[i])).abs() <= 1e-12);
    }
}

#[test]
fn stl_airpassengers_full_vectors() {
    golden(
        include_str!("fixtures/stl/airpassengers.csv"),
        144,
        12,
        7,
        23,
    );
}

#[test]
fn stl_synthetic_full_vectors() {
    golden(
        include_str!("fixtures/stl/outliers_even.csv"),
        96,
        12,
        7,
        23,
    );
    golden(include_str!("fixtures/stl/unequal_odd.csv"), 53, 7, 9, 15);
    golden(include_str!("fixtures/stl/shortest.csv"), 8, 4, 7, 9);
    golden(include_str!("fixtures/stl/wide_spans.csv"), 29, 5, 31, 41);
}

#[test]
fn stl_analytic_zero_constant_linear() {
    for y in [
        Array1::zeros(48),
        Array1::from_elem(48, 2.0),
        Array1::from_iter((0..48).map(|i| 0.25 + 0.03125 * i as f64)),
    ] {
        let fit = Decomposition::stl(&y, 12, 7, 23).unwrap();
        assert_eq!(fit.observed, y);
        for values in [&fit.trend, &fit.seasonal, &fit.residual] {
            assert_eq!(values.len(), 48);
            assert!(values.iter().all(|v| v.is_finite()));
        }
        for i in 0..48 {
            assert!((fit.trend[i] - y[i]).abs() <= 1e-12);
            assert!(fit.seasonal[i].abs() <= 1e-12);
            assert!(fit.residual[i].abs() <= 1e-12);
            assert!((y[i] - (fit.trend[i] + fit.seasonal[i] + fit.residual[i])).abs() <= 1e-12);
        }
    }
}

#[test]
fn stl_window_normalisation() {
    let rows = fixture(include_str!("fixtures/stl/airpassengers.csv"));
    let y = Array1::from_iter(rows.iter().map(|r| r[1]));
    for ((sw, tw), (expected_sw, expected_tw)) in [
        ((6, 23), (7, 23)),
        ((8, 23), (9, 23)),
        ((7, 22), (7, 23)),
        ((7, 0), (7, 23)),
    ] {
        let a = Decomposition::stl(&y, 12, sw, tw).unwrap();
        let b = Decomposition::stl(&y, 12, expected_sw, expected_tw).unwrap();
        assert_eq!(a.trend, b.trend);
        assert_eq!(a.seasonal, b.seasonal);
        assert_eq!(a.residual, b.residual);
    }
}

#[test]
fn stl_invalid_inputs_and_arithmetic_overflow() {
    let y = Array1::from_elem(48, 2.0);
    for p in [0, 1, usize::MAX] {
        assert!(Decomposition::stl(&y, p, 7, 23).is_err());
    }
    for n in [0, 1, 23] {
        assert!(Decomposition::stl(&Array1::zeros(n), 12, 7, 23).is_err());
    }
    for (p, sw, tw) in [
        (12, usize::MAX, 23),
        (12, 7, usize::MAX),
        (12, 7, 11),
        (13, 7, 12),
    ] {
        assert!(Decomposition::stl(&y, p, sw, tw).is_err());
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut input = y.clone();
        input[17] = bad;
        assert!(Decomposition::stl(&input, 12, 7, 23).is_err());
    }
    let large = Array1::from_iter((0..48).map(|i| {
        if i % 2 == 0 {
            f64::MAX / 2.0
        } else {
            -f64::MAX / 2.0
        }
    }));
    if let Ok(fit) = Decomposition::stl(&large, 12, 7, 23) {
        for values in [&fit.trend, &fit.seasonal, &fit.residual] {
            assert_eq!(values.len(), 48);
            assert!(values.iter().all(|v| v.is_finite()));
        }
        for i in 0..48 {
            // Scale before summing so the check itself cannot overflow.
            let reconstructed =
                fit.trend[i] / large[i] + fit.seasonal[i] / large[i] + fit.residual[i] / large[i];
            assert!((reconstructed - 1.0).abs() <= 1e-12);
        }
    }
}
