use greeners_timeseries::timeseries::TimeSeries;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

fn assert_all_close(actual: &Array1<f64>, expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= tol,
            "index {index}: expected {expected_value:.15e}, got {actual_value:.15e}"
        );
    }
}

fn cf_reference_series() -> Array1<f64> {
    Array1::from_iter((0..20).map(|t| {
        let t = t as f64;
        100.0
            + 0.8 * t
            + 5.0 * (2.0 * std::f64::consts::PI * t / 8.0).sin()
            + 2.0 * (2.0 * std::f64::consts::PI * t / 20.0).cos()
    }))
}

/// Full-sample reference vector from statsmodels 0.14.6 and R mFilter 0.1-5
/// with `root=TRUE` and `type="asymmetric"`.
#[test]
fn test_cf_filter_matches_asymmetric_references_without_drift() {
    let expected = [
        1.392_263_803_025_39,
        3.514_734_969_499_58,
        4.704849930276836e0,
        3.680713591183654e0,
        4.438002405497414e-1,
        -3.439961650013412e0,
        -5.725_787_969_349,
        -5.012_031_331_064_12,
        -1.802875667849575e0,
        1.726548680477419e0,
        3.167939872122422e0,
        1.519881345273176e0,
        -2.141152904650494e0,
        -5.423013702588612e0,
        -6.234955150526718e0,
        -4.098132997534803e0,
        -2.821563051486331e-1,
        3.199891021674485e0,
        4.914073726572562e0,
        4.713141983840186e0,
    ];

    let cycle = TimeSeries::cf_filter(&cf_reference_series(), 6, 32, false).unwrap();
    assert_all_close(&cycle, &expected, 1e-10);
}

/// Drift removal follows the same endpoint-to-endpoint convention as both references.
#[test]
fn test_cf_filter_matches_asymmetric_references_with_drift() {
    let expected = [
        2.630752163501848e0,
        4.712564332540873e0,
        5.659166242977804e0,
        4.303734589130151e0,
        7.764_491_000_666_14e-1,
        -3.278947485381263e0,
        -5.619623674380039e0,
        -4.905181678421286e0,
        -1.708620601228269e0,
        1.765254114078166e0,
        3.129234438521673e0,
        1.425_626_278_651_89,
        -2.248002557293336e0,
        -5.529177997557566e0,
        -6.395969315158868e0,
        -4.430781857051672e0,
        -9.051773030951376e-1,
        2.245574708973518e0,
        3.716244363531256e0,
        3.474653623363736e0,
    ];

    let cycle = TimeSeries::cf_filter(&cf_reference_series(), 6, 32, true).unwrap();
    assert_all_close(&cycle, &expected, 1e-10);
}

/// ACF at lag 0 is 1 and matches the AR(1) autocorrelation structure.
#[test]
fn test_acf_ar1() {
    let n = 200;
    let mut rng = StdRng::seed_from_u64(9201);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let phi = 0.6;
    let mut series = Vec::with_capacity(n);
    let mut prev = 0.0;
    for _ in 0..n {
        prev = phi * prev + noise.sample(&mut rng);
        series.push(prev);
    }
    let y = Array1::from_vec(series);

    let acf = TimeSeries::acf(&y, 5).unwrap();
    assert_eq!(acf.len(), 6);
    approx_zero(acf[0] - 1.0, 1e-10);
    approx_zero((acf[1] - phi).abs(), 0.15);
    assert!(acf.iter().skip(1).all(|&v| v.abs() <= 1.0));

    // ACF is symmetric in the sense of decaying for higher lags.
    for k in 1..5 {
        assert!(acf[k].abs() >= acf[k + 1].abs() - 0.2);
    }
}

/// PACF of an AR(1) has a single significant spike at lag 1.
#[test]
fn test_pacf_ar1() {
    let n = 200;
    let mut rng = StdRng::seed_from_u64(9202);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let phi = 0.6;
    let mut series = Vec::with_capacity(n);
    let mut prev = 0.0;
    for _ in 0..n {
        prev = phi * prev + noise.sample(&mut rng);
        series.push(prev);
    }
    let y = Array1::from_vec(series);

    let pacf = TimeSeries::pacf(&y, 5).unwrap();
    assert_eq!(pacf.len(), 6);
    approx_zero(pacf[0] - 1.0, 1e-10);
    approx_zero((pacf[1] - phi).abs(), 0.15);

    // Higher-order partial autocorrelations are small.
    for k in 2..=5 {
        assert!(pacf[k].abs() < 0.25, "pacf[{}] = {}", k, pacf[k]);
    }
}

/// ADF correctly identifies a stationary AR(1) series and a random walk.
#[test]
fn test_adf_stationarity() {
    let n = 300;
    let mut rng = StdRng::seed_from_u64(9203);
    let noise = Normal::new(0.0, 1.0).unwrap();

    // Stationary AR(1)
    let mut stationary = Vec::with_capacity(n);
    let mut prev = 0.0;
    for _ in 0..n {
        prev = 0.5 * prev + noise.sample(&mut rng);
        stationary.push(prev);
    }
    let y_stat = Array1::from_vec(stationary);
    let adf_stat = TimeSeries::adf(&y_stat, None).unwrap();
    assert!(adf_stat.is_stationary);
    assert!(adf_stat.test_statistic < adf_stat.critical_values.1); // below 5%
    assert!(adf_stat.n_obs > 0);

    // Random walk (unit root)
    let mut rw = Vec::with_capacity(n);
    let mut prev = 0.0;
    for _ in 0..n {
        prev += noise.sample(&mut rng);
        rw.push(prev);
    }
    let y_rw = Array1::from_vec(rw);
    let adf_rw = TimeSeries::adf(&y_rw, None).unwrap();
    assert!(!adf_rw.is_stationary);
    assert!(adf_rw.test_statistic > adf_rw.critical_values.2); // above 10%
}

/// ACF and PACF input validation.
#[test]
fn test_timeseries_input_validation() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(TimeSeries::acf(&y, 3).is_err());
    assert!(TimeSeries::pacf(&y, 3).is_err());

    let short = Array1::from_vec(vec![1.0]);
    assert!(TimeSeries::acf(&short, 1).is_err());
    assert!(TimeSeries::adf(&short, None).is_err());
}
