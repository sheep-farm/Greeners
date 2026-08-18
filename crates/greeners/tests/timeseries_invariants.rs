use greeners::TimeSeries;
use ndarray::Array1;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
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
