use greeners::{CovarianceType, Diagnostics, Family, Link, TimeSeries, GLM, OLS, VAR, WLS};
use ndarray::{Array1, Array2};

/// Simple LCG pseudo-random number generator for deterministic tests
fn lcg_sequence(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-0.5, 0.5]
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        })
        .collect()
}

// ─── ACF / PACF ──────────────────────────────────────────────────────────────

#[test]
fn test_acf_basic() {
    let noise = lcg_sequence(2000, 42);
    let series = Array1::from_vec(noise);
    let acf = TimeSeries::acf(&series, 10).unwrap();
    assert!((acf[0] - 1.0).abs() < 1e-10);
    // For large n, ACF of noise should be small
    for k in 1..=10 {
        assert!(acf[k].abs() < 0.1, "ACF at lag {} = {}", k, acf[k]);
    }
}

#[test]
fn test_acf_ar1() {
    let phi = 0.8;
    let noise = lcg_sequence(10000, 123);
    let mut series = vec![0.0f64; 10000];
    for t in 1..10000 {
        series[t] = phi * series[t - 1] + noise[t] * 0.5;
    }
    let series = Array1::from_vec(series);
    let acf = TimeSeries::acf(&series, 5).unwrap();
    assert!(
        (acf[1] - phi).abs() < 0.1,
        "ACF(1) = {}, expected ~{}",
        acf[1],
        phi
    );
}

#[test]
fn test_pacf_ar1() {
    let phi = 0.7;
    let noise = lcg_sequence(10000, 456);
    let mut series = vec![0.0f64; 10000];
    for t in 1..10000 {
        series[t] = phi * series[t - 1] + noise[t] * 0.5;
    }
    let series = Array1::from_vec(series);
    let pacf = TimeSeries::pacf(&series, 5).unwrap();
    assert!(
        (pacf[1] - phi).abs() < 0.1,
        "PACF(1) = {}, expected ~{}",
        pacf[1],
        phi
    );
    for k in 3..=5 {
        assert!(pacf[k].abs() < 0.1, "PACF({}) = {}", k, pacf[k]);
    }
}

// ─── KPSS ────────────────────────────────────────────────────────────────────

#[test]
fn test_kpss_stationary() {
    let series = Array1::from_vec(lcg_sequence(200, 789));
    let result = TimeSeries::kpss(&series, "c", Some(5)).unwrap();
    assert!(
        result.is_stationary,
        "Stationary series should not reject KPSS H0, stat={}",
        result.test_statistic
    );
}

#[test]
fn test_kpss_random_walk() {
    let noise = lcg_sequence(200, 321);
    let mut series = vec![0.0f64; 200];
    for t in 1..200 {
        series[t] = series[t - 1] + noise[t];
    }
    let series = Array1::from_vec(series);
    let result = TimeSeries::kpss(&series, "c", Some(5)).unwrap();
    assert!(
        result.test_statistic > result.critical_values.0,
        "Random walk should reject KPSS, stat={}, cv10={}",
        result.test_statistic,
        result.critical_values.0
    );
}

// ─── Ljung-Box ───────────────────────────────────────────────────────────────

#[test]
fn test_ljung_box_white_noise() {
    let series = Array1::from_vec(lcg_sequence(1000, 654));
    let result = TimeSeries::ljung_box(&series, 10).unwrap();
    assert!(
        result.p_value > 0.01,
        "White noise p-value should be > 0.01, got {}",
        result.p_value
    );
}

#[test]
fn test_ljung_box_autocorrelated() {
    let noise = lcg_sequence(1000, 987);
    let mut series = vec![0.0f64; 1000];
    for t in 1..1000 {
        series[t] = 0.9 * series[t - 1] + noise[t] * 0.3;
    }
    let series = Array1::from_vec(series);
    let result = TimeSeries::ljung_box(&series, 10).unwrap();
    assert!(
        result.p_value < 0.05,
        "AR(1) should reject Ljung-Box, p={}",
        result.p_value
    );
}

// ─── ARCH test ───────────────────────────────────────────────────────────────

#[test]
fn test_arch_no_effects() {
    let residuals = Array1::from_vec(lcg_sequence(500, 111));
    let result = TimeSeries::arch_test(&residuals, 5).unwrap();
    assert!(
        result.p_value > 0.01,
        "No ARCH effects expected, p={}",
        result.p_value
    );
}

// ─── Granger Causality ───────────────────────────────────────────────────────

#[test]
fn test_granger_causality_causal() {
    let noise_x = lcg_sequence(500, 222);
    let noise_y = lcg_sequence(500, 333);

    let mut x = vec![0.0f64; 500];
    let mut y = vec![0.0f64; 500];
    for t in 1..500 {
        x[t] = 0.5 * x[t - 1] + noise_x[t];
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + noise_y[t] * 0.3;
    }

    let x = Array1::from_vec(x);
    let y = Array1::from_vec(y);

    let result = TimeSeries::granger_causality(&y, &x, 2).unwrap();
    assert!(
        result.p_value < 0.05,
        "x should Granger-cause y, p={}",
        result.p_value
    );
}

// ─── Engle-Granger ───────────────────────────────────────────────────────────

#[test]
fn test_engle_granger_cointegrated() {
    let noise_rw = lcg_sequence(300, 444);
    let noise_stat = lcg_sequence(300, 555);

    let mut y2 = vec![0.0f64; 300];
    for t in 1..300 {
        y2[t] = y2[t - 1] + noise_rw[t];
    }
    let y1: Vec<f64> = (0..300)
        .map(|t| 2.0 * y2[t] + noise_stat[t] * 0.2)
        .collect();

    let y1 = Array1::from_vec(y1);
    let y2 = Array1::from_vec(y2);

    let result = TimeSeries::engle_granger(&y1, &y2).unwrap();
    assert!(
        result.adf_statistic < -2.0,
        "Cointegrated series ADF stat should be negative, got {}",
        result.adf_statistic
    );
}

// ─── Johansen ────────────────────────────────────────────────────────────────

#[test]
fn test_johansen_basic() {
    let noise1 = lcg_sequence(300, 666);
    let noise2 = lcg_sequence(300, 777);

    let mut data = Array2::<f64>::zeros((300, 2));
    for t in 1..300 {
        data[[t, 0]] = data[[t - 1, 0]] + noise1[t];
        data[[t, 1]] = data[[t, 0]] + noise2[t] * 0.2;
    }

    let result = TimeSeries::johansen(&data, 1, 1);
    assert!(result.is_ok());
    let res = result.unwrap();
    assert_eq!(res.n_vars, 2);
    assert_eq!(res.trace_stats.len(), 2);
}

// ─── FEVD ────────────────────────────────────────────────────────────────────

#[test]
fn test_fevd_sums_to_one() {
    let noise1 = lcg_sequence(200, 888);
    let noise2 = lcg_sequence(200, 999);

    let mut data = Array2::<f64>::zeros((200, 2));
    for t in 1..200 {
        data[[t, 0]] = 0.5 * data[[t - 1, 0]] + 0.2 * data[[t - 1, 1]] + noise1[t];
        data[[t, 1]] = 0.3 * data[[t - 1, 0]] + 0.4 * data[[t - 1, 1]] + noise2[t];
    }

    let var_result = VAR::fit(&data, 1, None).unwrap();
    let fevd = var_result.fevd(10).unwrap();

    for h in 0..10 {
        for i in 0..2 {
            let row_sum: f64 = (0..2).map(|j| fevd[[h, i, j]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "FEVD row sum at h={}, i={} = {}",
                h,
                i,
                row_sum
            );
        }
    }
}

// ─── OLS conf_int ────────────────────────────────────────────────────────────

#[test]
fn test_ols_conf_int() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let mut x = Array2::<f64>::zeros((10, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..10 {
        x[[i, 1]] = (i + 1) as f64;
    }
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    let ci_95 = result.conf_int(0.05).unwrap();
    let ci_99 = result.conf_int(0.01).unwrap();

    // 99% CI should be wider than 95% CI
    for i in 0..2 {
        let width_95 = ci_95[i].1 - ci_95[i].0;
        let width_99 = ci_99[i].1 - ci_99[i].0;
        assert!(width_99 > width_95);
    }
}

// ─── OLS get_prediction ──────────────────────────────────────────────────────

#[test]
fn test_ols_get_prediction() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let mut x = Array2::<f64>::zeros((10, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..10 {
        x[[i, 1]] = (i + 1) as f64;
    }
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    let mut x_new = Array2::<f64>::zeros((2, 2));
    x_new[[0, 0]] = 1.0;
    x_new[[0, 1]] = 11.0;
    x_new[[1, 0]] = 1.0;
    x_new[[1, 1]] = 12.0;

    let pred = result.get_prediction(&x_new, &x, 0.05).unwrap();
    assert_eq!(pred.mean.len(), 2);
    assert!(pred.se[0] > 0.0);
    assert!(pred.ci_lower[0] < pred.mean[0]);
    assert!(pred.ci_upper[0] > pred.mean[0]);
}

// ─── OLS Wald/F/t tests ─────────────────────────────────────────────────────

#[test]
fn test_ols_f_test() {
    let noise = lcg_sequence(50, 42);
    let y = Array1::from_vec((0..50).map(|i| 2.0 + 3.0 * i as f64 + noise[i]).collect());
    let mut x = Array2::<f64>::zeros((50, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..50 {
        x[[i, 1]] = i as f64;
    }
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    let (f_stat, p_value) = result.f_test(&[1], &x).unwrap();
    assert!(f_stat > 0.0);
    assert!(p_value < 0.01, "Slope should be significant, p={}", p_value);
}

#[test]
fn test_ols_t_test() {
    let noise = lcg_sequence(50, 77);
    let y = Array1::from_vec(
        (0..50)
            .map(|i| 1.0 + 2.0 * i as f64 + noise[i] * 0.5)
            .collect(),
    );
    let mut x = Array2::<f64>::zeros((50, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..50 {
        x[[i, 1]] = i as f64;
    }
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    // Test H0: slope = 2
    let r = Array1::from(vec![0.0, 1.0]);
    let (t_stat, _p_value) = result.t_test(&r, 2.0, &x).unwrap();
    // Should be close to 0 since true slope is 2
    assert!(
        t_stat.abs() < 5.0,
        "t-stat for beta=2 should be moderate, got {}",
        t_stat
    );
}

// ─── WLS ─────────────────────────────────────────────────────────────────────

#[test]
fn test_wls_equal_weights_equals_ols() {
    let y = Array1::from(vec![1.0, 2.1, 2.9, 4.0, 5.1]);
    let mut x = Array2::<f64>::zeros((5, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..5 {
        x[[i, 1]] = (i + 1) as f64;
    }
    let weights = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);

    let ols_result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let wls_result = WLS::fit(&y, &x, &weights, CovarianceType::NonRobust).unwrap();

    for i in 0..2 {
        assert!(
            (ols_result.params[i] - wls_result.params[i]).abs() < 1e-10,
            "OLS and WLS with equal weights should match"
        );
    }
}

#[test]
fn test_wls_different_weights() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut x = Array2::<f64>::zeros((8, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..8 {
        x[[i, 1]] = (i + 1) as f64;
    }
    let weights = Array1::from(vec![0.1, 0.2, 0.5, 1.0, 1.0, 2.0, 3.0, 5.0]);
    let result = WLS::fit(&y, &x, &weights, CovarianceType::NonRobust).unwrap();
    assert_eq!(result.params.len(), 2);
    assert!(result.params.iter().all(|p| p.is_finite()));
}

// ─── GLM ─────────────────────────────────────────────────────────────────────

#[test]
fn test_glm_conf_int() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut x = Array2::<f64>::zeros((5, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..5 {
        x[[i, 1]] = (i + 1) as f64;
    }
    let result = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();
    let ci = result.conf_int(0.05);
    assert_eq!(ci.len(), 2);
    for i in 0..2 {
        assert!(ci[i].0 < result.params[i]);
        assert!(ci[i].1 > result.params[i]);
    }
}

#[test]
fn test_glm_cloglog_link() {
    // Use more data and stronger signal for convergence
    let n = 50;
    let noise = lcg_sequence(n, 42);
    let y = Array1::from_vec(
        (0..n)
            .map(|i| {
                if i as f64 / n as f64 + noise[i] * 0.2 > 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
    );
    let mut x = Array2::<f64>::zeros((n, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..n {
        x[[i, 1]] = i as f64 / n as f64;
    }
    let result = GLM::fit_with_link(
        &y,
        &x,
        Family::Binomial,
        Link::CLogLog,
        CovarianceType::NonRobust,
    );
    assert!(result.is_ok(), "CLogLog GLM should not error");
}

#[test]
fn test_glm_cauchy_link() {
    let n = 50;
    let noise = lcg_sequence(n, 99);
    let y = Array1::from_vec(
        (0..n)
            .map(|i| {
                if i as f64 / n as f64 + noise[i] * 0.2 > 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
    );
    let mut x = Array2::<f64>::zeros((n, 2));
    x.column_mut(0).fill(1.0);
    for i in 0..n {
        x[[i, 1]] = i as f64 / n as f64;
    }
    let result = GLM::fit_with_link(
        &y,
        &x,
        Family::Binomial,
        Link::Cauchy,
        CovarianceType::NonRobust,
    );
    assert!(result.is_ok(), "Cauchy link GLM should not error");
}

// ─── Omnibus test ────────────────────────────────────────────────────────────

#[test]
fn test_omnibus_normal() {
    // Sum of multiple uniforms ~ approximately normal
    let u1 = lcg_sequence(200, 11);
    let u2 = lcg_sequence(200, 22);
    let u3 = lcg_sequence(200, 33);
    let residuals = Array1::from_vec((0..200).map(|i| u1[i] + u2[i] + u3[i]).collect());
    let (stat, p_value) = Diagnostics::omnibus(&residuals).unwrap();
    assert!(stat.is_finite());
    assert!(p_value >= 0.0 && p_value <= 1.0);
}
