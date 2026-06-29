use greeners::*;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

// ─── NegBinP ────────────────────────────────────────────────────────────────

#[test]
fn test_negbinp_nb2() {
    let n = 200;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (i as f64) / n as f64;
        let mu = (0.5 + 1.5 * x[[i, 1]]).exp();
        y[i] = (mu + 0.5).floor().max(0.0);
    }

    let result = NegBinP::fit(&y, &x, 2.0).unwrap();
    assert!(result.converged);
    assert_eq!(result.n_obs, n);
    assert!(result.alpha > 0.0);
    assert_eq!(result.p_param, 2.0);
    println!("{}", result);
}

#[test]
fn test_negbinp_nb1() {
    let n = 200;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (i as f64) / n as f64;
        y[i] = ((0.5 + x[[i, 1]]).exp()).floor().max(0.0);
    }

    let result = NegBinP::fit(&y, &x, 1.0).unwrap();
    assert!(result.converged);
    assert_eq!(result.p_param, 1.0);
}

// ─── GenPoisson ─────────────────────────────────────────────────────────────

#[test]
fn test_gen_poisson() {
    let n = 200;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (i as f64) / n as f64;
        y[i] = ((0.5 + x[[i, 1]]).exp()).floor().max(0.0);
    }

    let result = GenPoisson::fit(&y, &x).unwrap();
    assert_eq!(result.n_obs, n);
    assert!(result.params.len() == 2);
    println!("{}", result);
}

// ─── ConditionalMNLogit ─────────────────────────────────────────────────────

#[test]
fn test_conditional_mnlogit() {
    // 50 choice occasions, 3 alternatives each
    let n_occasions = 50;
    let n_alts = 3;
    let n_rows = n_occasions * n_alts;
    let k = 2;

    let mut x = Array2::<f64>::zeros((n_rows, k));
    let mut y = Array1::<f64>::zeros(n_occasions);
    let mut groups = vec![0usize; n_rows];

    for occ in 0..n_occasions {
        let base = occ * n_alts;
        for j in 0..n_alts {
            groups[base + j] = occ;
            x[[base + j, 0]] = (occ as f64 + j as f64) * 0.1;
            x[[base + j, 1]] = ((occ * 3 + j) % 7) as f64 * 0.2;
        }
        y[occ] = (occ % n_alts) as f64;
    }

    let result = ConditionalMNLogit::fit(&y, &x, &groups, n_alts).unwrap();
    assert_eq!(result.n_obs, n_rows);
    assert_eq!(result.n_groups, n_occasions);
    assert_eq!(result.params.len(), k);
    println!("{}", result);
}

// ─── NominalGEE ─────────────────────────────────────────────────────────────

#[test]
fn test_nominal_gee() {
    let n = 100;
    let k = 2;
    let mut x = Array2::<f64>::zeros((n, k));
    let mut y = Array1::<f64>::zeros(n);
    let mut groups = Array1::<usize>::zeros(n);

    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (i as f64) / n as f64;
        y[i] = (i % 3) as f64;
        groups[i] = i / 5;
    }

    let result = NominalGEE::fit(&y, &x, &groups).unwrap();
    assert_eq!(result.n_obs, n);
    // 3 categories -> 2*(k) params
    assert_eq!(result.params.len(), 2 * k);
    println!("{}", result);
}

// ─── OrdinalGEE ─────────────────────────────────────────────────────────────

#[test]
fn test_ordinal_gee() {
    let n = 100;
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    let mut groups = Array1::<usize>::zeros(n);

    for i in 0..n {
        x[[i, 0]] = (i as f64) / n as f64;
        y[i] = (i % 3) as f64;
        groups[i] = i / 5;
    }

    let result = OrdinalGEE::fit(&y, &x, &groups).unwrap();
    assert_eq!(result.n_obs, n);
    // 2 thresholds + 1 slope
    assert_eq!(result.params.len(), 3);
    println!("{}", result);
}

// ─── BayesMixedGLM ─────────────────────────────────────────────────────────

#[test]
fn test_bayes_mixed_glm() {
    let n = 100;
    let k = 2;
    let mut x = Array2::<f64>::zeros((n, k));
    let mut y = Array1::<f64>::zeros(n);
    let mut groups = Array1::<usize>::zeros(n);

    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (i as f64) / n as f64;
        let p = 1.0 / (1.0 + (-(0.5 + x[[i, 1]])).exp());
        y[i] = if p > 0.5 { 1.0 } else { 0.0 };
        groups[i] = i / 10;
    }

    let result = BayesMixedGLM::fit(&y, &x, &groups, "binomial").unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_groups, 10);
    assert_eq!(result.posterior_mean.len(), k);
    assert_eq!(result.posterior_sd.len(), k);
    assert!(!result.random_effects.is_empty());
    println!("{}", result);
}

// ─── Phillips-Perron ────────────────────────────────────────────────────────

#[test]
fn test_phillips_perron() {
    // Random walk (non-stationary)
    let n = 200;
    let mut series = Array1::<f64>::zeros(n);
    series[0] = 0.0;
    for i in 1..n {
        series[i] = series[i - 1] + 0.1 * ((i * 7) % 13) as f64 - 0.6;
    }

    let result = TimeSeries::phillips_perron(&series, None).unwrap();
    assert!(result.n_obs > 0);
    assert!(result.z_t.is_finite());
    assert!(result.z_alpha.is_finite());
}

#[test]
fn test_phillips_perron_stationary() {
    // Stationary AR(1)
    let n = 200;
    let mut series = Array1::<f64>::zeros(n);
    for i in 1..n {
        series[i] = 0.3 * series[i - 1] + ((i * 7) % 13) as f64 * 0.1 - 0.6;
    }

    let result = TimeSeries::phillips_perron(&series, None).unwrap();
    assert!(result.z_t.is_finite());
}

// ─── Zivot-Andrews ──────────────────────────────────────────────────────────

#[test]
fn test_zivot_andrews() {
    let n = 100;
    let mut series = Array1::<f64>::zeros(n);
    for i in 1..n {
        series[i] =
            series[i - 1] + if i > 50 { 0.5 } else { 0.0 } + ((i * 7) % 13) as f64 * 0.05 - 0.3;
    }

    let result = TimeSeries::zivot_andrews(&series, 0.15).unwrap();
    assert!(result.break_point > 0);
    assert!(result.break_point < n);
    assert!(result.statistic.is_finite());
}

// ─── lagmat ─────────────────────────────────────────────────────────────────

#[test]
fn test_lagmat() {
    let series = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mat = TimeSeries::lagmat(&series, 2);
    assert_eq!(mat.nrows(), 3);
    assert_eq!(mat.ncols(), 3);
    // Row 0: [3, 2, 1], Row 1: [4, 3, 2], Row 2: [5, 4, 3]
    assert!((mat[[0, 0]] - 3.0).abs() < 1e-10);
    assert!((mat[[0, 1]] - 2.0).abs() < 1e-10);
    assert!((mat[[0, 2]] - 1.0).abs() < 1e-10);
    assert!((mat[[2, 0]] - 5.0).abs() < 1e-10);
}

// ─── DeterministicProcess ───────────────────────────────────────────────────

#[test]
fn test_deterministic_process() {
    let mat = TimeSeries::deterministic_process(10, &["const", "trend"]);
    assert_eq!(mat.nrows(), 10);
    assert_eq!(mat.ncols(), 2);
    assert!((mat[[0, 0]] - 1.0).abs() < 1e-10); // const
    assert!((mat[[0, 1]] - 1.0).abs() < 1e-10); // trend starts at 1
    assert!((mat[[9, 1]] - 10.0).abs() < 1e-10);
}

#[test]
fn test_deterministic_process_seasonal() {
    let mat = TimeSeries::deterministic_process(12, &["seasonal:4"]);
    assert_eq!(mat.ncols(), 3); // 4-1 = 3 dummies
    assert!((mat[[0, 0]] - 1.0).abs() < 1e-10); // t=0 -> season 0
    assert!((mat[[1, 1]] - 1.0).abs() < 1e-10); // t=1 -> season 1
}

#[test]
fn test_deterministic_process_fourier() {
    let mat = TimeSeries::deterministic_process(12, &["fourier:12:2"]);
    assert_eq!(mat.ncols(), 4); // 2 orders * 2 (sin+cos)
}

// ─── KDEMultivariate ────────────────────────────────────────────────────────

#[test]
fn test_kde_multivariate() {
    let n = 50;
    let d = 2;
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        data[[i, 0]] = (i as f64) / n as f64;
        data[[i, 1]] = ((i * 7) % 13) as f64 / 13.0;
    }

    let result = KDEMultivariate::fit(&data, None, Kernel::Gaussian).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_dims, d);
    assert_eq!(result.bandwidths.len(), d);

    // Evaluate at a point
    let points = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
    let density = result.evaluate(&points);
    assert_eq!(density.len(), 1);
    assert!(density[0] > 0.0);
    println!("{}", result);
}

// ─── CanCorr ────────────────────────────────────────────────────────────────

#[test]
fn test_cancorr() {
    let n = 50;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / n as f64;
        x[[i, 0]] = t;
        x[[i, 1]] = t * t;
        y[[i, 0]] = t + 0.1 * ((i * 3) % 7) as f64;
        y[[i, 1]] = t * 0.5 + 0.1 * ((i * 5) % 11) as f64;
    }

    let result = CanCorr::fit(&x, &y).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.cancorr.len(), 2);
    assert!(result.cancorr[0] >= 0.0 && result.cancorr[0] <= 1.0);
    assert!(result.wilks_lambda >= 0.0 && result.wilks_lambda <= 1.0);
    println!("{}", result);
}

// ─── MICE ───────────────────────────────────────────────────────────────────

#[test]
fn test_mice() {
    let n = 50;
    let mut data = HashMap::new();
    let mut x1 = Array1::<f64>::zeros(n);
    let mut x2 = Array1::<f64>::zeros(n);
    for i in 0..n {
        x1[i] = i as f64;
        x2[i] = if i % 5 == 0 {
            f64::NAN
        } else {
            (i * 2) as f64 + 1.0
        };
    }
    data.insert("x1".to_string(), x1);
    data.insert("x2".to_string(), x2);

    let result = MICE::impute(&data, 3, 5).unwrap();
    assert_eq!(result.n_imputations, 3);
    assert_eq!(result.datasets.len(), 3);
    // Check no NaN in imputed datasets
    for ds in &result.datasets {
        let x2_imp = &ds["x2"];
        assert!(x2_imp.iter().all(|v| !v.is_nan()));
    }
    println!("{}", result);
}

// ─── BayesGaussMI ───────────────────────────────────────────────────────────

#[test]
fn test_bayes_gauss_mi() {
    let n = 50;
    let mut data = HashMap::new();
    let mut x1 = Array1::<f64>::zeros(n);
    let mut x2 = Array1::<f64>::zeros(n);
    for i in 0..n {
        x1[i] = i as f64;
        x2[i] = if i % 7 == 0 { f64::NAN } else { (i * 3) as f64 };
    }
    data.insert("x1".to_string(), x1);
    data.insert("x2".to_string(), x2);

    let result = BayesGaussMI::impute(&data, 3).unwrap();
    assert_eq!(result.n_imputations, 3);
    for ds in &result.datasets {
        let x2_imp = &ds["x2"];
        assert!(x2_imp.iter().all(|v| !v.is_nan()));
    }
    println!("{}", result);
}

// ─── Harvey-Collier ─────────────────────────────────────────────────────────

#[test]
fn test_harvey_collier() {
    let n = 50;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = i as f64;
        y[i] = 2.0 + 3.0 * x[[i, 1]] + ((i * 7) % 13) as f64 * 0.1;
    }

    let (t_stat, p_value) = Diagnostics::harvey_collier(&y, &x).unwrap();
    assert!(t_stat.is_finite());
    assert!(p_value >= 0.0 && p_value <= 1.0);
}

// ─── Anderson-Darling ───────────────────────────────────────────────────────

#[test]
fn test_anderson_darling() {
    // Approximately normal data
    let data = Array1::from(vec![
        -1.2, -0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.2, -1.0, -0.6, -0.2, 0.2, 0.6,
        1.0, -0.4, 0.4, 0.7,
    ]);

    let result = Diagnostics::anderson_darling(&data).unwrap();
    assert!(result.statistic >= 0.0);
    assert_eq!(result.critical_values.len(), 5);
    assert_eq!(result.n_obs, 20);
}

// ─── Lilliefors ─────────────────────────────────────────────────────────────

#[test]
fn test_lilliefors() {
    let data = Array1::from(vec![
        -1.2, -0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.2, -1.0, -0.6, -0.2, 0.2, 0.6,
        1.0, -0.4, 0.4, 0.7,
    ]);

    let (stat, p_value) = Diagnostics::lilliefors(&data).unwrap();
    assert!(stat >= 0.0);
    assert!(p_value >= 0.0 && p_value <= 1.0);
}

// ─── CompareMeans ───────────────────────────────────────────────────────────

#[test]
fn test_compare_means() {
    let data1 = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let data2 = Array1::from(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

    let result = Stats::compare_means(&data1, &data2, false).unwrap();
    assert!((result.diff - (-1.0)).abs() < 1e-10);
    assert!(result.t_statistic < 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(result.ci_lower < result.ci_upper);
    assert_eq!(result.n1, 10);
    assert_eq!(result.n2, 10);
    println!("{}", result);
}

#[test]
fn test_compare_means_equal() {
    let data1 = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let data2 = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let result = Stats::compare_means(&data1, &data2, false).unwrap();
    assert!((result.diff).abs() < 1e-10);
    assert!(result.p_value > 0.99);
    assert!((result.cohens_d).abs() < 1e-10);
}
