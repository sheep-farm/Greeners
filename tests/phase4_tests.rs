use greeners::*;
use ndarray::{Array1, Array2};

// ─── RLM Tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_rlm_huber() {
    let y = Array1::from(vec![1.0, 2.1, 2.9, 4.0, 5.1, 100.0]); // outlier at 100
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    )
    .unwrap();

    let result = RLM::fit(&y, &x, &RobustNorm::Huber(1.345), CovarianceType::NonRobust).unwrap();
    assert!(result.converged);
    // Slope should be close to 1.0 despite outlier
    assert!((result.params[1] - 1.0).abs() < 1.0);
    println!("{}", result);
}

#[test]
fn test_rlm_tukey() {
    let y = Array1::from(vec![1.0, 2.1, 2.9, 4.0, 5.1, 100.0]);
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    )
    .unwrap();

    let result = RLM::fit(&y, &x, &RobustNorm::Tukey(4.685), CovarianceType::NonRobust).unwrap();
    assert!(result.n_obs == 6);
    assert!(result.params.len() == 2);
}

#[test]
fn test_rlm_andrew_wave() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = RLM::fit(
        &y,
        &x,
        &RobustNorm::AndrewWave(std::f64::consts::PI),
        CovarianceType::NonRobust,
    )
    .unwrap();
    assert!(result.params.len() == 2);
}

#[test]
fn test_rlm_hampel() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )
    .unwrap();

    let result = RLM::fit(
        &y,
        &x,
        &RobustNorm::Hampel(2.0, 4.0, 8.0),
        CovarianceType::NonRobust,
    )
    .unwrap();
    assert!(result.params.len() == 2);
}

// ─── Rolling Tests ─────────────────────────────────────────────────────────────

#[test]
fn test_recursive_ls() {
    let n = 50;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
    let y = Array1::from_vec(
        (0..n)
            .map(|i| 0.5 + 1.0 * i as f64 + 0.1 * (i as f64).sin())
            .collect(),
    );

    let result = RecursiveLS::fit(&y, &x).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.params_history.nrows(), n);
    // Final params should be close to [0.5, 1.0]
    assert!((result.params[1] - 1.0).abs() < 0.5);
    println!("{}", result);
}

#[test]
fn test_rolling_ols() {
    let n = 30;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
    let y = Array1::from_vec((0..n).map(|i| 1.0 + 2.0 * i as f64).collect());

    let result = RollingOLS::fit(&y, &x, 10, None, None).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.window, 10);
    // Last window should give slope ~ 2.0
    let last_slope = result.params_history[[n - 1, 1]];
    assert!((last_slope - 2.0).abs() < 0.01);
    println!("{}", result);
}

#[test]
fn test_rolling_wls() {
    let n = 30;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
    let y = Array1::from_vec((0..n).map(|i| 1.0 + 2.0 * i as f64).collect());
    let weights = Array1::from_elem(n, 1.0);

    let result = RollingWLS::fit(&y, &x, 10, &weights, None, None).unwrap();
    assert_eq!(result.n_obs, n);
}

// ─── GLSAR Tests ───────────────────────────────────────────────────────────────

#[test]
fn test_glsar_ar1() {
    let n = 100;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 * 0.1 });
    // y with AR(1) errors
    let mut y = Array1::<f64>::zeros(n);
    let mut e = 0.0;
    for i in 0..n {
        e = 0.7 * e + 0.1 * ((i * 7) as f64).sin();
        y[i] = 1.0 + 2.0 * x[[i, 1]] + e;
    }

    let result = GLSAR::fit(&y, &x, 1, 50).unwrap();
    assert!(result.n_obs == n);
    assert!(result.rho.len() == 1);
    // AR coefficient should be somewhat close to 0.7
    assert!((result.rho[0] - 0.7).abs() < 0.5);
    println!("{}", result);
}

#[test]
fn test_glsar_ar2() {
    let n = 100;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 * 0.1 });
    let y = Array1::from_vec((0..n).map(|i| 1.0 + 2.0 * i as f64 * 0.1).collect());

    let result = GLSAR::fit(&y, &x, 2, 50).unwrap();
    assert_eq!(result.rho.len(), 2);
}

// ─── Nonparametric Tests ───────────────────────────────────────────────────────

#[test]
fn test_kde_gaussian() {
    let data = Array1::from(vec![-1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 1.2, 1.5, 2.0, 2.5]);

    let result = KDEUnivariate::fit(&data, None, Kernel::Gaussian).unwrap();
    assert_eq!(result.n_obs, 10);
    assert!(result.bandwidth > 0.0);
    // Density should be positive everywhere
    assert!(result.density.iter().all(|&d| d >= 0.0));
    println!("{}", result);
}

#[test]
fn test_kde_epanechnikov() {
    let data = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = KDEUnivariate::fit(&data, Some(1.0), Kernel::Epanechnikov).unwrap();
    assert!(result.density.iter().all(|&d| d >= 0.0));
}

#[test]
fn test_lowess() {
    let x = Array1::from_vec((0..30).map(|i| i as f64).collect());
    let y = x.mapv(|xi| xi.sin() + 0.1 * xi);

    let result = Lowess::fit(&y, &x, 0.3, 1).unwrap();
    assert_eq!(result.n_obs, 30);
    assert_eq!(result.smoothed.len(), 30);
    println!("{}", result);
}

#[test]
fn test_kernel_reg() {
    let x = Array1::from_vec((0..50).map(|i| i as f64 * 0.1).collect());
    let y = x.mapv(|xi| (xi * 2.0).sin());

    let result = KernelReg::fit(&y, &x, None, Kernel::Gaussian).unwrap();
    assert_eq!(result.n_obs, 50);
    assert!(result.bandwidth > 0.0);
    println!("{}", result);
}

// ─── PCA Tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_pca() {
    // 3 variables, 2 are highly correlated
    let data = Array2::from_shape_fn((50, 3), |(i, j)| match j {
        0 => i as f64,
        1 => i as f64 + 0.1 * (i as f64).sin(),
        2 => (i as f64 * 0.3).cos(),
        _ => 0.0,
    });

    let result = PCA::fit(&data, 2).unwrap();
    assert_eq!(result.n_components, 2);
    assert_eq!(result.scores.nrows(), 50);
    assert_eq!(result.scores.ncols(), 2);
    // First component should explain most variance
    assert!(result.explained_variance_ratio[0] > result.explained_variance_ratio[1]);
    println!("{}", result);

    // Test transform
    let transformed = result.transform(&data);
    assert_eq!(transformed.nrows(), 50);
    assert_eq!(transformed.ncols(), 2);
}

#[test]
fn test_factor_analysis() {
    let data = Array2::from_shape_fn((50, 4), |(i, j)| {
        (i as f64 * (j + 1) as f64 * 0.1).sin() + i as f64 * 0.01
    });

    let result = FactorAnalysis::fit(&data, 2, Rotation::None).unwrap();
    assert_eq!(result.n_factors, 2);
    assert_eq!(result.loadings.nrows(), 4);
    assert_eq!(result.communalities.len(), 4);
    println!("{}", result);
}

#[test]
fn test_factor_analysis_varimax() {
    let data = Array2::from_shape_fn((50, 4), |(i, j)| {
        (i as f64 * (j + 1) as f64 * 0.1).sin() + i as f64 * 0.01
    });

    let result = FactorAnalysis::fit(&data, 2, Rotation::Varimax).unwrap();
    assert_eq!(result.n_factors, 2);
}

#[test]
fn test_manova() {
    let n = 60;
    let y = Array2::from_shape_fn((n, 3), |(i, j)| {
        let group = i / 20;
        (group as f64 + 1.0) * (j as f64 + 1.0) + 0.1 * (i as f64).sin()
    });
    let groups = Array1::from_vec((0..n).map(|i| i / 20).collect());

    let result = MANOVA::fit(&y, &groups).unwrap();
    assert_eq!(result.n_groups, 3);
    assert_eq!(result.n_vars, 3);
    assert!(result.wilks_lambda >= 0.0 && result.wilks_lambda <= 1.0);
    println!("{}", result);
}

// ─── Beta Model Tests ──────────────────────────────────────────────────────────

#[test]
fn test_beta_model() {
    let n = 100;
    // Response in (0, 1)
    let x = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            (i as f64 - 50.0) / 50.0
        }
    });
    let y = Array1::from_vec(
        (0..n)
            .map(|i| {
                let eta = -0.5 + 1.0 * (i as f64 - 50.0) / 50.0;
                let p = 1.0 / (1.0 + (-eta).exp());
                p.clamp(0.01, 0.99)
            })
            .collect(),
    );

    let result = BetaModel::fit(&y, &x, &BetaLink::Logit).unwrap();
    assert_eq!(result.n_obs, n);
    assert!(result.params.len() == 2);
    assert!(result.precision_param > 0.0);
    println!("{}", result);
}

// ─── MixedLM Tests ─────────────────────────────────────────────────────────────

#[test]
fn test_mixed_lm() {
    let n = 60;
    let n_groups = 6;
    let per_group = n / n_groups;

    let x_fixed = Array2::from_shape_fn(
        (n, 2),
        |(i, j)| {
            if j == 0 {
                1.0
            } else {
                (i % per_group) as f64
            }
        },
    );
    let x_random = Array2::from_shape_fn((n, 1), |_| 1.0); // random intercept
    let groups = Array1::from_vec((0..n).map(|i| i / per_group).collect());

    // y = fixed + random intercept + noise
    let y = Array1::from_vec(
        (0..n)
            .map(|i| {
                let group = i / per_group;
                1.0 + 0.5 * (i % per_group) as f64 + group as f64 * 0.3 + 0.1 * (i as f64).sin()
            })
            .collect(),
    );

    let result = MixedLM::fit(&y, &x_fixed, &groups, &x_random).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_groups, n_groups);
    assert_eq!(result.fixed_effects.len(), 2);
    assert!(result.var_resid > 0.0);
    println!("{}", result);
}

// ─── Survival Tests ────────────────────────────────────────────────────────────

#[test]
fn test_kaplan_meier() {
    let times = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let events = Array1::from(vec![1u8, 0, 1, 1, 0, 1, 0, 1, 1, 0]);

    let result = KaplanMeier::fit(&times, &events).unwrap();
    assert_eq!(result.n_obs, 10);
    assert_eq!(result.n_events, 6);
    assert!(result.survival_probs[0] < 1.0);
    // Survival should be non-increasing
    for i in 1..result.survival_probs.len() {
        assert!(result.survival_probs[i] <= result.survival_probs[i - 1] + 1e-10);
    }
    println!("{}", result);
}

#[test]
fn test_cox_ph() {
    let n = 50;
    let times = Array1::from_vec(
        (0..n)
            .map(|i| (i as f64 + 1.0) + 0.5 * (i as f64).sin())
            .collect(),
    );
    let events = Array1::from_vec((0..n).map(|i| if i % 3 != 0 { 1u8 } else { 0 }).collect());
    let x = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            i as f64 / n as f64
        } else {
            ((i * 3) as f64).sin()
        }
    });

    let result = CoxPH::fit(&times, &events, &x).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.params.len(), 2);
    assert!(result.concordance >= 0.0 && result.concordance <= 1.0);
    assert_eq!(result.hazard_ratios.len(), 2);
    println!("{}", result);
}

// ─── GEE Tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_gee_independence() {
    let n = 60;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { (i % 10) as f64 });
    let y = Array1::from_vec(
        (0..n)
            .map(|i| 1.0 + 0.5 * (i % 10) as f64 + 0.1 * (i as f64).sin())
            .collect(),
    );
    let groups = Array1::from_vec((0..n).map(|i| i / 10).collect());

    let result = GEE::fit(
        &y,
        &x,
        &groups,
        &Family::Gaussian,
        &Link::Identity,
        &CorrStructure::Independence,
    )
    .unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_groups, 6);
    assert!(result.robust_se.iter().all(|&s| s > 0.0));
    println!("{}", result);
}

#[test]
fn test_gee_exchangeable() {
    let n = 60;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { (i % 10) as f64 });
    let y = Array1::from_vec((0..n).map(|i| 1.0 + 0.5 * (i % 10) as f64).collect());
    let groups = Array1::from_vec((0..n).map(|i| i / 10).collect());

    let result = GEE::fit(
        &y,
        &x,
        &groups,
        &Family::Gaussian,
        &Link::Identity,
        &CorrStructure::Exchangeable,
    )
    .unwrap();
    assert_eq!(result.n_groups, 6);
}

#[test]
fn test_gee_ar1() {
    let n = 60;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 * 0.1 });
    let y = Array1::from_vec((0..n).map(|i| 1.0 + 0.3 * i as f64 * 0.1).collect());
    let groups = Array1::from_vec((0..n).map(|i| i / 10).collect());

    let result = GEE::fit(
        &y,
        &x,
        &groups,
        &Family::Gaussian,
        &Link::Identity,
        &CorrStructure::AR1,
    )
    .unwrap();
    assert_eq!(result.n_groups, 6);
}

// ─── GLMGam Tests ──────────────────────────────────────────────────────────────

#[test]
fn test_bspline_basis() {
    let x = Array1::from_vec((0..50).map(|i| i as f64 * 0.1).collect());
    let basis = BSplineBasis::generate(&x, 8, 3).unwrap();
    assert_eq!(basis.nrows(), 50);
    assert_eq!(basis.ncols(), 8);
}

#[test]
fn test_glmgam_gaussian() {
    let n = 100;
    let x_lin = Array2::from_shape_fn((n, 1), |_| 1.0); // intercept only
    let x_raw = Array1::from_vec((0..n).map(|i| i as f64 * 0.1).collect());
    let x_smooth = BSplineBasis::generate(&x_raw, 6, 3).unwrap();

    let y = Array1::from_vec(
        (0..n)
            .map(|i| (i as f64 * 0.1).sin() + 0.05 * (i as f64).cos())
            .collect(),
    );

    let result = GLMGam::fit(
        &y,
        &x_lin,
        &x_smooth,
        &Family::Gaussian,
        &Link::Identity,
        1.0,
    )
    .unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_linear, 1);
    assert_eq!(result.n_smooth, 6);
    assert!(result.edf > 0.0);
    println!("{}", result);
}

// ─── Display Tests ─────────────────────────────────────────────────────────────

#[test]
fn test_all_displays() {
    // Just verify Display impls don't panic
    let y5 = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x5 = Array2::from_shape_fn((5, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });

    let rlm = RLM::fit(&y5, &x5, &RobustNorm::default(), CovarianceType::NonRobust).unwrap();
    let _ = format!("{}", rlm);

    let times = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let events = Array1::from(vec![1u8, 0, 1, 1, 0]);
    let km = KaplanMeier::fit(&times, &events).unwrap();
    let _ = format!("{}", km);
}
