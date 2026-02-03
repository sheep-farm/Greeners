use greeners::*;
use ndarray::{Array1, Array2};

// LCG PRNG for deterministic test data
fn lcg_sequence(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [0, 1)
            (state >> 33) as f64 / (1u64 << 31) as f64
        })
        .collect()
}

fn with_intercept(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let mut x_new = Array2::<f64>::zeros((n, k + 1));
    x_new.column_mut(0).fill(1.0);
    x_new.slice_mut(ndarray::s![.., 1..]).assign(x);
    x_new
}

// ====================== Poisson ======================

#[test]
fn test_poisson_basic() {
    let rng = lcg_sequence(200, 42);
    let n = 100;
    let x_vals: Vec<f64> = rng[..n].to_vec();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let mu = (0.5 + 1.5 * x_vals[i]).exp();
            // Approximate Poisson draw: round(mu + noise)
            (mu + (rng[n + i] - 0.5) * mu.sqrt() * 2.0).round().max(0.0)
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = Poisson::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    assert!(res.converged);
    assert!(res.log_likelihood.is_finite());
    assert!(res.params.len() == 2);
    assert!(res.aic.is_finite());
}

#[test]
fn test_poisson_overdispersion() {
    let rng = lcg_sequence(400, 123);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].to_vec();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let mu = (1.0 + x_vals[i]).exp();
            // Overdispersed: add extra variance
            (mu + (rng[n + i] - 0.5) * mu * 3.0).round().max(0.0)
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = Poisson::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let (t_stat, _p_val) = res.overdispersion_test().unwrap();
    assert!(t_stat.is_finite());
}

#[test]
fn test_poisson_predict() {
    let rng = lcg_sequence(200, 77);
    let n = 50;
    let x_vals: Vec<f64> = rng[..n].to_vec();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| ((1.0 + x_vals[i]).exp()).round().max(0.0))
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = Poisson::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let counts = res.predict_count(&x);
    assert_eq!(counts.len(), n);
    assert!(counts.iter().all(|c| *c > 0.0));

    let fitted = res.fitted_values();
    assert_eq!(fitted.len(), n);

    let me = res.marginal_effects(&x);
    assert_eq!(me.len(), 2);
}

// ====================== NegBin ======================

#[test]
fn test_negbin_basic() {
    let rng = lcg_sequence(400, 99);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].to_vec();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let mu = (0.5 + x_vals[i]).exp();
            (mu + (rng[n + i] - 0.5) * mu * 2.0).round().max(0.0)
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = NegBin::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    assert!(res.converged);
    assert!(res.alpha > 0.0);
    assert!(res.log_likelihood.is_finite());
}

#[test]
fn test_negbin_with_known_alpha() {
    let rng = lcg_sequence(200, 55);
    let n = 100;
    let x_vals: Vec<f64> = rng[..n].to_vec();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let mu = (0.5 + x_vals[i]).exp();
            (mu + (rng[n + i] - 0.5) * mu).round().max(0.0)
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = NegBin::fit_with_alpha(&y, &x, 1.0, CovarianceType::NonRobust, None).unwrap();
    assert!(res.converged);
    assert!((res.alpha - 1.0).abs() < 1e-10);
}

#[test]
fn test_negbin_lr_test() {
    let rng = lcg_sequence(400, 88);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].to_vec();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let mu = (0.5 + x_vals[i]).exp();
            (mu + (rng[n + i] - 0.5) * mu * 3.0).round().max(0.0)
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let pois_res = Poisson::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let nb_res = NegBin::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    let (lr, p) = nb_res.lr_test_vs_poisson(pois_res.log_likelihood);
    assert!(lr.is_finite());
    assert!(p >= 0.0 && p <= 1.0);
}

// ====================== MNLogit ======================

#[test]
fn test_mnlogit_basic() {
    let rng = lcg_sequence(600, 42);
    let n = 150;
    let x_vals: Vec<f64> = rng[..n].iter().map(|v| v * 4.0 - 2.0).collect();

    // 3 categories based on x thresholds + noise
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let v = x_vals[i] + (rng[n + i] - 0.5) * 2.0;
            if v < -0.5 {
                0.0
            } else if v < 0.5 {
                1.0
            } else {
                2.0
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = MNLogit::fit(&y, &x).unwrap();
    assert!(res.converged);
    assert_eq!(res.n_categories, 3);
    assert_eq!(res.params.nrows(), 2); // k = 2 (const + x)
    assert_eq!(res.params.ncols(), 2); // J-1 = 2
    assert!(res.log_likelihood.is_finite());
}

#[test]
fn test_mnlogit_predict() {
    let rng = lcg_sequence(600, 77);
    let n = 150;
    let x_vals: Vec<f64> = rng[..n].iter().map(|v| v * 4.0 - 2.0).collect();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let v = x_vals[i] + (rng[n + i] - 0.5);
            if v < -0.3 {
                0.0
            } else if v < 0.3 {
                1.0
            } else {
                2.0
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = MNLogit::fit(&y, &x).unwrap();
    let probs = res.predict_proba(&x);
    assert_eq!(probs.nrows(), n);
    assert_eq!(probs.ncols(), 3);

    // Each row sums to ~1
    for i in 0..n {
        let row_sum: f64 = probs.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }

    let preds = res.predict(&x);
    assert_eq!(preds.len(), n);

    let rrr = res.rrr();
    assert!(rrr.iter().all(|v| *v > 0.0));
}

// ====================== OrderedModel ======================

#[test]
fn test_ordered_logit_basic() {
    let rng = lcg_sequence(600, 42);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].iter().map(|v| v * 4.0 - 2.0).collect();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let v = x_vals[i] + (rng[n + i] - 0.5) * 2.0;
            if v < -1.0 {
                1.0
            } else if v < 0.0 {
                2.0
            } else if v < 1.0 {
                3.0
            } else {
                4.0
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();

    let res = OrderedLogit::fit(&y, &x_raw).unwrap();
    assert!(res.converged);
    assert_eq!(res.n_categories, 4);
    assert_eq!(res.thresholds.len(), 3); // J-1 = 3
    assert!(res.params.len() == 1);
    // Thresholds should be ordered
    for i in 1..res.thresholds.len() {
        assert!(res.thresholds[i] > res.thresholds[i - 1]);
    }
}

#[test]
fn test_ordered_probit_basic() {
    let rng = lcg_sequence(600, 55);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].iter().map(|v| v * 4.0 - 2.0).collect();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let v = x_vals[i] + (rng[n + i] - 0.5) * 2.0;
            if v < -0.5 {
                0.0
            } else if v < 0.5 {
                1.0
            } else {
                2.0
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();

    let res = OrderedProbit::fit(&y, &x_raw).unwrap();
    assert!(res.converged);
    assert_eq!(res.n_categories, 3);
    assert!(res.log_likelihood.is_finite());
}

#[test]
fn test_ordered_predict() {
    let rng = lcg_sequence(600, 33);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].iter().map(|v| v * 4.0 - 2.0).collect();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let v = x_vals[i] + (rng[n + i] - 0.5) * 2.0;
            if v < -0.5 {
                0.0
            } else if v < 0.5 {
                1.0
            } else {
                2.0
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();

    let res = OrderedLogit::fit(&y, &x_raw).unwrap();
    let probs = res.predict_proba(&x_raw);
    assert_eq!(probs.nrows(), n);
    assert_eq!(probs.ncols(), 3);

    for i in 0..n {
        let row_sum: f64 = probs.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }

    let preds = res.predict(&x_raw);
    assert_eq!(preds.len(), n);
}

// ====================== Zero-Inflated ======================

#[test]
fn test_zip_basic() {
    let rng = lcg_sequence(600, 42);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].to_vec();

    // Generate zero-inflated Poisson data
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            // Inflate probability ~ 0.3
            if rng[n + i] < 0.3 {
                0.0 // structural zero
            } else {
                let mu = (0.5 + x_vals[i]).exp();
                (mu + (rng[2 * n + i] - 0.5) * mu.sqrt() * 2.0)
                    .round()
                    .max(0.0)
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = ZIP::fit(&y, &x, None).unwrap();
    assert!(res.log_likelihood.is_finite());
    assert_eq!(res.count_params.len(), 2);
    assert_eq!(res.inflate_params.len(), 2);
    assert!(res.alpha.is_none());
}

#[test]
fn test_zinb_basic() {
    let rng = lcg_sequence(600, 99);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].to_vec();

    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            if rng[n + i] < 0.3 {
                0.0
            } else {
                let mu = (0.5 + x_vals[i]).exp();
                (mu + (rng[2 * n + i] - 0.5) * mu * 2.0).round().max(0.0)
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = ZINB::fit(&y, &x, None).unwrap();
    assert!(res.log_likelihood.is_finite());
    assert!(res.alpha.is_some());
}

#[test]
fn test_zip_predict() {
    let rng = lcg_sequence(600, 55);
    let n = 200;
    let x_vals: Vec<f64> = rng[..n].to_vec();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            if rng[n + i] < 0.25 {
                0.0
            } else {
                let mu = (0.5 + x_vals[i]).exp();
                (mu + (rng[2 * n + i] - 0.5) * mu.sqrt()).round().max(0.0)
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);

    let res = ZIP::fit(&y, &x, None).unwrap();
    let counts = res.predict_count(&x, &x);
    assert_eq!(counts.len(), n);
    assert!(counts.iter().all(|c| *c >= 0.0));

    let p_zero = res.predict_proba_zero(&x, &x);
    assert_eq!(p_zero.len(), n);
    assert!(p_zero.iter().all(|p| *p >= 0.0 && *p <= 1.0));
}

// ====================== Conditional Models ======================

#[test]
fn test_conditional_logit_basic() {
    let rng = lcg_sequence(600, 42);
    // 20 groups, 5 obs each
    let n_groups = 20;
    let n_per_group = 5;
    let n = n_groups * n_per_group;

    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut groups = Vec::new();

    for g in 0..n_groups {
        let group_effect = (rng[g] - 0.5) * 2.0;
        for t in 0..n_per_group {
            let idx = g * n_per_group + t;
            let x = rng[n_groups + idx] * 2.0;
            x_vals.push(x);
            // P(y=1) depends on x + group effect
            let prob = 1.0 / (1.0 + (-(group_effect + 1.5 * x)).exp());
            y_vals.push(if rng[2 * n_groups + idx] < prob {
                1.0
            } else {
                0.0
            });
            groups.push(g);
        }
    }

    let y = Array1::from(y_vals);
    let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();

    let res = ConditionalLogit::fit(&y, &x, &groups).unwrap();
    assert!(res.converged);
    assert!(res.params.len() == 1);
    assert!(res.log_likelihood.is_finite());
    // Coefficient should be positive (true effect is +1.5)
    assert!(res.params[0] > 0.0);
}

#[test]
fn test_conditional_poisson_basic() {
    let rng = lcg_sequence(600, 77);
    let n_groups = 20;
    let n_per_group = 5;
    let n = n_groups * n_per_group;

    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut groups = Vec::new();

    for g in 0..n_groups {
        let group_effect = rng[g] * 2.0;
        for t in 0..n_per_group {
            let idx = g * n_per_group + t;
            let x = rng[n_groups + idx] * 2.0;
            x_vals.push(x);
            let mu = (group_effect + 0.5 * x).exp();
            y_vals.push(
                (mu + (rng[2 * n_groups + idx] - 0.5) * mu.sqrt())
                    .round()
                    .max(0.0),
            );
            groups.push(g);
        }
    }

    let y = Array1::from(y_vals);
    let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();

    let res = ConditionalPoisson::fit(&y, &x, &groups).unwrap();
    assert!(res.converged);
    assert!(res.params.len() == 1);
    assert!(res.log_likelihood.is_finite());
}

// ====================== Display tests ======================

#[test]
fn test_poisson_display() {
    let y = Array1::from(vec![1.0, 3.0, 2.0, 5.0, 0.0, 4.0, 1.0, 2.0, 3.0, 6.0]);
    let x_raw = Array2::from_shape_vec(
        (10, 1),
        vec![0.1, 0.3, 0.2, 0.5, 0.0, 0.4, 0.1, 0.2, 0.3, 0.6],
    )
    .unwrap();
    let x = with_intercept(&x_raw);
    let res = Poisson::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let display = format!("{}", res);
    assert!(display.contains("Poisson Regression Results"));
}

#[test]
fn test_mnlogit_display() {
    let rng = lcg_sequence(300, 42);
    let n = 100;
    let x_vals: Vec<f64> = rng[..n].iter().map(|v| v * 4.0 - 2.0).collect();
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let v = x_vals[i] + (rng[n + i] - 0.5) * 2.0;
            if v < -0.5 {
                0.0
            } else if v < 0.5 {
                1.0
            } else {
                2.0
            }
        })
        .collect();

    let y = Array1::from(y_vals);
    let x_raw = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let x = with_intercept(&x_raw);
    let res = MNLogit::fit(&y, &x).unwrap();
    let display = format!("{}", res);
    assert!(display.contains("Multinomial Logit"));
}
