use greeners::{BinaryDiagnostics, Logit};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_logit_data(seed: u64, n: usize) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    let mut probs = Vec::with_capacity(n);

    for _ in 0..n {
        let x1 = noise.sample(&mut rng);
        let lp = -0.5 + 1.5 * x1;
        let p = 1.0 / (1.0 + f64::exp(-lp));
        let y_i = if rng.gen::<f64>() < p { 1.0 } else { 0.0 };
        x.push(1.0);
        x.push(x1);
        y.push(y_i);
        probs.push(p);
    }

    (
        Array1::from_vec(y),
        Array2::from_shape_vec((n, 2), x).unwrap(),
        Array1::from_vec(probs),
    )
}

/// Binary diagnostics return finite classification, ROC and Hosmer-Lemeshow results.
#[test]
fn test_binary_diagnostics_invariants() {
    let (y, _, probs) = make_logit_data(12345, 100);
    let y_slice: Vec<f64> = y.iter().copied().collect();
    let p_slice: Vec<f64> = probs.iter().copied().collect();

    let cls = BinaryDiagnostics::classification(&y_slice, &p_slice, 0.5).unwrap();
    assert_eq!(cls.n, 100);
    assert!(cls.sensitivity >= 0.0 && cls.sensitivity <= 1.0);
    assert!(cls.specificity >= 0.0 && cls.specificity <= 1.0);
    assert!(cls.correct_rate >= 0.0 && cls.correct_rate <= 1.0);
    assert_eq!(cls.tp + cls.tn + cls.fp + cls.fn_count, 100);

    let roc = BinaryDiagnostics::roc(&y_slice, &p_slice).unwrap();
    assert!(roc.auc >= 0.0 && roc.auc <= 1.0);
    assert!(roc.gini >= -1.0 && roc.gini <= 1.0);
    assert_eq!(roc.fpr.len(), roc.tpr.len());
    assert!(!roc.fpr.is_empty());

    let hl = BinaryDiagnostics::hosmer_lemeshow(&y_slice, &p_slice, 10).unwrap();
    assert!(hl.hl_stat.is_finite());
    assert!(hl.p_value >= 0.0 && hl.p_value <= 1.0);
    assert_eq!(hl.n_groups, 10);
}

/// Linktest returns finite coefficients and specification diagnostics.
#[test]
fn test_binary_diagnostics_linktest_invariants() {
    let (y, x, _) = make_logit_data(23456, 100);
    let logit = Logit::fit(&y, &x).unwrap();
    let result = BinaryDiagnostics::linktest(&y, &x, &logit.params).unwrap();

    assert_eq!(result.n, 100);
    assert!(result.hat_coef.is_finite());
    assert!(result.hat_p >= 0.0 && result.hat_p <= 1.0);
    assert!(result.hatsq_coef.is_finite());
    assert!(result.hatsq_p >= 0.0 && result.hatsq_p <= 1.0);
}

/// Input validation rejects mismatched lengths and degenerate ROC/H-L inputs.
#[test]
fn test_binary_diagnostics_input_validation() {
    let (y, _, probs) = make_logit_data(11111, 100);
    let y_slice: Vec<f64> = y.iter().copied().collect();
    let mut p_slice: Vec<f64> = probs.iter().copied().collect();
    p_slice.pop();

    assert!(BinaryDiagnostics::classification(&y_slice, &p_slice, 0.5).is_err());
    assert!(BinaryDiagnostics::roc(&y_slice, &p_slice).is_err());
    assert!(BinaryDiagnostics::hosmer_lemeshow(&y_slice, &p_slice, 10).is_err());

    let all_zero = vec![0.0; 100];
    assert!(BinaryDiagnostics::roc(&all_zero, &p_slice).is_err());

    let all_one = vec![1.0; 100];
    assert!(BinaryDiagnostics::roc(&all_one, &p_slice).is_err());

    assert!(BinaryDiagnostics::hosmer_lemeshow(&y_slice, &y_slice, 200).is_err());
}
