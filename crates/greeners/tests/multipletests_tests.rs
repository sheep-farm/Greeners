use greeners::{MultiTestMethod, MultipleTests};

const PVALUES: [f64; 4] = [0.01, 0.04, 0.03, 0.005];
const ALPHA: f64 = 0.05;

#[test]
fn test_bonferroni() {
    let (reject, corrected) =
        MultipleTests::multipletests(&PVALUES, ALPHA, MultiTestMethod::Bonferroni).unwrap();

    // p_adj = p * 4
    assert!((corrected[0] - 0.04).abs() < 1e-10);
    assert!((corrected[1] - 0.16).abs() < 1e-10);
    assert!((corrected[2] - 0.12).abs() < 1e-10);
    assert!((corrected[3] - 0.02).abs() < 1e-10);

    // At alpha=0.05: 0.04 < 0.05, 0.16 >= 0.05, 0.12 >= 0.05, 0.02 < 0.05
    assert_eq!(reject, vec![true, false, false, true]);
}

#[test]
fn test_benjamini_hochberg() {
    let (reject, corrected) =
        MultipleTests::multipletests(&PVALUES, ALPHA, MultiTestMethod::BenjaminiHochberg).unwrap();

    // Sorted: 0.005(idx3,rank1), 0.01(idx0,rank2), 0.03(idx2,rank3), 0.04(idx1,rank4)
    // Raw adj: 0.005*4/1=0.02, 0.01*4/2=0.02, 0.03*4/3=0.04, 0.04*4/4=0.04
    // Enforce monotonicity from bottom: 0.04, 0.04, 0.02, 0.02
    // So: idx3=0.02, idx0=0.02, idx2=0.04, idx1=0.04
    assert!((corrected[3] - 0.02).abs() < 1e-10);
    assert!((corrected[0] - 0.02).abs() < 1e-10);
    assert!((corrected[2] - 0.04).abs() < 1e-10);
    assert!((corrected[1] - 0.04).abs() < 1e-10);

    // All corrected < 0.05
    assert_eq!(reject, vec![true, true, true, true]);
}

#[test]
fn test_holm() {
    let (reject, corrected) =
        MultipleTests::multipletests(&PVALUES, ALPHA, MultiTestMethod::HolmBonferroni).unwrap();

    // Sorted ascending: 0.005(idx3), 0.01(idx0), 0.03(idx2), 0.04(idx1)
    // adj: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
    // Enforce monotonicity (cummax): 0.02, 0.03, 0.06, 0.06
    // Map back: idx3=0.02, idx0=0.03, idx2=0.06, idx1=0.06
    assert!((corrected[3] - 0.02).abs() < 1e-10);
    assert!((corrected[0] - 0.03).abs() < 1e-10);
    assert!((corrected[2] - 0.06).abs() < 1e-10);
    assert!((corrected[1] - 0.06).abs() < 1e-10);

    assert_eq!(reject, vec![true, false, false, true]);
}

#[test]
fn test_sidak() {
    let (reject, corrected) =
        MultipleTests::multipletests(&PVALUES, ALPHA, MultiTestMethod::Sidak).unwrap();

    // p_adj = 1 - (1 - p)^4
    let expected_0 = 1.0 - (1.0 - 0.01_f64).powi(4);
    let expected_3 = 1.0 - (1.0 - 0.005_f64).powi(4);
    assert!((corrected[0] - expected_0).abs() < 1e-10);
    assert!((corrected[3] - expected_3).abs() < 1e-10);

    assert!(reject[0]); // ~0.0394 < 0.05
    assert!(reject[3]); // ~0.0199 < 0.05
}

#[test]
fn test_empty_pvalues() {
    let result = MultipleTests::multipletests(&[], ALPHA, MultiTestMethod::Bonferroni);
    assert!(result.is_err());
}
