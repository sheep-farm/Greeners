use greeners_core::multipletests::MultiTestMethod;
use greeners_core::multipletests::MultipleTests;

const PVALUES: [f64; 5] = [0.01, 0.04, 0.03, 0.005, 0.02];
const ALPHA: f64 = 0.05;

/// Multiple tests methods return corrected p-values and rejection flags of the correct length.
#[test]
fn test_multipletests_methods_invariants() {
    for method in [
        MultiTestMethod::Bonferroni,
        MultiTestMethod::Sidak,
        MultiTestMethod::HolmBonferroni,
        MultiTestMethod::BenjaminiHochberg,
        MultiTestMethod::BenjaminiYekutieli,
    ] {
        let (reject, corrected) = MultipleTests::multipletests(&PVALUES, ALPHA, method).unwrap();

        assert_eq!(reject.len(), PVALUES.len());
        assert_eq!(corrected.len(), PVALUES.len());
        assert!(corrected.iter().all(|&p| p >= 0.0 && p <= 1.0));

        // Reject flag is consistent with the corrected p-value and alpha
        for (&r, &p) in reject.iter().zip(corrected.iter()) {
            assert_eq!(r, p < ALPHA);
        }
    }
}

/// Corrected p-values are monotonically non-decreasing across the methods (or at least bounded).
#[test]
fn test_multipletests_ordering_invariants() {
    let (_, bonferroni) =
        MultipleTests::multipletests(&PVALUES, ALPHA, MultiTestMethod::Bonferroni).unwrap();
    let (_, bh) =
        MultipleTests::multipletests(&PVALUES, ALPHA, MultiTestMethod::BenjaminiHochberg).unwrap();

    // Bonferroni is the most conservative: its adjusted values are >= BH adjusted values
    for j in 0..PVALUES.len() {
        assert!(bonferroni[j] >= bh[j]);
    }

    // Sidak and Bonferroni are identical to first order for these small p-values
    let (_, sidak) = MultipleTests::multipletests(&PVALUES, ALPHA, MultiTestMethod::Sidak).unwrap();
    for j in 0..PVALUES.len() {
        assert!(sidak[j] >= PVALUES[j]);
        assert!(sidak[j] <= 1.0);
    }
}

/// Input validation rejects empty slices, out-of-range p-values and invalid alpha.
#[test]
fn test_multipletests_input_validation() {
    assert!(MultipleTests::multipletests(&[], ALPHA, MultiTestMethod::Bonferroni).is_err());

    let bad_pvalues = [-0.01, 0.5, 1.5];
    assert!(
        MultipleTests::multipletests(&bad_pvalues, ALPHA, MultiTestMethod::Bonferroni).is_err()
    );

    assert!(MultipleTests::multipletests(&PVALUES, 0.0, MultiTestMethod::Bonferroni).is_err());
    assert!(MultipleTests::multipletests(&PVALUES, 1.0, MultiTestMethod::Bonferroni).is_err());
}
