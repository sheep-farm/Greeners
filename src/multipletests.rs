use crate::GreenersError;
use std::fmt;

/// Method for multiple testing correction.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiTestMethod {
    /// Bonferroni: p_adj = min(p * n, 1.0)
    Bonferroni,
    /// Sidak: p_adj = 1 - (1 - p)^n
    Sidak,
    /// Holm step-down (Holm-Bonferroni)
    HolmBonferroni,
    /// Benjamini-Hochberg FDR
    BenjaminiHochberg,
    /// Benjamini-Yekutieli FDR under dependence
    BenjaminiYekutieli,
}

impl fmt::Display for MultiTestMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiTestMethod::Bonferroni => write!(f, "Bonferroni"),
            MultiTestMethod::Sidak => write!(f, "Sidak"),
            MultiTestMethod::HolmBonferroni => write!(f, "Holm-Bonferroni"),
            MultiTestMethod::BenjaminiHochberg => write!(f, "Benjamini-Hochberg"),
            MultiTestMethod::BenjaminiYekutieli => write!(f, "Benjamini-Yekutieli"),
        }
    }
}

/// Multiple testing corrections for p-values.
///
/// Provides methods analogous to `statsmodels.stats.multitest.multipletests`.
pub struct MultipleTests;

impl MultipleTests {
    /// Apply multiple testing correction to a slice of p-values.
    ///
    /// Returns `(reject, corrected_pvalues)` where `reject[i]` is true if the
    /// corrected p-value is below `alpha`.
    pub fn multipletests(
        pvalues: &[f64],
        alpha: f64,
        method: MultiTestMethod,
    ) -> Result<(Vec<bool>, Vec<f64>), GreenersError> {
        if pvalues.is_empty() {
            return Err(GreenersError::InvalidOperation(
                "p-values slice is empty".to_string(),
            ));
        }
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(GreenersError::InvalidOperation(
                "alpha must be between 0 and 1 (exclusive)".to_string(),
            ));
        }
        for (i, &p) in pvalues.iter().enumerate() {
            if !(0.0..=1.0).contains(&p) {
                return Err(GreenersError::InvalidOperation(format!(
                    "p-value at index {} is {}, must be in [0, 1]",
                    i, p
                )));
            }
        }

        let corrected = match method {
            MultiTestMethod::Bonferroni => Self::bonferroni(pvalues),
            MultiTestMethod::Sidak => Self::sidak(pvalues),
            MultiTestMethod::HolmBonferroni => Self::holm(pvalues),
            MultiTestMethod::BenjaminiHochberg => Self::benjamini_hochberg(pvalues),
            MultiTestMethod::BenjaminiYekutieli => Self::benjamini_yekutieli(pvalues),
        };

        let reject = corrected.iter().map(|&p| p < alpha).collect();
        Ok((reject, corrected))
    }

    fn bonferroni(pvalues: &[f64]) -> Vec<f64> {
        let n = pvalues.len() as f64;
        pvalues.iter().map(|&p| (p * n).min(1.0)).collect()
    }

    fn sidak(pvalues: &[f64]) -> Vec<f64> {
        let n = pvalues.len() as f64;
        pvalues
            .iter()
            .map(|&p| (1.0 - (1.0 - p).powf(n)).min(1.0))
            .collect()
    }

    fn holm(pvalues: &[f64]) -> Vec<f64> {
        let n = pvalues.len();
        // Create index-sorted order (ascending by p-value)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| pvalues[a].partial_cmp(&pvalues[b]).unwrap());

        let mut corrected = vec![0.0_f64; n];

        // Forward pass: p_adj[i] = p[order[i]] * (n - i), enforce monotonicity upward
        let mut cummax = 0.0_f64;
        for (i, &idx) in order.iter().enumerate() {
            let adj = (pvalues[idx] * (n - i) as f64).min(1.0);
            cummax = cummax.max(adj);
            corrected[idx] = cummax;
        }

        corrected
    }

    fn benjamini_hochberg(pvalues: &[f64]) -> Vec<f64> {
        let n = pvalues.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| pvalues[a].partial_cmp(&pvalues[b]).unwrap());

        let mut corrected = vec![0.0_f64; n];

        // Backward pass from largest to smallest: enforce monotonicity downward
        let mut cummin = 1.0_f64;
        for i in (0..n).rev() {
            let idx = order[i];
            let rank = (i + 1) as f64;
            let adj = (pvalues[idx] * n as f64 / rank).min(1.0);
            cummin = cummin.min(adj);
            corrected[idx] = cummin;
        }

        corrected
    }

    fn benjamini_yekutieli(pvalues: &[f64]) -> Vec<f64> {
        let n = pvalues.len();
        let c_n: f64 = (1..=n).map(|k| 1.0 / k as f64).sum();

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| pvalues[a].partial_cmp(&pvalues[b]).unwrap());

        let mut corrected = vec![0.0_f64; n];

        let mut cummin = 1.0_f64;
        for i in (0..n).rev() {
            let idx = order[i];
            let rank = (i + 1) as f64;
            let adj = (pvalues[idx] * n as f64 * c_n / rank).min(1.0);
            cummin = cummin.min(adj);
            corrected[idx] = cummin;
        }

        corrected
    }
}
