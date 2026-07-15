//! Diagnostics for binary choice models (logit/probit):
//! classification table, ROC/AUC, Hosmer-Lemeshow goodness-of-fit.

use crate::error::GreenersError;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of the classification table for binary models.
#[derive(Debug)]
pub struct ClassificationResult {
    /// Threshold used (default 0.5)
    pub threshold: f64,
    /// True positives
    pub tp: usize,
    /// True negatives
    pub tn: usize,
    /// False positives
    pub fp: usize,
    /// False negatives
    pub fn_count: usize,
    /// Sensitivity (recall): TP / (TP + FN)
    pub sensitivity: f64,
    /// Specificity: TN / (TN + FP)
    pub specificity: f64,
    /// Overall correctness rate: (TP + TN) / N
    pub correct_rate: f64,
    /// Total observations
    pub n: usize,
}

impl std::fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n{:=^60}", " Classification Table ")?;
        writeln!(f, "Threshold: {:.2}", self.threshold)?;
        writeln!(f, "{:-^60}", "")?;
        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>10}",
            "", "Pred=0", "Pred=1", "Total"
        )?;
        writeln!(f, "{:-^60}", "")?;
        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>10}",
            "Actual=0", self.tn, self.fp, self.tn + self.fp
        )?;
        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>10}",
            "Actual=1", self.fn_count, self.tp, self.fn_count + self.tp
        )?;
        writeln!(f, "{:-^60}", "")?;
        writeln!(
            f,
            "{:<20} {:>10.4}", "Sensitivity:", self.sensitivity
        )?;
        writeln!(
            f,
            "{:<20} {:>10.4}", "Specificity:", self.specificity
        )?;
        writeln!(
            f,
            "{:<20} {:>10.4}", "Correct rate:", self.correct_rate
        )?;
        write!(f, "{:=^60}", "")
    }
}

/// Result of the ROC / AUC analysis.
#[derive(Debug)]
pub struct RocResult {
    /// Area under the ROC curve
    pub auc: f64,
    /// Number of thresholds evaluated
    pub n_thresholds: usize,
    /// Gini coefficient: 2 * AUC - 1
    pub gini: f64,
}

impl std::fmt::Display for RocResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n{:=^60}", " ROC / AUC ")?;
        writeln!(f, "{:-^60}", "")?;
        writeln!(f, "{:<24} {:>12.4}", "AUC:", self.auc)?;
        writeln!(f, "{:<24} {:>12.4}", "Gini (2*AUC-1):", self.gini)?;
        writeln!(f, "{:<24} {:>12}", "thresholds:", self.n_thresholds)?;
        let interpretation = if self.auc >= 0.9 {
            "Excellent discrimination"
        } else if self.auc >= 0.8 {
            "Good discrimination"
        } else if self.auc >= 0.7 {
            "Acceptable discrimination"
        } else if self.auc >= 0.6 {
            "Poor discrimination"
        } else {
            "No discrimination (random or worse)"
        };
        writeln!(f, "{:-^60}", "")?;
        writeln!(f, "Interpretation: {interpretation}")?;
        write!(f, "{:=^60}", "")
    }
}

/// Result of the Hosmer-Lemeshow goodness-of-fit test.
#[derive(Debug)]
pub struct HosmerLemeshowResult {
    /// H-L chi-squared statistic
    pub hl_stat: f64,
    /// p-value from chi²(g - 2)
    pub p_value: f64,
    /// Number of groups (deciles by default)
    pub n_groups: usize,
    /// Degrees of freedom: g - 2
    pub df: usize,
}

impl std::fmt::Display for HosmerLemeshowResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n{:=^60}", " Hosmer-Lemeshow Goodness-of-Fit ")?;
        writeln!(f, "H0: model fits the data adequately")?;
        writeln!(f, "{:-^60}", "")?;
        writeln!(f, "{:<24} {:>12.4}", "H-L statistic:", self.hl_stat)?;
        writeln!(f, "{:<24} {:>12}", "groups (g):", self.n_groups)?;
        writeln!(f, "{:<24} {:>12}", "df (g-2):", self.df)?;
        writeln!(f, "{:<24} {:>12.4}", "p-value:", self.p_value)?;
        let verdict = if self.p_value < 0.05 {
            "Reject H0 — model does NOT fit adequately"
        } else {
            "Fail to reject H0 — model fits adequately"
        };
        writeln!(f, "{:-^60}", "")?;
        writeln!(f, "Conclusion: {verdict}")?;
        write!(f, "{:=^60}", "")
    }
}

/// Diagnostics for binary choice models.
pub struct BinaryDiagnostics;

impl BinaryDiagnostics {
    /// Classification table for binary models.
    ///
    /// Compares predicted probabilities (using `threshold`) with actual
    /// outcomes to compute sensitivity, specificity, and correct rate.
    ///
    /// # Arguments
    /// * `y` - Actual binary outcomes (0 or 1), length n
    /// * `probs` - Predicted probabilities Pr(y=1|x), length n
    /// * `threshold` - Classification threshold (default 0.5)
    pub fn classification(
        y: &[f64],
        probs: &[f64],
        threshold: f64,
    ) -> Result<ClassificationResult, GreenersError> {
        if y.len() != probs.len() {
            return Err(GreenersError::ShapeMismatch(format!(
                "y (len={}) and probs (len={}) must have same length",
                y.len(),
                probs.len()
            )));
        }

        let mut tp = 0usize;
        let mut tn = 0usize;
        let mut fp = 0usize;
        let mut fn_count = 0usize;

        for (yi, pi) in y.iter().zip(probs.iter()) {
            let pred = if *pi >= threshold { 1.0 } else { 0.0 };
            if pred == 1.0 && *yi == 1.0 {
                tp += 1;
            } else if pred == 0.0 && *yi == 0.0 {
                tn += 1;
            } else if pred == 1.0 && *yi == 0.0 {
                fp += 1;
            } else {
                fn_count += 1;
            }
        }

        let n = y.len();
        let sensitivity = if (tp + fn_count) > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        let specificity = if (tn + fp) > 0 {
            tn as f64 / (tn + fp) as f64
        } else {
            0.0
        };
        let correct_rate = if n > 0 {
            (tp + tn) as f64 / n as f64
        } else {
            0.0
        };

        Ok(ClassificationResult {
            threshold,
            tp,
            tn,
            fp,
            fn_count,
            sensitivity,
            specificity,
            correct_rate,
            n,
        })
    }

    /// ROC curve and AUC (area under the curve).
    ///
    /// Computes the AUC using the rank-based approach (equivalent to
    /// the Wilcoxon-Mann-Whitney statistic):
    ///
    /// AUC = (Σ_i∈positive rank_i - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    ///
    /// This is exact and does not require threshold enumeration.
    ///
    /// # Arguments
    /// * `y` - Actual binary outcomes (0 or 1), length n
    /// * `probs` - Predicted probabilities Pr(y=1|x), length n
    pub fn roc(y: &[f64], probs: &[f64]) -> Result<RocResult, GreenersError> {
        if y.len() != probs.len() {
            return Err(GreenersError::ShapeMismatch(format!(
                "y (len={}) and probs (len={}) must have same length",
                y.len(),
                probs.len()
            )));
        }

        let n = y.len();
        let n_pos = y.iter().filter(|&&v| v == 1.0).count();
        let n_neg = n - n_pos;

        if n_pos == 0 || n_neg == 0 {
            return Err(GreenersError::InvalidOperation(
                "ROC: need at least one positive and one negative observation".into(),
            ));
        }

        // Rank probabilities (ascending), average ranks for ties
        let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign average ranks (1-based)
        let mut ranks = vec![0.0_f64; n];
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            // Average rank for positions i..j (1-based: i+1..j+1)
            let avg_rank = ((i + 1) + j) as f64 / 2.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }

        // Sum of ranks for positive class
        let sum_ranks_pos: f64 = y
            .iter()
            .zip(ranks.iter())
            .filter(|(yi, _)| **yi == 1.0)
            .map(|(_, &r)| r)
            .sum();

        let auc = (sum_ranks_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0)
            / (n_pos as f64 * n_neg as f64);
        let gini = 2.0 * auc - 1.0;

        // Number of unique thresholds
        let n_thresholds = probs
            .iter()
            .map(|p| p.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();

        Ok(RocResult {
            auc,
            gini,
            n_thresholds,
        })
    }

    /// Hosmer-Lemeshow goodness-of-fit test.
    ///
    /// Divides observations into `n_groups` groups (default 10, deciles)
    /// based on predicted probabilities. For each group, compares observed
    /// vs expected counts of positives using a chi-squared statistic:
    ///
    /// H = Σ_g (O_g - E_g)² / (E_g * (1 - E_g/n_g))
    ///
    /// distributed as chi²(g - 2).
    ///
    /// # Arguments
    /// * `y` - Actual binary outcomes (0 or 1), length n
    /// * `probs` - Predicted probabilities Pr(y=1|x), length n
    /// * `n_groups` - Number of groups (default 10)
    pub fn hosmer_lemeshow(
        y: &[f64],
        probs: &[f64],
        n_groups: usize,
    ) -> Result<HosmerLemeshowResult, GreenersError> {
        if y.len() != probs.len() {
            return Err(GreenersError::ShapeMismatch(format!(
                "y (len={}) and probs (len={}) must have same length",
                y.len(),
                probs.len()
            )));
        }

        let n = y.len();
        if n_groups < 3 {
            return Err(GreenersError::InvalidOperation(
                "hosmer_lemeshow: n_groups must be at least 3".into(),
            ));
        }
        if n < n_groups {
            return Err(GreenersError::InvalidOperation(format!(
                "hosmer_lemeshow: n ({n}) must be >= n_groups ({n_groups})"
            )));
        }

        // Sort by predicted probability
        let mut indexed: Vec<(f64, f64)> = y.iter().zip(probs.iter()).map(|(&yi, &pi)| (yi, pi)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Divide into groups of approximately equal size
        let group_size = n as f64 / n_groups as f64;
        let mut hl_stat = 0.0_f64;

        for g in 0..n_groups {
            let start = (g as f64 * group_size).round() as usize;
            let end = if g == n_groups - 1 {
                n
            } else {
                ((g + 1) as f64 * group_size).round() as usize
            };
            let n_g = end - start;
            if n_g == 0 {
                continue;
            }

            let mut observed = 0.0_f64;
            let mut expected = 0.0_f64;
            for (yi, pi) in indexed.iter().take(end).skip(start) {
                observed += yi;
                expected += pi;
            }

            // H-L contribution: (O - E)² / (E * (1 - E/n_g))
            // Guard against division by zero
            let p_bar = expected / n_g as f64;
            let denom = expected * (1.0 - p_bar);
            if denom.abs() > 1e-15 {
                hl_stat += (observed - expected).powi(2) / denom;
            }
        }

        let df = n_groups.saturating_sub(2);
        let p_value = if df > 0 {
            let chi2 =
                ChiSquared::new(df as f64).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
            1.0 - chi2.cdf(hl_stat)
        } else {
            f64::NAN
        };

        Ok(HosmerLemeshowResult {
            hl_stat,
            p_value,
            n_groups,
            df,
        })
    }
}
