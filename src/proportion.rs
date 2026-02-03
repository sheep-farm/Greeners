use crate::GreenersError;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal as NormalDist};

/// Proportion-based statistical tests.
///
/// Provides methods analogous to `statsmodels.stats.proportion`.
pub struct ProportionTests;

impl ProportionTests {
    /// One-sample proportion z-test.
    ///
    /// Tests H0: p = `value` against H1: p != `value` (two-sided).
    ///
    /// Returns `(z_stat, p_value)`.
    pub fn proportions_ztest_1samp(
        count: usize,
        nobs: usize,
        value: f64,
    ) -> Result<(f64, f64), GreenersError> {
        if nobs == 0 {
            return Err(GreenersError::InvalidOperation(
                "Number of observations must be > 0".to_string(),
            ));
        }
        if count > nobs {
            return Err(GreenersError::InvalidOperation(
                "Count cannot exceed number of observations".to_string(),
            ));
        }
        if value <= 0.0 || value >= 1.0 {
            return Err(GreenersError::InvalidOperation(
                "Null proportion must be in (0, 1)".to_string(),
            ));
        }

        let p_hat = count as f64 / nobs as f64;
        let se = (value * (1.0 - value) / nobs as f64).sqrt();
        let z = (p_hat - value) / se;

        let normal = NormalDist::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));

        Ok((z, p_value))
    }

    /// Two-sample proportion z-test.
    ///
    /// Tests H0: p1 = p2 against H1: p1 != p2 (two-sided).
    ///
    /// Returns `(z_stat, p_value)`.
    pub fn proportions_ztest_2samp(
        count1: usize,
        nobs1: usize,
        count2: usize,
        nobs2: usize,
    ) -> Result<(f64, f64), GreenersError> {
        if nobs1 == 0 || nobs2 == 0 {
            return Err(GreenersError::InvalidOperation(
                "Number of observations must be > 0 for both samples".to_string(),
            ));
        }
        if count1 > nobs1 || count2 > nobs2 {
            return Err(GreenersError::InvalidOperation(
                "Count cannot exceed number of observations".to_string(),
            ));
        }

        let p1 = count1 as f64 / nobs1 as f64;
        let p2 = count2 as f64 / nobs2 as f64;
        let p_pool = (count1 + count2) as f64 / (nobs1 + nobs2) as f64;
        let se = (p_pool * (1.0 - p_pool) * (1.0 / nobs1 as f64 + 1.0 / nobs2 as f64)).sqrt();
        let z = (p1 - p2) / se;

        let normal = NormalDist::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));

        Ok((z, p_value))
    }

    /// Confidence interval for a proportion using the Wilson score interval.
    ///
    /// Returns `(lower, upper)`.
    pub fn proportion_confint(
        count: usize,
        nobs: usize,
        alpha: f64,
    ) -> Result<(f64, f64), GreenersError> {
        if nobs == 0 {
            return Err(GreenersError::InvalidOperation(
                "Number of observations must be > 0".to_string(),
            ));
        }
        if count > nobs {
            return Err(GreenersError::InvalidOperation(
                "Count cannot exceed number of observations".to_string(),
            ));
        }
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(GreenersError::InvalidOperation(
                "alpha must be in (0, 1)".to_string(),
            ));
        }

        let normal = NormalDist::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let z = normal.inverse_cdf(1.0 - alpha / 2.0);

        let n = nobs as f64;
        let p_hat = count as f64 / n;

        let denom = 1.0 + z * z / n;
        let center = (p_hat + z * z / (2.0 * n)) / denom;
        let margin = (z / denom) * ((p_hat * (1.0 - p_hat) / n) + (z * z / (4.0 * n * n))).sqrt();

        let lower = (center - margin).max(0.0);
        let upper = (center + margin).min(1.0);

        Ok((lower, upper))
    }

    /// Chi-square test of independence for a 2x2 contingency table.
    ///
    /// Returns `(chi2_stat, p_value)`.
    pub fn chi2_contingency(table: &[[usize; 2]; 2]) -> Result<(f64, f64), GreenersError> {
        let a = table[0][0] as f64;
        let b = table[0][1] as f64;
        let c = table[1][0] as f64;
        let d = table[1][1] as f64;
        let n = a + b + c + d;

        if n == 0.0 {
            return Err(GreenersError::InvalidOperation(
                "Contingency table is all zeros".to_string(),
            ));
        }

        let row_totals = [a + b, c + d];
        let col_totals = [a + c, b + d];

        let mut chi2 = 0.0;
        let observed = [[a, b], [c, d]];
        for i in 0..2 {
            for j in 0..2 {
                let expected = row_totals[i] * col_totals[j] / n;
                if expected == 0.0 {
                    return Err(GreenersError::InvalidOperation(
                        "Expected frequency is zero; test is not valid".to_string(),
                    ));
                }
                let diff = observed[i][j] - expected;
                chi2 += diff * diff / expected;
            }
        }

        let chi2_dist = ChiSquared::new(1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - chi2_dist.cdf(chi2);

        Ok((chi2, p_value))
    }
}
