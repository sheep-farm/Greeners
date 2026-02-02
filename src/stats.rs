use crate::error::GreenersError;
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, FisherSnedecor, Normal, StudentsT};
use std::fmt;

// ─── ANOVA ─────────────────────────────────────────────────────────────────────

/// One-way ANOVA result.
#[derive(Debug)]
pub struct AnovaResult {
    pub ss_between: f64,
    pub ss_within: f64,
    pub ss_total: f64,
    pub df_between: usize,
    pub df_within: usize,
    pub ms_between: f64,
    pub ms_within: f64,
    pub f_statistic: f64,
    pub p_value: f64,
    pub n_groups: usize,
    pub n_obs: usize,
}

impl fmt::Display for AnovaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^70}", " One-Way ANOVA ")?;
        writeln!(
            f,
            "{:<12} {:>10} {:>6} {:>12} {:>10} {:>10}",
            "Source", "SS", "df", "MS", "F", "P>F"
        )?;
        writeln!(f, "{:-^70}", "")?;
        writeln!(
            f,
            "{:<12} {:>10.4} {:>6} {:>12.4} {:>10.4} {:>10.4}",
            "Between",
            self.ss_between,
            self.df_between,
            self.ms_between,
            self.f_statistic,
            self.p_value
        )?;
        writeln!(
            f,
            "{:<12} {:>10.4} {:>6} {:>12.4}",
            "Within", self.ss_within, self.df_within, self.ms_within
        )?;
        writeln!(
            f,
            "{:<12} {:>10.4} {:>6}",
            "Total",
            self.ss_total,
            self.df_between + self.df_within
        )?;
        writeln!(f, "{:=^70}", "")
    }
}

/// ANOVA table for regression models.
#[derive(Debug)]
pub struct AnovaRegressionResult {
    pub ss_model: f64,
    pub ss_resid: f64,
    pub ss_total: f64,
    pub df_model: usize,
    pub df_resid: usize,
    pub ms_model: f64,
    pub ms_resid: f64,
    pub f_statistic: f64,
    pub p_value: f64,
}

impl fmt::Display for AnovaRegressionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^70}", " ANOVA (Regression) ")?;
        writeln!(
            f,
            "{:<12} {:>10} {:>6} {:>12} {:>10} {:>10}",
            "Source", "SS", "df", "MS", "F", "P>F"
        )?;
        writeln!(f, "{:-^70}", "")?;
        writeln!(
            f,
            "{:<12} {:>10.4} {:>6} {:>12.4} {:>10.4} {:>10.4}",
            "Model", self.ss_model, self.df_model, self.ms_model, self.f_statistic, self.p_value
        )?;
        writeln!(
            f,
            "{:<12} {:>10.4} {:>6} {:>12.4}",
            "Residual", self.ss_resid, self.df_resid, self.ms_resid
        )?;
        writeln!(
            f,
            "{:<12} {:>10.4} {:>6}",
            "Total",
            self.ss_total,
            self.df_model + self.df_resid
        )?;
        writeln!(f, "{:=^70}", "")
    }
}

/// Result of comparing two sample means.
#[derive(Debug)]
pub struct CompareMeansResult {
    pub mean1: f64,
    pub mean2: f64,
    pub diff: f64,
    pub t_statistic: f64,
    pub p_value: f64,
    pub df: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub cohens_d: f64,
    pub n1: usize,
    pub n2: usize,
}

impl fmt::Display for CompareMeansResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Compare Means (Welch) ")?;
        writeln!(f, "{:<20} {:>10.4} (n={})", "Mean 1:", self.mean1, self.n1)?;
        writeln!(f, "{:<20} {:>10.4} (n={})", "Mean 2:", self.mean2, self.n2)?;
        writeln!(f, "{:<20} {:>10.4}", "Difference:", self.diff)?;
        writeln!(f, "{:<20} {:>10.4}", "t-statistic:", self.t_statistic)?;
        writeln!(f, "{:<20} {:>10.4}", "P-value:", self.p_value)?;
        writeln!(f, "{:<20} {:>10.1}", "df:", self.df)?;
        writeln!(
            f,
            "{:<20} [{:.4}, {:.4}]",
            "95% CI:", self.ci_lower, self.ci_upper
        )?;
        writeln!(f, "{:<20} {:>10.4}", "Cohen's d:", self.cohens_d)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Statistical tests and utilities.
pub struct Stats;

impl Stats {
    /// One-way ANOVA.
    pub fn anova_oneway(
        data: &Array1<f64>,
        groups: &Array1<usize>,
    ) -> Result<AnovaResult, GreenersError> {
        let n = data.len();
        if n != groups.len() {
            return Err(GreenersError::ShapeMismatch(
                "data and groups length mismatch".into(),
            ));
        }

        let mut unique: Vec<usize> = groups.iter().cloned().collect();
        unique.sort();
        unique.dedup();
        let g = unique.len();

        if g < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 groups".into(),
            ));
        }

        let grand_mean = data.mean().unwrap_or(0.0);

        let mut ss_between = 0.0;
        let mut ss_within = 0.0;

        for &grp in &unique {
            let vals: Vec<f64> = (0..n)
                .filter(|&i| groups[i] == grp)
                .map(|i| data[i])
                .collect();
            let ni = vals.len() as f64;
            let group_mean = vals.iter().sum::<f64>() / ni;

            ss_between += ni * (group_mean - grand_mean).powi(2);
            ss_within += vals.iter().map(|&x| (x - group_mean).powi(2)).sum::<f64>();
        }

        let ss_total = ss_between + ss_within;
        let df_between = g - 1;
        let df_within = n - g;
        let ms_between = ss_between / df_between as f64;
        let ms_within = ss_within / df_within.max(1) as f64;
        let f_stat = ms_between / ms_within.max(1e-15);

        let p_value = match FisherSnedecor::new(df_between as f64, df_within as f64) {
            Ok(dist) => 1.0 - dist.cdf(f_stat),
            Err(_) => 1.0,
        };

        Ok(AnovaResult {
            ss_between,
            ss_within,
            ss_total,
            df_between,
            df_within,
            ms_between,
            ms_within,
            f_statistic: f_stat,
            p_value,
            n_groups: g,
            n_obs: n,
        })
    }

    /// ANOVA table for a regression model.
    pub fn anova_regression(
        y: &Array1<f64>,
        residuals: &Array1<f64>,
        df_model: usize,
    ) -> Result<AnovaRegressionResult, GreenersError> {
        let n = y.len();
        let y_mean = y.mean().unwrap_or(0.0);
        let ss_total: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_resid: f64 = residuals.iter().map(|r| r * r).sum();
        let ss_model = ss_total - ss_resid;
        let df_resid = n.saturating_sub(df_model + 1);
        let ms_model = ss_model / df_model.max(1) as f64;
        let ms_resid = ss_resid / df_resid.max(1) as f64;
        let f_stat = ms_model / ms_resid.max(1e-15);

        let p_value = match FisherSnedecor::new(df_model as f64, df_resid as f64) {
            Ok(dist) => 1.0 - dist.cdf(f_stat),
            Err(_) => 1.0,
        };

        Ok(AnovaRegressionResult {
            ss_model,
            ss_resid,
            ss_total,
            df_model,
            df_resid,
            ms_model,
            ms_resid,
            f_statistic: f_stat,
            p_value,
        })
    }

    // ─── Proportion Tests ──────────────────────────────────────────────────

    /// One-sample proportion z-test.
    /// Tests H0: p = p0.
    /// Returns (z_statistic, p_value).
    pub fn proportion_ztest(
        count: usize,
        nobs: usize,
        p0: f64,
    ) -> Result<(f64, f64), GreenersError> {
        if nobs == 0 {
            return Err(GreenersError::InvalidOperation("nobs must be > 0".into()));
        }
        let p_hat = count as f64 / nobs as f64;
        let se = (p0 * (1.0 - p0) / nobs as f64).sqrt();
        if se < 1e-15 {
            return Ok((0.0, 1.0));
        }
        let z = (p_hat - p0) / se;
        let normal = Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));
        Ok((z, p_value))
    }

    /// Two-sample proportion z-test.
    /// Returns (z_statistic, p_value).
    pub fn proportion_ztest_2samp(
        count1: usize,
        nobs1: usize,
        count2: usize,
        nobs2: usize,
    ) -> Result<(f64, f64), GreenersError> {
        if nobs1 == 0 || nobs2 == 0 {
            return Err(GreenersError::InvalidOperation("nobs must be > 0".into()));
        }
        let p1 = count1 as f64 / nobs1 as f64;
        let p2 = count2 as f64 / nobs2 as f64;
        let p_pool = (count1 + count2) as f64 / (nobs1 + nobs2) as f64;
        let se = (p_pool * (1.0 - p_pool) * (1.0 / nobs1 as f64 + 1.0 / nobs2 as f64)).sqrt();
        if se < 1e-15 {
            return Ok((0.0, 1.0));
        }
        let z = (p1 - p2) / se;
        let normal = Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));
        Ok((z, p_value))
    }

    // ─── Multiple Testing Corrections ──────────────────────────────────────

    /// Bonferroni correction for multiple testing.
    /// Returns adjusted p-values.
    pub fn bonferroni(p_values: &[f64]) -> Vec<f64> {
        let m = p_values.len() as f64;
        p_values.iter().map(|&p| (p * m).min(1.0)).collect()
    }

    /// Benjamini-Hochberg (FDR) correction.
    /// Returns adjusted p-values.
    pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
        let m = p_values.len();
        let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut adjusted = vec![0.0; m];
        let mut cum_min: f64 = 1.0;

        for i in (0..m).rev() {
            let rank = i + 1;
            let adj = (indexed[i].1 * m as f64 / rank as f64).min(1.0);
            cum_min = cum_min.min(adj);
            adjusted[indexed[i].0] = cum_min;
        }

        adjusted
    }

    /// Holm-Bonferroni (step-down) correction.
    /// Returns adjusted p-values.
    pub fn holm(p_values: &[f64]) -> Vec<f64> {
        let m = p_values.len();
        let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut adjusted = vec![0.0; m];
        let mut cum_max: f64 = 0.0;

        for (i, &(orig_idx, p)) in indexed.iter().enumerate() {
            let adj = (p * (m - i) as f64).min(1.0);
            cum_max = cum_max.max(adj);
            adjusted[orig_idx] = cum_max;
        }

        adjusted
    }

    // ─── t-tests ───────────────────────────────────────────────────────────

    /// One-sample t-test.
    /// Tests H0: mean = mu0.
    /// Returns (t_statistic, p_value).
    pub fn ttest_1samp(data: &Array1<f64>, mu0: f64) -> Result<(f64, f64), GreenersError> {
        let n = data.len();
        if n < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 observations".into(),
            ));
        }
        let mean = data.mean().unwrap_or(0.0);
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let se = (var / n as f64).sqrt();
        if se < 1e-15 {
            return Ok((0.0, 1.0));
        }
        let t = (mean - mu0) / se;
        let df = (n - 1) as f64;
        let dist = StudentsT::new(0.0, 1.0, df).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 2.0 * (1.0 - dist.cdf(t.abs()));
        Ok((t, p_value))
    }

    /// Two-sample t-test (Welch's, unequal variances).
    /// Returns (t_statistic, p_value).
    pub fn ttest_ind(
        data1: &Array1<f64>,
        data2: &Array1<f64>,
    ) -> Result<(f64, f64), GreenersError> {
        let n1 = data1.len();
        let n2 = data2.len();
        if n1 < 2 || n2 < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 observations per group".into(),
            ));
        }

        let m1 = data1.mean().unwrap_or(0.0);
        let m2 = data2.mean().unwrap_or(0.0);
        let v1 = data1.iter().map(|&x| (x - m1).powi(2)).sum::<f64>() / (n1 - 1) as f64;
        let v2 = data2.iter().map(|&x| (x - m2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

        let se = (v1 / n1 as f64 + v2 / n2 as f64).sqrt();
        if se < 1e-15 {
            return Ok((0.0, 1.0));
        }

        let t = (m1 - m2) / se;

        // Welch-Satterthwaite df
        let num = (v1 / n1 as f64 + v2 / n2 as f64).powi(2);
        let den =
            (v1 / n1 as f64).powi(2) / (n1 - 1) as f64 + (v2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let df = num / den.max(1e-15);

        let dist = StudentsT::new(0.0, 1.0, df).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 2.0 * (1.0 - dist.cdf(t.abs()));
        Ok((t, p_value))
    }

    /// Compare means of two samples.
    ///
    /// Returns Welch t-test, confidence interval for the difference,
    /// and Cohen's d effect size.
    pub fn compare_means(
        data1: &Array1<f64>,
        data2: &Array1<f64>,
    ) -> Result<CompareMeansResult, GreenersError> {
        let n1 = data1.len();
        let n2 = data2.len();
        if n1 < 2 || n2 < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 observations per group".into(),
            ));
        }

        let m1 = data1.mean().unwrap_or(0.0);
        let m2 = data2.mean().unwrap_or(0.0);
        let v1 = data1.iter().map(|&x| (x - m1).powi(2)).sum::<f64>() / (n1 - 1) as f64;
        let v2 = data2.iter().map(|&x| (x - m2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

        let se = (v1 / n1 as f64 + v2 / n2 as f64).sqrt();
        let diff = m1 - m2;

        let t_stat = if se > 1e-15 { diff / se } else { 0.0 };

        // Welch-Satterthwaite df
        let num = (v1 / n1 as f64 + v2 / n2 as f64).powi(2);
        let den = (v1 / n1 as f64).powi(2) / (n1 - 1) as f64
            + (v2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let df = num / den.max(1e-15);

        let dist =
            StudentsT::new(0.0, 1.0, df).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 2.0 * (1.0 - dist.cdf(t_stat.abs()));

        // 95% CI for difference
        let t_crit = dist.inverse_cdf(0.975);
        let ci_lower = diff - t_crit * se;
        let ci_upper = diff + t_crit * se;

        // Cohen's d: pooled std
        let pooled_var = ((n1 - 1) as f64 * v1 + (n2 - 1) as f64 * v2)
            / (n1 + n2 - 2) as f64;
        let cohens_d = if pooled_var > 1e-15 {
            diff / pooled_var.sqrt()
        } else {
            0.0
        };

        Ok(CompareMeansResult {
            mean1: m1,
            mean2: m2,
            diff,
            t_statistic: t_stat,
            p_value,
            df,
            ci_lower,
            ci_upper,
            cohens_d,
            n1,
            n2,
        })
    }

    /// Paired t-test.
    /// Returns (t_statistic, p_value).
    pub fn ttest_paired(
        data1: &Array1<f64>,
        data2: &Array1<f64>,
    ) -> Result<(f64, f64), GreenersError> {
        if data1.len() != data2.len() {
            return Err(GreenersError::ShapeMismatch(
                "data1 and data2 must have same length".into(),
            ));
        }
        let diff = data1 - data2;
        Self::ttest_1samp(&diff, 0.0)
    }
}
