use crate::error::GreenersError;
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fmt;

/// Weighted descriptive statistics.
#[derive(Debug)]
pub struct DescrStatsW {
    pub mean: f64,
    pub std: f64,
    pub var: f64,
    pub std_mean: f64,
    pub min: f64,
    pub max: f64,
    pub nobs: f64,
    pub sum_weights: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
}

impl DescrStatsW {
    /// Compute weighted descriptive statistics.
    /// If weights is None, equal weights are used.
    pub fn new(data: &Array1<f64>, weights: Option<&Array1<f64>>) -> Result<Self, GreenersError> {
        let n = data.len();
        if n == 0 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 1 observation".into(),
            ));
        }

        let w: Array1<f64> = match weights {
            Some(wt) => {
                if wt.len() != n {
                    return Err(GreenersError::ShapeMismatch(
                        "weights length mismatch".into(),
                    ));
                }
                wt.clone()
            }
            None => Array1::from_elem(n, 1.0),
        };

        let sum_w: f64 = w.iter().sum();
        if sum_w < 1e-15 {
            return Err(GreenersError::InvalidOperation(
                "Sum of weights must be positive".into(),
            ));
        }

        // Weighted mean
        let mean: f64 = (0..n).map(|i| w[i] * data[i]).sum::<f64>() / sum_w;

        // Weighted variance (reliability weights)
        let sum_w2: f64 = w.iter().map(|wi| wi * wi).sum();
        let var_num: f64 = (0..n).map(|i| w[i] * (data[i] - mean).powi(2)).sum();
        let var = var_num / (sum_w - sum_w2 / sum_w).max(1e-15);
        let std = var.sqrt();
        let std_mean = std / sum_w.sqrt();

        // Weighted skewness and kurtosis
        let m3: f64 = (0..n).map(|i| w[i] * (data[i] - mean).powi(3)).sum::<f64>() / sum_w;
        let m2: f64 = var_num / sum_w;
        let skewness = if m2 > 1e-15 { m3 / m2.powf(1.5) } else { 0.0 };

        let m4: f64 = (0..n).map(|i| w[i] * (data[i] - mean).powi(4)).sum::<f64>() / sum_w;
        let kurtosis = if m2 > 1e-15 { m4 / m2.powi(2) } else { 0.0 };

        // Quantiles (unweighted for simplicity — use sorted data)
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = sorted[0];
        let max = sorted[n - 1];
        let q25 = percentile(&sorted, 25.0);
        let median = percentile(&sorted, 50.0);
        let q75 = percentile(&sorted, 75.0);

        Ok(DescrStatsW {
            mean,
            std,
            var,
            std_mean,
            min,
            max,
            nobs: n as f64,
            sum_weights: sum_w,
            skewness,
            kurtosis,
            median,
            q25,
            q75,
        })
    }

    /// Weighted t-test for H0: mean = mu0.
    /// Returns (t_statistic, p_value).
    pub fn ttest_mean(&self, mu0: f64) -> Result<(f64, f64), GreenersError> {
        if self.std_mean < 1e-15 {
            return Ok((0.0, 1.0));
        }
        let t = (self.mean - mu0) / self.std_mean;
        let df = self.nobs - 1.0;
        if df < 1.0 {
            return Ok((t, 1.0));
        }
        let dist = StudentsT::new(0.0, 1.0, df).map_err(|_| GreenersError::OptimizationFailed)?;
        let p = 2.0 * (1.0 - dist.cdf(t.abs()));
        Ok((t, p))
    }

    /// Confidence interval for the mean.
    pub fn conf_int_mean(&self, alpha: f64) -> Result<(f64, f64), GreenersError> {
        let df = self.nobs - 1.0;
        if df < 1.0 {
            return Ok((f64::NEG_INFINITY, f64::INFINITY));
        }
        let dist = StudentsT::new(0.0, 1.0, df).map_err(|_| GreenersError::OptimizationFailed)?;
        let t_crit = dist.inverse_cdf(1.0 - alpha / 2.0);
        Ok((
            self.mean - t_crit * self.std_mean,
            self.mean + t_crit * self.std_mean,
        ))
    }
}

impl fmt::Display for DescrStatsW {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^50}", " Descriptive Statistics ")?;
        writeln!(f, "{:<20} {:>12.4}", "Mean:", self.mean)?;
        writeln!(f, "{:<20} {:>12.4}", "Std Dev:", self.std)?;
        writeln!(f, "{:<20} {:>12.4}", "Variance:", self.var)?;
        writeln!(f, "{:<20} {:>12.4}", "SE(Mean):", self.std_mean)?;
        writeln!(f, "{:<20} {:>12.4}", "Skewness:", self.skewness)?;
        writeln!(f, "{:<20} {:>12.4}", "Kurtosis:", self.kurtosis)?;
        writeln!(f, "{:<20} {:>12.4}", "Min:", self.min)?;
        writeln!(f, "{:<20} {:>12.4}", "Q25:", self.q25)?;
        writeln!(f, "{:<20} {:>12.4}", "Median:", self.median)?;
        writeln!(f, "{:<20} {:>12.4}", "Q75:", self.q75)?;
        writeln!(f, "{:<20} {:>12.4}", "Max:", self.max)?;
        writeln!(f, "{:<20} {:>12.0}", "N:", self.nobs)?;
        writeln!(f, "{:<20} {:>12.2}", "Sum weights:", self.sum_weights)?;
        writeln!(f, "{:=^50}", "")
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = (p / 100.0) * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil().min((n - 1) as f64) as usize;
    let w = idx - lower as f64;
    sorted[lower] * (1.0 - w) + sorted[upper] * w
}
