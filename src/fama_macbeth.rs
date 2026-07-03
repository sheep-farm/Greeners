use crate::error::GreenersError;
use crate::{CovarianceType, DataFrame, Formula, OLS};
use ndarray::Array1;

/// Result of Fama-MacBeth (1973) cross-sectional regression.
#[derive(Debug, Clone)]
pub struct FamaMacBethResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub variable_names: Vec<String>,
    pub n_periods: usize,
    pub n_obs_total: usize,
    pub nw_lags: usize,
}

/// Fama-MacBeth (1973) two-step procedure.
///
/// Step 1: Run cross-sectional OLS for each time period.
/// Step 2: Average coefficients across periods; SE = σ(β̂_t)/√T.
/// Optional Newey-West correction for autocorrelation in the β̂_t series.
pub struct FamaMacBeth;

impl FamaMacBeth {
    /// Estimate Fama-MacBeth regression.
    ///
    /// # Arguments
    /// * `formula` - Parsed formula (y ~ x1 + x2)
    /// * `df` - DataFrame with all observations
    /// * `time_col` - Column name identifying time periods
    /// * `nw_lags` - Newey-West lags (0 = no correction)
    pub fn fit(
        formula: &Formula,
        df: &DataFrame,
        time_col: &str,
        nw_lags: usize,
    ) -> Result<FamaMacBethResult, GreenersError> {
        let t_column = df.get_column(time_col)?;
        let t_vals = t_column.to_float();
        let mut periods: Vec<i64> = t_vals
            .iter()
            .map(|&v| v as i64)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        periods.sort();
        let n_periods = periods.len();

        if n_periods < 2 {
            return Err(GreenersError::ShapeMismatch(
                "Fama-MacBeth requires at least 2 time periods".into(),
            ));
        }

        let mut all_coefs: Vec<Vec<f64>> = Vec::new();
        let mut var_names: Vec<String> = Vec::new();
        let mut n_obs_total = 0usize;

        for &t in &periods {
            let mask: Vec<usize> = t_vals
                .iter()
                .enumerate()
                .filter(|(_, &v)| v as i64 == t)
                .map(|(i, _)| i)
                .collect();

            if mask.len() < 3 {
                continue;
            }

            let sub_df = df.iloc(Some(&mask), None)?;

            if let Ok(result) = OLS::from_formula(formula, &sub_df, CovarianceType::NonRobust) {
                if var_names.is_empty() {
                    var_names = result.variable_names.clone().unwrap_or_default();
                }
                all_coefs.push(result.params.to_vec());
                n_obs_total += sub_df.n_rows();
            }
        }

        let t_ok = all_coefs.len();
        if t_ok < 2 {
            return Err(GreenersError::OptimizationFailed);
        }

        let k = all_coefs[0].len();
        let t_f = t_ok as f64;

        // Mean coefficients
        let mut mean_coef = vec![0.0; k];
        for coefs in &all_coefs {
            for j in 0..k {
                mean_coef[j] += coefs[j];
            }
        }
        for item in mean_coef.iter_mut().take(k) {
            *item /= t_f;
        }

        // Fama-MacBeth variance (with optional Newey-West)
        let mut fm_se = vec![0.0; k];
        for j in 0..k {
            let var_j: f64 = all_coefs
                .iter()
                .map(|c| (c[j] - mean_coef[j]).powi(2))
                .sum::<f64>()
                / t_f;

            if nw_lags > 0 {
                let mut nw_var = var_j;
                for lag in 1..=nw_lags.min(t_ok - 1) {
                    let w = 1.0 - lag as f64 / (nw_lags as f64 + 1.0);
                    let gamma: f64 = (lag..t_ok)
                        .map(|t| {
                            (all_coefs[t][j] - mean_coef[j])
                                * (all_coefs[t - lag][j] - mean_coef[j])
                        })
                        .sum::<f64>()
                        / t_f;
                    nw_var += 2.0 * w * gamma;
                }
                fm_se[j] = (nw_var / t_f).max(0.0).sqrt();
            } else {
                fm_se[j] = (var_j / (t_f - 1.0)).sqrt();
            }
        }

        // t-values and p-values
        let df_t = (t_ok - 1) as f64;
        let mut t_values = vec![0.0; k];
        let mut p_values = vec![0.0; k];
        for j in 0..k {
            t_values[j] = if fm_se[j] > 1e-15 {
                mean_coef[j] / fm_se[j]
            } else {
                f64::NAN
            };
            p_values[j] = crate::t_pvalue_two(t_values[j], df_t);
        }

        Ok(FamaMacBethResult {
            params: Array1::from(mean_coef),
            std_errors: Array1::from(fm_se),
            t_values: Array1::from(t_values),
            p_values: Array1::from(p_values),
            variable_names: var_names,
            n_periods: t_ok,
            n_obs_total,
            nw_lags,
        })
    }
}

impl std::fmt::Display for FamaMacBethResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nw_label = if self.nw_lags > 0 {
            format!("  NW({})", self.nw_lags)
        } else {
            String::new()
        };
        let thick = "═".repeat(70);
        let thin = "─".repeat(70);
        writeln!(f, "\n{thick}")?;
        writeln!(f, "{:^70}", format!(" Fama-MacBeth (1973){nw_label} "))?;
        writeln!(f, "{thin}")?;
        writeln!(
            f,
            "  Periods: {}   N total: {}",
            self.n_periods, self.n_obs_total
        )?;
        writeln!(f, "{thin}")?;
        writeln!(
            f,
            "{:<18} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Coef", "FM-SE", "t", "p"
        )?;
        writeln!(f, "{thin}")?;
        for j in 0..self.params.len() {
            let name = self
                .variable_names
                .get(j)
                .map(|s| s.as_str())
                .unwrap_or("?");
            let sig = if self.p_values[j] < 0.01 {
                "***"
            } else if self.p_values[j] < 0.05 {
                "**"
            } else if self.p_values[j] < 0.10 {
                "*"
            } else {
                ""
            };
            writeln!(
                f,
                "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {sig}",
                name, self.params[j], self.std_errors[j], self.t_values[j], self.p_values[j]
            )?;
        }
        write!(f, "{thick}")?;
        Ok(())
    }
}
