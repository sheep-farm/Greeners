//! Panel quantile regression with fixed effects.
//!
//! Quantile regression with individual fixed effects, estimated via
//! the within transformation (demeaning by entity) followed by
//! standard quantile regression (Koenker & Bassett 1978) on the
//! demeaned data.
//!
//! For quantile τ, minimize Σ ρ_τ(y_it - x_it'β - α_i)
//! where ρ_τ(u) = u(τ - I(u < 0)).
//!
//! The fixed effects are removed by within transformation, then
//! quantile regression is applied to the demeaned data.

use crate::error::GreenersError;
use crate::quantile::QuantileReg;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of panel quantile regression.
#[derive(Debug)]
pub struct PanelQuantileResult {
    /// Quantile level (0-1)
    pub tau: f64,
    /// Coefficients (excluding FE)
    pub beta: Array1<f64>,
    /// Standard errors (bootstrap)
    pub std_errors: Array1<f64>,
    /// t-statistics
    pub t_values: Array1<f64>,
    /// p-values
    pub p_values: Array1<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Pseudo R-squared
    pub pseudo_r2: f64,
    /// Variable names
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for PanelQuantileResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" Panel Quantile Regression (tau={:.2}) ", self.tau)
        )?;
        writeln!(f, "Method: within transformation + quantile regression")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Entities:", self.n_entities)?;
        writeln!(f, "{:<20} {:>12.6}", "Pseudo R-squared:", self.pseudo_r2)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.beta.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            let t_str = if self.t_values[i].is_nan() || self.t_values[i].is_infinite() {
                "—".to_string()
            } else {
                format!("{:>10.3}", self.t_values[i])
            };
            let p_str = if self.p_values[i].is_nan() || self.p_values[i].is_infinite() {
                "—".to_string()
            } else {
                format!("{:>10.4}", self.p_values[i])
            };
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {t_str} {p_str}",
                name, self.beta[i], self.std_errors[i]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct PanelQuantile;

impl PanelQuantile {
    /// Estimate panel quantile regression with fixed effects.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Regressors (n × k, no intercept — FE absorb it)
    /// * `entity_ids` - Entity identifier (n)
    /// * `tau` - Quantile level (0-1)
    /// * `variable_names` - Optional names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
        tau: f64,
        variable_names: Option<Vec<String>>,
    ) -> Result<PanelQuantileResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if x.nrows() != n || entity_ids.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "PanelQuantile: dimension mismatch".into(),
            ));
        }
        if tau <= 0.0 || tau >= 1.0 {
            return Err(GreenersError::InvalidOperation(
                "PanelQuantile: tau must be in (0, 1)".into(),
            ));
        }

        // Within transformation: demean by entity
        let mut entity_y_sums: std::collections::HashMap<i64, (f64, usize)> =
            std::collections::HashMap::new();
        for i in 0..n {
            let entry = entity_y_sums.entry(entity_ids[i]).or_insert((0.0, 0));
            entry.0 += y[i];
            entry.1 += 1;
        }

        let mut entity_x_sums: std::collections::HashMap<i64, Vec<(f64, usize)>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let entry = entity_x_sums
                .entry(entity_ids[i])
                .or_insert_with(|| vec![(0.0, 0); k]);
            for j in 0..k {
                entry[j].0 += x[(i, j)];
                entry[j].1 += 1;
            }
        }

        let n_entities = entity_y_sums.len();

        // Demean
        let mut y_dm = Array1::zeros(n);
        let mut x_dm = Array2::zeros((n, k));
        for i in 0..n {
            let y_mean = {
                let (s, c) = entity_y_sums[&entity_ids[i]];
                s / c as f64
            };
            y_dm[i] = y[i] - y_mean;
            for j in 0..k {
                let x_mean = {
                    let (s, c) = entity_x_sums[&entity_ids[i]][j];
                    s / c as f64
                };
                x_dm[(i, j)] = x[(i, j)] - x_mean;
            }
        }

        // Quantile regression on demeaned data
        let result = QuantileReg::fit(&y_dm, &x_dm, tau, 100)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        // SE from bootstrap
        let std_errors = result.std_errors.clone();
        let t_values = &result.params / &std_errors;
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values = t_values.mapv(|t| {
            if t.is_nan() || t.is_infinite() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal.cdf(t.abs()))
            }
        });

        Ok(PanelQuantileResult {
            tau,
            beta: result.params,
            std_errors,
            t_values,
            p_values,
            n_obs: n,
            n_entities,
            pseudo_r2: result.r_squared,
            variable_names,
        })
    }
}
