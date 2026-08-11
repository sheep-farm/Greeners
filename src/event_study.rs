//! Event study estimation: DiD with leads and lags.
//!
//! Estimates dynamic treatment effects by regressing the outcome on
//! a set of event-time dummies (relative to a reference period),
//! plus controls and fixed effects.

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::{CovarianceType, OlsResult};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fmt;

/// Result of an event study estimation.
#[derive(Debug)]
pub struct EventStudyResult {
    /// Coefficients on event-time dummies (excluding reference period)
    pub event_coefs: Array1<f64>,
    /// Standard errors on event-time dummies
    pub event_se: Array1<f64>,
    /// t-statistics on event-time dummies
    pub event_t: Array1<f64>,
    /// p-values on event-time dummies
    pub event_p: Array1<f64>,
    /// Event times corresponding to each coefficient (excluding reference)
    pub event_times: Vec<i64>,
    /// Reference period (usually 0 or -1)
    pub reference: i64,
    /// Full OLS result (all coefficients)
    pub ols: OlsResult,
    /// Column indices of event dummies in the full design matrix
    pub event_col_indices: Vec<usize>,
}

impl fmt::Display for EventStudyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Event Study Results ")?;
        writeln!(
            f,
            "{:<20} {:>12}  Reference period: t={}",
            "Reference:", "", self.reference
        )?;
        writeln!(f, "{:-^78}", "")?;
        let header = format!(
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Event time", "Coef.", "Std.Err.", "t", "P>|t|"
        );
        writeln!(f, "{header}")?;
        writeln!(f, "{:-^78}", "")?;

        for (i, &t) in self.event_times.iter().enumerate() {
            writeln!(
                f,
                "t={:<8} {:>12.4} {:>12.4} {:>10.3} {:>10.4}",
                t, self.event_coefs[i], self.event_se[i], self.event_t[i], self.event_p[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct EventStudy;

impl EventStudy {
    /// Estimate an event study (dynamic DiD with leads and lags).
    ///
    /// Constructs event-time dummies for each period relative to the
    /// event, excluding the reference period, and runs OLS.
    ///
    /// # Arguments
    /// * `y` - Outcome variable (n × 1)
    /// * `event_time` - Event time for each observation (e.g., -3, -2, -1, 0, 1, 2)
    ///   Use `i64::MIN` or a very negative number for never-treated / not-yet-treated
    /// * `x_controls` - Additional controls (n × k), can be empty (n × 0)
    /// * `reference` - Event time to exclude as reference (usually -1 or 0)
    /// * `min_event_time` - Minimum event time to include (e.g., -5)
    /// * `max_event_time` - Maximum event time to include (e.g., 5)
    /// * `cov_type` - Covariance type for standard errors
    pub fn fit(
        y: &Array1<f64>,
        event_time: &[i64],
        x_controls: &Array2<f64>,
        reference: i64,
        min_event_time: i64,
        max_event_time: i64,
        cov_type: CovarianceType,
    ) -> Result<EventStudyResult, GreenersError> {
        let n = y.len();
        if event_time.len() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "y (len={n}) and event_time (len={}) must have same length",
                event_time.len()
            )));
        }
        if x_controls.nrows() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "y (n={n}) and x_controls (nrows={}) must have same number of rows",
                x_controls.nrows()
            )));
        }

        // Build event-time dummies (excluding reference period)
        let mut event_times: Vec<i64> = Vec::new();
        for t in min_event_time..=max_event_time {
            if t != reference {
                event_times.push(t);
            }
        }
        let n_event = event_times.len();
        let k_controls = x_controls.ncols();
        let k = 1 + n_event + k_controls; // intercept + event dummies + controls

        let mut x = Array2::<f64>::zeros((n, k));
        // Intercept
        for i in 0..n {
            x[(i, 0)] = 1.0;
        }
        // Event dummies
        let mut event_col_indices = Vec::new();
        for (j, &t) in event_times.iter().enumerate() {
            let col = 1 + j;
            event_col_indices.push(col);
            for i in 0..n {
                // Only include observations with valid event time
                if event_time[i] != i64::MIN && event_time[i] == t {
                    x[(i, col)] = 1.0;
                }
            }
        }
        // Controls
        for j in 0..k_controls {
            for i in 0..n {
                x[(i, 1 + n_event + j)] = x_controls[(i, j)];
            }
        }

        // OLS
        let x_t = x.t();
        let xtx = x_t.dot(&x);
        let xtx_inv = xtx.inv()?;
        let xty = x_t.dot(y);
        let beta = xtx_inv.dot(&xty);

        let residuals = y - x.dot(&beta);
        let df_resid = n - k;
        let sigma2 = residuals.dot(&residuals) / df_resid as f64;

        let cov = match cov_type {
            CovarianceType::NonRobust => xtx_inv * sigma2,
            CovarianceType::HC1 => {
                let mut meat = Array2::<f64>::zeros((k, k));
                for i in 0..n {
                    let xi = x.row(i);
                    let ei = residuals[i];
                    let w = if df_resid > 0 {
                        n as f64 / df_resid as f64
                    } else {
                        1.0
                    };
                    for a in 0..k {
                        for b in 0..k {
                            meat[(a, b)] += w * ei * ei * xi[a] * xi[b];
                        }
                    }
                }
                xtx_inv.dot(&meat).dot(&xtx_inv)
            }
            _ => {
                // For other cov types, fall back to HC1
                let mut meat = Array2::<f64>::zeros((k, k));
                for i in 0..n {
                    let xi = x.row(i);
                    let ei = residuals[i];
                    let w = if df_resid > 0 {
                        n as f64 / df_resid as f64
                    } else {
                        1.0
                    };
                    for a in 0..k {
                        for b in 0..k {
                            meat[(a, b)] += w * ei * ei * xi[a] * xi[b];
                        }
                    }
                }
                xtx_inv.dot(&meat).dot(&xtx_inv)
            }
        };

        let std_errors = cov.diag().mapv(|v| v.sqrt());
        let t_values = &beta / &std_errors;
        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - t_dist.cdf(t.abs())));

        // Extract event dummy coefficients
        let mut event_coefs = Vec::new();
        let mut event_se = Vec::new();
        let mut event_t = Vec::new();
        let mut event_p = Vec::new();
        for &col in &event_col_indices {
            event_coefs.push(beta[col]);
            event_se.push(std_errors[col]);
            event_t.push(t_values[col]);
            event_p.push(p_values[col]);
        }

        let r_squared = {
            let y_mean = y.mean().unwrap_or(0.0);
            let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
            let rss = residuals.dot(&residuals);
            if tss > 1e-15 {
                1.0 - rss / tss
            } else {
                0.0
            }
        };

        let ols = OlsResult {
            params: beta,
            std_errors,
            t_values,
            p_values,
            conf_lower: Array1::zeros(k),
            conf_upper: Array1::zeros(k),
            r_squared,
            adj_r_squared: 1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_resid.max(1) as f64,
            f_statistic: 0.0,
            prob_f: 0.0,
            log_likelihood: 0.0,
            aic: 0.0,
            bic: 0.0,
            n_obs: n,
            df_resid,
            df_model: k - 1,
            sigma: sigma2.sqrt(),
            cov_type,
            inference_type: crate::InferenceType::StudentT,
            variable_names: None,
            omitted_vars: Vec::new(),
            x_clean: None,
        };

        Ok(EventStudyResult {
            event_coefs: Array1::from(event_coefs),
            event_se: Array1::from(event_se),
            event_t: Array1::from(event_t),
            event_p: Array1::from(event_p),
            event_times,
            reference,
            ols,
            event_col_indices,
        })
    }
}
