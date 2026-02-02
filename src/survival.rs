use crate::error::GreenersError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

// ─── Kaplan-Meier ──────────────────────────────────────────────────────────────

/// Result of Kaplan-Meier estimation.
#[derive(Debug)]
pub struct KMResult {
    /// Unique event times
    pub times: Array1<f64>,
    /// Survival probability at each time
    pub survival_probs: Array1<f64>,
    /// Greenwood standard errors
    pub std_errors: Array1<f64>,
    /// Lower 95% CI
    pub conf_lower: Array1<f64>,
    /// Upper 95% CI
    pub conf_upper: Array1<f64>,
    /// Median survival time (may be NaN if never drops below 0.5)
    pub median_survival: f64,
    pub n_obs: usize,
    pub n_events: usize,
}

impl fmt::Display for KMResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Kaplan-Meier Survival Estimate ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Events:", self.n_events)?;
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "Median survival:", self.median_survival
        )?;

        writeln!(
            f,
            "\n{:<10} {:>10} {:>10} {:>10} {:>10}",
            "Time", "S(t)", "SE", "Lower", "Upper"
        )?;
        writeln!(f, "{:-^55}", "")?;
        let show = self.times.len().min(20);
        for i in 0..show {
            writeln!(
                f,
                "{:<10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                self.times[i],
                self.survival_probs[i],
                self.std_errors[i],
                self.conf_lower[i],
                self.conf_upper[i]
            )?;
        }
        if self.times.len() > show {
            writeln!(f, "... ({} more time points)", self.times.len() - show)?;
        }
        writeln!(f, "{:=^60}", "")
    }
}

/// Kaplan-Meier survival estimator.
pub struct KaplanMeier;

impl KaplanMeier {
    /// Fit Kaplan-Meier estimator.
    /// times: observed times
    /// events: 1 = event, 0 = censored
    pub fn fit(times: &Array1<f64>, events: &Array1<u8>) -> Result<KMResult, GreenersError> {
        let n = times.len();
        if n != events.len() {
            return Err(GreenersError::ShapeMismatch(
                "times and events length mismatch".into(),
            ));
        }
        if n == 0 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 1 observation".into(),
            ));
        }

        // Sort by time
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());

        // Find unique event times and compute d_i, n_i
        let mut unique_times: Vec<f64> = Vec::new();
        let mut n_events_at: Vec<usize> = Vec::new();
        let mut n_at_risk: Vec<usize> = Vec::new();

        let mut at_risk = n;
        let mut i = 0;
        while i < n {
            let t = times[indices[i]];
            let mut d = 0;
            let mut c = 0;
            while i < n && times[indices[i]] == t {
                if events[indices[i]] == 1 {
                    d += 1;
                } else {
                    c += 1;
                }
                i += 1;
            }
            if d > 0 {
                unique_times.push(t);
                n_events_at.push(d);
                n_at_risk.push(at_risk);
            }
            at_risk -= d + c;
        }

        let m = unique_times.len();
        let mut survival_probs = Array1::<f64>::zeros(m);
        let mut std_errors = Array1::<f64>::zeros(m);
        let mut greenwood_sum = 0.0;
        let mut s = 1.0;
        let total_events: usize = n_events_at.iter().sum();

        for j in 0..m {
            let nj = n_at_risk[j] as f64;
            let dj = n_events_at[j] as f64;
            s *= 1.0 - dj / nj;
            survival_probs[j] = s;

            if nj > dj {
                greenwood_sum += dj / (nj * (nj - dj));
            }
            std_errors[j] = s * greenwood_sum.sqrt();
        }

        let z = 1.96;
        let conf_lower = &survival_probs - &(&std_errors * z);
        let conf_upper = &survival_probs + &(&std_errors * z);
        let conf_lower = conf_lower.mapv(|v| v.max(0.0));
        let conf_upper = conf_upper.mapv(|v| v.min(1.0));

        // Median survival
        let median_survival = {
            let mut med = f64::NAN;
            for j in 0..m {
                if survival_probs[j] <= 0.5 {
                    med = unique_times[j];
                    break;
                }
            }
            med
        };

        Ok(KMResult {
            times: Array1::from(unique_times),
            survival_probs,
            std_errors,
            conf_lower,
            conf_upper,
            median_survival,
            n_obs: n,
            n_events: total_events,
        })
    }
}

// ─── Cox PH ────────────────────────────────────────────────────────────────────

/// Result of Cox Proportional Hazards model.
#[derive(Debug)]
pub struct CoxResult {
    /// Coefficients (log hazard ratios)
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    /// exp(beta)
    pub hazard_ratios: Array1<f64>,
    /// Partial log-likelihood
    pub log_likelihood: f64,
    /// Concordance index
    pub concordance: f64,
    pub n_obs: usize,
    pub n_events: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub variable_names: Option<Vec<String>>,
}

impl CoxResult {
    /// Predict log-partial hazard for new data.
    pub fn predict_log_hazard(&self, x_new: &Array2<f64>) -> Array1<f64> {
        x_new.dot(&self.params)
    }

    /// Predict hazard ratio relative to baseline.
    pub fn predict_hazard_ratio(&self, x_new: &Array2<f64>) -> Array1<f64> {
        x_new.dot(&self.params).mapv(f64::exp)
    }
}

impl fmt::Display for CoxResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Cox Proportional Hazards Model ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Events:", self.n_events)?;
        writeln!(f, "{:<20} {:>10.4}", "Log-Likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>10.4}", "Concordance:", self.concordance)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>8} | {:>8} | {:>10}",
            "Variable", "coef", "std err", "z", "P>|z|", "exp(coef)"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3} | {:>10.4}",
                name,
                self.params[i],
                self.std_errors[i],
                self.z_values[i],
                self.p_values[i],
                self.hazard_ratios[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// Cox Proportional Hazards model.
pub struct CoxPH;

impl CoxPH {
    pub fn fit(
        times: &Array1<f64>,
        events: &Array1<u8>,
        x: &Array2<f64>,
    ) -> Result<CoxResult, GreenersError> {
        Self::fit_with_names(times, events, x, None)
    }

    pub fn fit_with_names(
        times: &Array1<f64>,
        events: &Array1<u8>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<CoxResult, GreenersError> {
        let n = times.len();
        let k = x.ncols();

        if n != events.len() || n != x.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "times, events, and x dimension mismatch".into(),
            ));
        }

        let n_events: usize = events.iter().map(|&e| e as usize).sum();
        if n_events == 0 {
            return Err(GreenersError::InvalidOperation("No events observed".into()));
        }

        // Sort by time (descending for risk set computation)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());

        let mut beta = Array1::<f64>::zeros(k);
        let max_iter = 100;
        let tol = 1e-9;
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            let exp_xb: Array1<f64> = x.dot(&beta).mapv(f64::exp);

            let mut gradient = Array1::<f64>::zeros(k);
            let mut hessian = Array2::<f64>::zeros((k, k));

            for &i in &order {
                if events[i] == 1 {
                    // Risk set: all j with time[j] >= time[i]
                    let mut rs = 0.0;
                    let mut rs_x = Array1::<f64>::zeros(k);
                    let mut rs_xx = Array2::<f64>::zeros((k, k));

                    for &j in &order {
                        if times[j] >= times[i] {
                            rs += exp_xb[j];
                            let xj = x.row(j);
                            for a in 0..k {
                                rs_x[a] += exp_xb[j] * xj[a];
                                for b in 0..k {
                                    rs_xx[[a, b]] += exp_xb[j] * xj[a] * xj[b];
                                }
                            }
                        }
                    }

                    let xi = x.row(i);

                    for a in 0..k {
                        gradient[a] += xi[a] - rs_x[a] / rs;
                        for b in 0..k {
                            hessian[[a, b]] -= rs_xx[[a, b]] / rs - (rs_x[a] * rs_x[b]) / (rs * rs);
                        }
                    }
                }
            }

            let neg_hessian = hessian.mapv(|h| -h);
            let delta = match neg_hessian.inv() {
                Ok(inv) => inv.dot(&gradient),
                Err(_) => break,
            };

            let new_beta = &beta + &delta;
            let diff = delta.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
            beta = new_beta;

            if diff < tol {
                converged = true;
                break;
            }
        }

        // Standard errors from observed information
        let exp_xb: Array1<f64> = x.dot(&beta).mapv(f64::exp);
        let mut info = Array2::<f64>::zeros((k, k));

        for &i in &order {
            if events[i] == 1 {
                let mut rs = 0.0;
                let mut rs_x = Array1::<f64>::zeros(k);
                let mut rs_xx = Array2::<f64>::zeros((k, k));

                for &j in &order {
                    if times[j] >= times[i] {
                        rs += exp_xb[j];
                        let xj = x.row(j);
                        for a in 0..k {
                            rs_x[a] += exp_xb[j] * xj[a];
                            for b in 0..k {
                                rs_xx[[a, b]] += exp_xb[j] * xj[a] * xj[b];
                            }
                        }
                    }
                }

                for a in 0..k {
                    for b in 0..k {
                        info[[a, b]] += rs_xx[[a, b]] / rs - (rs_x[a] * rs_x[b]) / (rs * rs);
                    }
                }
            }
        }

        let cov = info.inv()?;
        let std_errors: Array1<f64> = (0..k)
            .map(|j| cov[[j, j]].abs().sqrt())
            .collect::<Vec<_>>()
            .into();

        let z_values = &beta / &std_errors;
        let normal = Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal.cdf(z.abs())));
        let hazard_ratios = beta.mapv(f64::exp);

        // Concordance index
        let concordance = compute_concordance(times, events, &x.dot(&beta));

        // Final log-likelihood
        let mut ll = 0.0;
        for &i in &order {
            if events[i] == 1 {
                let mut rs = 0.0;
                for &j in &order {
                    if times[j] >= times[i] {
                        rs += exp_xb[j];
                    }
                }
                ll += x.row(i).dot(&beta) - rs.ln();
            }
        }

        Ok(CoxResult {
            params: beta,
            std_errors,
            z_values,
            p_values,
            hazard_ratios,
            log_likelihood: ll,
            concordance,
            n_obs: n,
            n_events,
            n_iter,
            converged,
            variable_names,
        })
    }
}

fn compute_concordance(times: &Array1<f64>, events: &Array1<u8>, risk_scores: &Array1<f64>) -> f64 {
    let n = times.len();
    let mut concordant = 0u64;
    let mut discordant = 0u64;

    for i in 0..n {
        if events[i] != 1 {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            if times[j] > times[i] {
                if risk_scores[i] > risk_scores[j] {
                    concordant += 1;
                } else if risk_scores[i] < risk_scores[j] {
                    discordant += 1;
                }
            }
        }
    }

    let total = concordant + discordant;
    if total == 0 {
        0.5
    } else {
        concordant as f64 / total as f64
    }
}
