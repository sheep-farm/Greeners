//! Hawkes process — self-exciting point process.
//!
//! Hawkes (1971). Models event arrival times where each event
//! increases the intensity of future events temporarily:
//!
//! lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
//!
//! where:
//!   - mu: baseline intensity
//!   - alpha: excitation magnitude (each event adds alpha to intensity)
//!   - beta: decay rate of excitation
//!
//! Branching ratio: eta = alpha / beta (must be < 1 for stationarity)
//!
//! Estimation: maximum likelihood via grid search over (mu, alpha, beta).
//! Log-likelihood:
//!   L = sum log(lambda(t_i)) - integral_0^T lambda(t) dt

use crate::GreenersError;
use ndarray::Array1;
use std::fmt;

/// Result of Hawkes process estimation.
#[derive(Debug)]
pub struct HawkesResult {
    /// Baseline intensity (mu)
    pub mu: f64,
    /// Excitation magnitude (alpha)
    pub alpha: f64,
    /// Decay rate (beta)
    pub beta: f64,
    /// Branching ratio (eta = alpha/beta)
    pub branching_ratio: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of events
    pub n_events: usize,
    /// Observation window [0, T]
    pub time_window: f64,
    /// Estimated intensity at event times (n_events)
    pub intensity_at_events: Array1<f64>,
}

impl fmt::Display for HawkesResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Hawkes Process ")?;
        writeln!(f, "Hawkes (1971) — self-exciting point process")?;
        writeln!(f, "lambda(t) = mu + sum alpha*exp(-beta*(t-t_i))")?;
        writeln!(f, "{:<20} {:>12}", "Events:", self.n_events)?;
        writeln!(f, "{:<20} {:>12.4}", "Time window:", self.time_window)?;
        writeln!(f, "{:<20} {:>12.6}", "mu (baseline):", self.mu)?;
        writeln!(f, "{:<20} {:>12.6}", "alpha (excitation):", self.alpha)?;
        writeln!(f, "{:<20} {:>12.6}", "beta (decay):", self.beta)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Branching ratio:", self.branching_ratio
        )?;
        let stability = if self.branching_ratio < 1.0 {
            "STABLE"
        } else {
            "UNSTABLE"
        };
        writeln!(f, "{:<20} {}", "Stationarity:", stability)?;
        writeln!(f, "{:<20} {:>12.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        // Intensity at selected events
        let n_show = 5.min(self.n_events);
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Intensity at selected events:")?;
        writeln!(f, "  {:<8} {:>12}", "Event", "lambda(t_i)")?;
        let indices: Vec<usize> = if self.n_events <= n_show {
            (0..self.n_events).collect()
        } else {
            (0..n_show)
                .map(|i| i * (self.n_events - 1) / (n_show - 1).max(1))
                .collect()
        };
        for &idx in &indices {
            let _ = writeln!(
                f,
                "  {:<8} {:>12.6}",
                idx + 1,
                self.intensity_at_events[idx]
            );
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct Hawkes;

impl Hawkes {
    /// Estimate Hawkes process parameters via MLE (grid search).
    ///
    /// # Arguments
    /// * `event_times` - Event arrival times (sorted ascending)
    /// * `time_window` - Observation window T (last event time or larger)
    pub fn fit(
        event_times: &[f64],
        time_window: Option<f64>,
    ) -> Result<HawkesResult, GreenersError> {
        let n = event_times.len();
        if n < 5 {
            return Err(GreenersError::InvalidOperation(
                "Hawkes: need at least 5 events".into(),
            ));
        }

        // Check sorted
        for i in 1..n {
            if event_times[i] < event_times[i - 1] {
                return Err(GreenersError::InvalidOperation(
                    "Hawkes: event times must be sorted ascending".into(),
                ));
            }
        }

        let t_max = time_window.unwrap_or(event_times[n - 1]);
        if t_max <= 0.0 {
            return Err(GreenersError::InvalidOperation(
                "Hawkes: time window must be positive".into(),
            ));
        }

        // Grid search over (mu, alpha, beta)
        // mu: baseline intensity (events per unit time)
        let mu_init = n as f64 / t_max;

        let mut best_mu = mu_init * 0.5;
        let mut best_alpha = 0.1;
        let mut best_beta = 1.0;
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_intensity = Array1::zeros(n);

        let n_grid = 8;
        for mi in 0..n_grid {
            let mu = mu_init * (0.1 + 0.9 * mi as f64 / (n_grid - 1) as f64);
            for ai in 0..n_grid {
                let alpha = 0.01 + 2.0 * ai as f64 / (n_grid - 1) as f64;
                for bi in 0..n_grid {
                    let beta = 0.1 + 10.0 * bi as f64 / (n_grid - 1) as f64;

                    // Stationarity constraint: alpha/beta < 1
                    if alpha / beta >= 0.95 {
                        continue;
                    }

                    let (ll, intensity) = Self::log_likelihood(event_times, mu, alpha, beta, t_max);
                    if ll > best_ll {
                        best_ll = ll;
                        best_mu = mu;
                        best_alpha = alpha;
                        best_beta = beta;
                        best_intensity = intensity;
                    }
                }
            }
        }

        let branching_ratio = best_alpha / best_beta;
        let n_params = 3;
        let aic = -2.0 * best_ll + 2.0 * n_params as f64;
        let bic = -2.0 * best_ll + (n as f64) * n_params as f64;

        Ok(HawkesResult {
            mu: best_mu,
            alpha: best_alpha,
            beta: best_beta,
            branching_ratio,
            log_likelihood: best_ll,
            aic,
            bic,
            n_events: n,
            time_window: t_max,
            intensity_at_events: best_intensity,
        })
    }

    /// Compute log-likelihood and intensity at event times.
    fn log_likelihood(
        events: &[f64],
        mu: f64,
        alpha: f64,
        beta: f64,
        t_max: f64,
    ) -> (f64, Array1<f64>) {
        let n = events.len();
        let mut ll = 0.0_f64;
        let mut intensity = Array1::zeros(n);

        for i in 0..n {
            // lambda(t_i) = mu + sum_{j<i} alpha * exp(-beta * (t_i - t_j))
            let mut lam = mu;
            for j in 0..i {
                let dt = events[i] - events[j];
                lam += alpha * (-beta * dt).exp();
            }
            intensity[i] = lam;
            if lam > 0.0 {
                ll += lam.ln();
            } else {
                ll += -1e10;
            }
        }

        // Integral: int_0^T lambda(t) dt = mu*T + sum_i (alpha/beta)*(1 - exp(-beta*(T - t_i)))
        let integral = mu * t_max
            + (alpha / beta)
                * events
                    .iter()
                    .map(|&t| 1.0 - (-beta * (t_max - t)).exp())
                    .sum::<f64>();

        ll -= integral;

        (ll, intensity)
    }
}
