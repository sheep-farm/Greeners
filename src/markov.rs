use crate::GreenersError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use std::fmt;

/// Markov switching model result.
#[derive(Debug)]
pub struct MarkovSwitchingResult {
    /// Regime-specific parameters (intercept + AR coefficients per regime).
    pub regime_params: Vec<Array1<f64>>,
    /// Regime-specific variances.
    pub regime_variances: Array1<f64>,
    /// Transition probability matrix (K x K), element [i,j] = P(s_t=j | s_{t-1}=i).
    pub transition_matrix: Array2<f64>,
    /// Filtered regime probabilities (T x K).
    pub filtered_probs: Array2<f64>,
    /// Smoothed regime probabilities (T x K).
    pub smoothed_probs: Array2<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub n_regimes: usize,
    pub ar_order: usize,
}

impl MarkovSwitchingResult {
    /// Forecast `steps` ahead using regime-weighted predictions.
    pub fn predict(&self, y: &Array1<f64>, steps: usize) -> Array1<f64> {
        let k = self.n_regimes;
        let p = self.ar_order;

        // Start with last smoothed probabilities
        let n_eff = self.smoothed_probs.nrows();
        let mut probs = Array1::<f64>::zeros(k);
        for j in 0..k {
            probs[j] = self.smoothed_probs[[n_eff - 1, j]];
        }

        let mut history: Vec<f64> = y.to_vec();
        let mut forecasts = Array1::<f64>::zeros(steps);

        for h in 0..steps {
            // Evolve probabilities: P(s_{t+1}) = P' * P(s_t)
            let new_probs = self.transition_matrix.t().dot(&probs);

            let mut weighted_forecast = 0.0;
            for j in 0..k {
                let params = &self.regime_params[j];
                let mut yhat = params[0]; // intercept
                let cur_len = history.len();
                for lag in 0..p {
                    if 1 + lag < params.len() && lag < cur_len {
                        yhat += params[1 + lag] * history[cur_len - 1 - lag];
                    }
                }
                weighted_forecast += new_probs[j] * yhat;
            }

            forecasts[h] = weighted_forecast;
            history.push(weighted_forecast);
            probs = new_probs;
        }

        forecasts
    }

    /// Expected duration in each regime: 1 / (1 - p_ii).
    pub fn expected_durations(&self) -> Array1<f64> {
        let k = self.n_regimes;
        Array1::from_vec(
            (0..k)
                .map(|i| {
                    let p_ii = self.transition_matrix[[i, i]];
                    if (1.0 - p_ii).abs() < 1e-15 {
                        f64::INFINITY
                    } else {
                        1.0 / (1.0 - p_ii)
                    }
                })
                .collect(),
        )
    }
}

impl fmt::Display for MarkovSwitchingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(
                " Markov Switching AR({}) — {} regimes ",
                self.ar_order, self.n_regimes
            )
        )?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>10.4}", "BIC:", self.bic)?;

        writeln!(f, "\n{:-^78}", " Transition Matrix ")?;
        for row in self.transition_matrix.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>8.4} ", val)?;
            }
            writeln!(f, "]")?;
        }

        for (j, params) in self.regime_params.iter().enumerate() {
            writeln!(f, "\n{:-^78}", format!(" Regime {} ", j))?;
            writeln!(f, "{:<20} {:>10.4}", "Variance:", self.regime_variances[j])?;
            writeln!(f, "{:<20} {:>10.4}", "Intercept:", params[0])?;
            for lag in 1..params.len() {
                writeln!(f, "{:<20} {:>10.4}", format!("AR.L{}", lag), params[lag])?;
            }
        }

        let durations = self.expected_durations();
        writeln!(f, "\n{:-^78}", " Expected Durations ")?;
        for j in 0..self.n_regimes {
            writeln!(f, "Regime {}: {:.1} periods", j, durations[j])?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// Markov Switching model with regime-dependent mean and variance.
pub struct MarkovSwitching;

impl MarkovSwitching {
    /// Fit a Markov Switching AR model via EM algorithm (Hamilton filter).
    ///
    /// * `y` — time series
    /// * `n_regimes` — number of regimes (typically 2)
    /// * `ar_order` — autoregressive order
    pub fn fit(
        y: &Array1<f64>,
        n_regimes: usize,
        ar_order: usize,
    ) -> Result<MarkovSwitchingResult, GreenersError> {
        let n = y.len();
        let p = ar_order;
        let k = n_regimes;

        if n < p + 10 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for Markov Switching".into(),
            ));
        }
        if k < 2 {
            return Err(GreenersError::ShapeMismatch(
                "Need at least 2 regimes".into(),
            ));
        }

        let effective_n = n - p;
        let y_mean = y.mean().unwrap_or(0.0);
        let y_var = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64;

        // Initialize parameters
        // Spread regime means across the data range
        let mut regime_params: Vec<Array1<f64>> = Vec::new();
        for j in 0..k {
            let mut params = Array1::<f64>::zeros(1 + p);
            // Spread intercepts
            params[0] = y_mean + (j as f64 - (k - 1) as f64 / 2.0) * y_var.sqrt();
            // Small AR coefficients
            for lag in 0..p {
                params[1 + lag] = 0.1 / (1 + lag) as f64;
            }
            regime_params.push(params);
        }

        let mut regime_variances = Array1::from_elem(k, y_var);
        // Give different initial variances to each regime
        for j in 0..k {
            regime_variances[j] = y_var * (0.5 + j as f64);
        }

        // Initialize transition matrix: high persistence
        let mut trans = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    trans[[i, j]] = 0.9;
                } else {
                    trans[[i, j]] = 0.1 / (k - 1) as f64;
                }
            }
        }

        let max_iter = 200;
        let tol = 1e-6;
        let mut prev_ll = f64::NEG_INFINITY;

        let mut filtered_probs = Array2::<f64>::zeros((effective_n, k));
        let mut smoothed_probs = Array2::<f64>::zeros((effective_n, k));
        let mut log_likelihood = 0.0;

        for _iter in 0..max_iter {
            // ===== E-step: Hamilton filter =====
            let mut xi_filtered = Array2::<f64>::zeros((effective_n, k));
            log_likelihood = 0.0;

            // Ergodic probabilities as initial
            let mut xi_prev = Array1::from_elem(k, 1.0 / k as f64);

            for t in 0..effective_n {
                let t_idx = t + p;

                // Compute likelihood for each regime
                let mut eta = Array1::<f64>::zeros(k);
                for j in 0..k {
                    let params = &regime_params[j];
                    let mut y_hat = params[0];
                    for lag in 0..p {
                        y_hat += params[1 + lag] * y[t_idx - 1 - lag];
                    }
                    let resid = y[t_idx] - y_hat;
                    let var = regime_variances[j].max(1e-10);
                    eta[j] = (-0.5 * resid * resid / var).exp()
                        / (2.0 * std::f64::consts::PI * var).sqrt();
                }

                // Prediction: P(s_t | Y_{t-1}) = P' * xi_{t-1|t-1}
                let xi_pred = trans.t().dot(&xi_prev);

                // Joint: P(s_t, y_t | Y_{t-1}) = eta_t * xi_pred
                let joint = &eta * &xi_pred;
                let f_t: f64 = joint.sum();

                if f_t > 1e-30 {
                    // Filtered: P(s_t | Y_t)
                    let xi_filt = &joint / f_t;
                    xi_filtered.row_mut(t).assign(&xi_filt);
                    xi_prev = xi_filt;
                    log_likelihood += f_t.ln();
                } else {
                    // Degenerate case
                    xi_filtered.row_mut(t).fill(1.0 / k as f64);
                    xi_prev = Array1::from_elem(k, 1.0 / k as f64);
                }
            }

            filtered_probs = xi_filtered.clone();

            // Backward pass: Kim's smoothing
            smoothed_probs = xi_filtered.clone();
            for t in (0..effective_n - 1).rev() {
                let xi_filt_t = xi_filtered.row(t).to_owned();
                let xi_pred_next = trans.t().dot(&xi_filt_t);

                for j in 0..k {
                    let mut sum = 0.0;
                    for l in 0..k {
                        let pred_l = xi_pred_next[l].max(1e-30);
                        sum += trans[[j, l]] * smoothed_probs[[t + 1, l]] / pred_l;
                    }
                    smoothed_probs[[t, j]] = xi_filt_t[j] * sum;
                }

                // Normalize
                let row_sum: f64 = smoothed_probs.row(t).sum();
                if row_sum > 1e-30 {
                    for j in 0..k {
                        smoothed_probs[[t, j]] /= row_sum;
                    }
                }
            }

            // Check convergence
            if (log_likelihood - prev_ll).abs() < tol {
                break;
            }
            prev_ll = log_likelihood;

            // ===== M-step =====
            // Update regime parameters via weighted least squares
            for j in 0..k {
                let weights: Vec<f64> = (0..effective_n)
                    .map(|t| smoothed_probs[[t, j]].max(1e-15))
                    .collect();
                let w_sum: f64 = weights.iter().sum();

                if w_sum < 1e-10 {
                    continue;
                }

                // WLS: solve (X'WX) beta = X'Wy
                let n_params = 1 + p;
                let mut xtwx = Array2::<f64>::zeros((n_params, n_params));
                let mut xtwy = Array1::<f64>::zeros(n_params);

                for (t, &w) in weights.iter().enumerate().take(effective_n) {
                    let t_idx = t + p;

                    let mut x_row = vec![1.0]; // intercept
                    for lag in 0..p {
                        x_row.push(y[t_idx - 1 - lag]);
                    }

                    for a in 0..n_params {
                        for b in 0..n_params {
                            xtwx[[a, b]] += w * x_row[a] * x_row[b];
                        }
                        xtwy[a] += w * x_row[a] * y[t_idx];
                    }
                }

                // Solve
                if let Ok(xtwx_inv) = xtwx.inv() {
                    let result: Array1<f64> = xtwx_inv.dot(&xtwy);
                    regime_params[j] = result;
                }

                // Update variance
                let mut wss = 0.0;
                for (t, &w) in weights.iter().enumerate().take(effective_n) {
                    let t_idx = t + p;
                    let params = &regime_params[j];
                    let mut y_hat = params[0];
                    for lag in 0..p {
                        y_hat += params[1 + lag] * y[t_idx - 1 - lag];
                    }
                    let resid = y[t_idx] - y_hat;
                    wss += w * resid * resid;
                }
                regime_variances[j] = (wss / w_sum).max(1e-10);
            }

            // Update transition matrix
            for i in 0..k {
                let mut row_sum = 0.0;
                for j in 0..k {
                    let mut num = 0.0;
                    for t in 0..effective_n - 1 {
                        let xi_filt_t = filtered_probs[[t, i]].max(1e-30);
                        let xi_pred_next_j = {
                            let xi_pred = trans.t().dot(&filtered_probs.row(t).to_owned());
                            xi_pred[j].max(1e-30)
                        };
                        num +=
                            trans[[i, j]] * xi_filt_t * smoothed_probs[[t + 1, j]] / xi_pred_next_j;
                    }
                    trans[[i, j]] = num.max(1e-10);
                    row_sum += trans[[i, j]];
                }
                // Normalize row
                if row_sum > 1e-30 {
                    for j in 0..k {
                        trans[[i, j]] /= row_sum;
                    }
                }
            }
        }

        // Compute AIC/BIC
        let n_params = k * (1 + p) + k + k * (k - 1); // regime params + variances + transition probs
        let nf = effective_n as f64;
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + n_params as f64 * nf.ln();

        Ok(MarkovSwitchingResult {
            regime_params,
            regime_variances,
            transition_matrix: trans,
            filtered_probs,
            smoothed_probs,
            log_likelihood,
            aic,
            bic,
            n_obs: effective_n,
            n_regimes: k,
            ar_order: p,
        })
    }
}
