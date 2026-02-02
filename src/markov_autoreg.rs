use crate::GreenersError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use std::fmt;

/// Markov-Switching Autoregression model.
///
/// Model: y_t = mu_{s_t} + phi_1(s_t) * y_{t-1} + ... + phi_p(s_t) * y_{t-p} + sigma_{s_t} * eps_t
/// where s_t follows a Markov chain with transition matrix P.
pub struct MarkovAutoregression;

/// Result of fitting a Markov-Switching AR(p) model.
#[derive(Debug)]
pub struct MarkovAutoregResult {
    /// Regime-specific intercepts (k).
    pub regime_means: Array1<f64>,
    /// AR coefficients per regime (k x p). Row j = AR params for regime j.
    pub ar_params: Array2<f64>,
    /// Regime-specific standard deviations (k).
    pub regime_sigmas: Array1<f64>,
    /// Transition probability matrix (k x k). Element [i,j] = P(s_t=j | s_{t-1}=i).
    pub transition_matrix: Array2<f64>,
    /// Smoothed regime probabilities (T x k).
    pub smoothed_probs: Array2<f64>,
    /// Filtered regime probabilities (T x k).
    pub filtered_probs: Array2<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub k_regimes: usize,
    pub ar_order: usize,
}

impl MarkovAutoregResult {
    /// Return the most likely regime at each time point.
    pub fn predict_regime(&self) -> Array1<usize> {
        let t = self.smoothed_probs.nrows();
        let mut regimes = Array1::<usize>::zeros(t);
        for i in 0..t {
            let mut best_j = 0;
            let mut best_p = self.smoothed_probs[[i, 0]];
            for j in 1..self.k_regimes {
                if self.smoothed_probs[[i, j]] > best_p {
                    best_p = self.smoothed_probs[[i, j]];
                    best_j = j;
                }
            }
            regimes[i] = best_j;
        }
        regimes
    }
}

impl fmt::Display for MarkovAutoregResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(
                " Markov Autoregression AR({}) — {} regimes ",
                self.ar_order, self.k_regimes
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

        for j in 0..self.k_regimes {
            writeln!(f, "\n{:-^78}", format!(" Regime {} ", j))?;
            writeln!(f, "{:<20} {:>10.4}", "Sigma:", self.regime_sigmas[j])?;
            writeln!(f, "{:<20} {:>10.4}", "Intercept:", self.regime_means[j])?;
            for lag in 0..self.ar_order {
                writeln!(
                    f,
                    "{:<20} {:>10.4}",
                    format!("AR.L{}", lag + 1),
                    self.ar_params[[j, lag]]
                )?;
            }
        }

        // Expected durations
        writeln!(f, "\n{:-^78}", " Expected Durations ")?;
        for j in 0..self.k_regimes {
            let p_jj = self.transition_matrix[[j, j]];
            let dur = if (1.0 - p_jj).abs() < 1e-15 {
                f64::INFINITY
            } else {
                1.0 / (1.0 - p_jj)
            };
            writeln!(f, "Regime {}: {:.1} periods", j, dur)?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

impl MarkovAutoregression {
    /// Fit a Markov-switching AR(p) model with k regimes via EM algorithm.
    ///
    /// * `y` — time series data
    /// * `k_regimes` — number of regimes (>= 2)
    /// * `ar_order` — autoregressive order (p >= 0)
    pub fn fit(
        y: &Array1<f64>,
        k_regimes: usize,
        ar_order: usize,
    ) -> Result<MarkovAutoregResult, GreenersError> {
        let n = y.len();
        let p = ar_order;
        let k = k_regimes;

        if n < p + 10 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for MarkovAutoregression".into(),
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

        // Initialize regime means spread across data range
        let mut regime_means = Array1::<f64>::zeros(k);
        for j in 0..k {
            regime_means[j] = y_mean + (j as f64 - (k - 1) as f64 / 2.0) * y_var.sqrt();
        }

        // Initialize AR params to small values
        let mut ar_params = Array2::<f64>::zeros((k, p.max(1)));
        let ar_params_cols = p;
        if p > 0 {
            ar_params = Array2::<f64>::zeros((k, p));
            for j in 0..k {
                for lag in 0..p {
                    ar_params[[j, lag]] = 0.1 / (1 + lag) as f64;
                }
            }
        }

        // Initialize sigmas
        let mut regime_sigmas = Array1::<f64>::zeros(k);
        for j in 0..k {
            regime_sigmas[j] = (y_var * (0.5 + j as f64)).sqrt();
        }

        // Transition matrix: high persistence
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

            let mut xi_prev = Array1::from_elem(k, 1.0 / k as f64);

            for t in 0..effective_n {
                let t_idx = t + p;

                // Compute conditional density for each regime
                let mut eta = Array1::<f64>::zeros(k);
                for j in 0..k {
                    let mut y_hat = regime_means[j];
                    for lag in 0..p {
                        y_hat += ar_params[[j, lag]] * y[t_idx - 1 - lag];
                    }
                    let resid = y[t_idx] - y_hat;
                    let var = (regime_sigmas[j] * regime_sigmas[j]).max(1e-10);
                    eta[j] = (-0.5 * resid * resid / var).exp()
                        / (2.0 * std::f64::consts::PI * var).sqrt();
                }

                // Prediction step
                let xi_pred = trans.t().dot(&xi_prev);

                // Update step
                let joint = &eta * &xi_pred;
                let f_t: f64 = joint.sum();

                if f_t > 1e-30 {
                    let xi_filt = &joint / f_t;
                    xi_filtered.row_mut(t).assign(&xi_filt);
                    xi_prev = xi_filt;
                    log_likelihood += f_t.ln();
                } else {
                    xi_filtered.row_mut(t).fill(1.0 / k as f64);
                    xi_prev = Array1::from_elem(k, 1.0 / k as f64);
                }
            }

            filtered_probs = xi_filtered.clone();

            // Kim smoother (backward pass)
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

            // ===== M-step: weighted least squares =====
            for j in 0..k {
                let weights: Vec<f64> = (0..effective_n)
                    .map(|t| smoothed_probs[[t, j]].max(1e-15))
                    .collect();
                let w_sum: f64 = weights.iter().sum();

                if w_sum < 1e-10 {
                    continue;
                }

                let n_coefs = 1 + p;
                let mut xtwx = Array2::<f64>::zeros((n_coefs, n_coefs));
                let mut xtwy = Array1::<f64>::zeros(n_coefs);

                for (t, &w) in weights.iter().enumerate().take(effective_n) {
                    let t_idx = t + p;
                    let mut x_row = vec![1.0];
                    for lag in 0..p {
                        x_row.push(y[t_idx - 1 - lag]);
                    }

                    for a in 0..n_coefs {
                        for b in 0..n_coefs {
                            xtwx[[a, b]] += w * x_row[a] * x_row[b];
                        }
                        xtwy[a] += w * x_row[a] * y[t_idx];
                    }
                }

                if let Ok(xtwx_inv) = xtwx.inv() {
                    let beta = xtwx_inv.dot(&xtwy);
                    regime_means[j] = beta[0];
                    for lag in 0..p {
                        ar_params[[j, lag]] = beta[1 + lag];
                    }
                }

                // Update sigma
                let mut wss = 0.0;
                for (t, &w) in weights.iter().enumerate().take(effective_n) {
                    let t_idx = t + p;
                    let mut y_hat = regime_means[j];
                    for lag in 0..p {
                        y_hat += ar_params[[j, lag]] * y[t_idx - 1 - lag];
                    }
                    let resid = y[t_idx] - y_hat;
                    wss += w * resid * resid;
                }
                regime_sigmas[j] = (wss / w_sum).max(1e-10).sqrt();
            }

            // Update transition matrix
            for i in 0..k {
                let mut row_sum = 0.0;
                for j in 0..k {
                    let mut num = 0.0;
                    for t in 0..effective_n - 1 {
                        let xi_filt_t_i = filtered_probs[[t, i]].max(1e-30);
                        let xi_pred = trans.t().dot(&filtered_probs.row(t).to_owned());
                        let xi_pred_next_j = xi_pred[j].max(1e-30);
                        num += trans[[i, j]] * xi_filt_t_i * smoothed_probs[[t + 1, j]]
                            / xi_pred_next_j;
                    }
                    trans[[i, j]] = num.max(1e-10);
                    row_sum += trans[[i, j]];
                }
                if row_sum > 1e-30 {
                    for j in 0..k {
                        trans[[i, j]] /= row_sum;
                    }
                }
            }
        }

        // AIC / BIC
        let n_free_params = k + k * ar_params_cols + k + k * (k - 1);
        let nf = effective_n as f64;
        let aic = -2.0 * log_likelihood + 2.0 * n_free_params as f64;
        let bic = -2.0 * log_likelihood + n_free_params as f64 * nf.ln();

        Ok(MarkovAutoregResult {
            regime_means,
            ar_params: if p > 0 {
                ar_params
            } else {
                Array2::zeros((k, 0))
            },
            regime_sigmas,
            transition_matrix: trans,
            smoothed_probs,
            filtered_probs,
            log_likelihood,
            aic,
            bic,
            n_obs: effective_n,
            k_regimes: k,
            ar_order: p,
        })
    }
}
