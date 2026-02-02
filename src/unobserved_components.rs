use crate::statespace::{KalmanFilter, KalmanSmoother, StateSpaceModel};
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Unobserved Components model.
pub struct UnobservedComponents;

/// Level/trend specification for UC models.
#[derive(Debug, Clone)]
pub enum UCLevel {
    /// y_t = mu_t + eps_t; mu_{t+1} = mu_t + eta_t
    LocalLevel,
    /// Adds trend: mu_{t+1} = mu_t + nu_t + eta_t; nu_{t+1} = nu_t + zeta_t
    LocalLinearTrend,
    /// nu_{t+1} = nu_t + zeta_t (no level disturbance, only trend disturbance)
    SmoothTrend,
    /// mu_{t+1} = mu_t + eta_t (no observation noise)
    RandomWalk,
}

/// Seasonal component specification.
#[derive(Debug, Clone)]
pub enum UCSeasonal {
    /// No seasonal component.
    None,
    /// Fixed seasonal dummies with given period.
    Deterministic(usize),
    /// Time-varying seasonal with given period.
    Stochastic(usize),
}

/// Result of fitting an Unobserved Components model.
#[derive(Debug)]
pub struct UCResult {
    /// Smoothed level component.
    pub level: Array1<f64>,
    /// Smoothed trend component (if applicable).
    pub trend: Option<Array1<f64>>,
    /// Smoothed seasonal component (if applicable).
    pub seasonal: Option<Array1<f64>>,
    /// Residuals (y - level - trend - seasonal).
    pub residuals: Array1<f64>,
    /// Estimated variance parameters.
    pub params: Vec<f64>,
    /// Names of the variance parameters.
    pub param_names: Vec<String>,
    /// Log-likelihood at the optimum.
    pub log_likelihood: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,
    /// Number of observations.
    pub n_obs: usize,
    /// Level type used.
    pub level_type: UCLevel,
    /// Seasonal type used.
    pub seasonal_type: UCSeasonal,
    /// The fitted state space model (for forecasting).
    ssm: StateSpaceModel,
    /// Last filtered state (for forecasting).
    last_state: Array1<f64>,
}

impl UCResult {
    /// Forecast `steps` ahead.
    pub fn predict(&self, steps: usize) -> Vec<f64> {
        let mut forecasts = Vec::with_capacity(steps);
        let mut s = self.last_state.clone();
        for _ in 0..steps {
            s = self.ssm.f.dot(&s);
            let y_hat = self.ssm.h.dot(&s);
            forecasts.push(y_hat[0]);
        }
        forecasts
    }
}

impl fmt::Display for UCResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Unobserved Components Model ")?;
        writeln!(f, "{:<25} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<25} {:>10?}", "Level type:", self.level_type)?;
        writeln!(f, "{:<25} {:>10?}", "Seasonal type:", self.seasonal_type)?;
        writeln!(
            f,
            "{:<25} {:>10.4}",
            "Log-likelihood:", self.log_likelihood
        )?;
        writeln!(f, "{:<25} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<25} {:>10.4}", "BIC:", self.bic)?;
        writeln!(f, "\n{:-^60}", " Estimated Parameters ")?;
        for (name, val) in self.param_names.iter().zip(self.params.iter()) {
            writeln!(f, "  {:<30} {:>12.6}", name, val)?;
        }
        writeln!(f, "{:=^60}", "")
    }
}

/// Numerical gradient via central differences.
fn numerical_gradient<F: Fn(&[f64]) -> f64>(f: &F, params: &[f64], eps: f64) -> Vec<f64> {
    let n = params.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[i] += eps;
        p_minus[i] -= eps;
        grad[i] = (f(&p_plus) - f(&p_minus)) / (2.0 * eps);
    }
    grad
}

/// BFGS optimizer (similar to garch.rs).
fn optimize(
    neg_ll: impl Fn(&[f64]) -> f64,
    init: &[f64],
    max_iter: usize,
    constrain: impl Fn(&mut [f64]),
) -> (Vec<f64>, bool) {
    let n = init.len();
    let mut params = init.to_vec();
    constrain(&mut params);
    let mut best_val = neg_ll(&params);
    let mut best_params = params.clone();

    let mut inv_hess = vec![vec![0.0; n]; n];
    for (i, row) in inv_hess.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    let mut prev_grad = numerical_gradient(&neg_ll, &params, 1e-5);

    for iter in 0..max_iter {
        let grad = if iter == 0 {
            prev_grad.clone()
        } else {
            numerical_gradient(&neg_ll, &params, 1e-5)
        };
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-6 {
            return (params, true);
        }

        let direction: Vec<f64> = (0..n)
            .map(|i| -(0..n).map(|j| inv_hess[i][j] * grad[j]).sum::<f64>())
            .collect();

        let mut step = 1.0;
        let mut improved = false;
        let slope: f64 = direction.iter().zip(grad.iter()).map(|(d, g)| d * g).sum();

        for _ in 0..30 {
            let mut candidate: Vec<f64> = params
                .iter()
                .zip(direction.iter())
                .map(|(p, d)| p + step * d)
                .collect();
            constrain(&mut candidate);
            let val = neg_ll(&candidate);
            if val.is_finite() && val < best_val + 1e-4 * step * slope {
                let new_grad = numerical_gradient(&neg_ll, &candidate, 1e-5);
                let s: Vec<f64> = candidate
                    .iter()
                    .zip(params.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let yk: Vec<f64> = new_grad
                    .iter()
                    .zip(grad.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let sy: f64 = s.iter().zip(yk.iter()).map(|(a, b)| a * b).sum();

                if sy > 1e-10 {
                    let hy: Vec<f64> = (0..n)
                        .map(|i| (0..n).map(|j| inv_hess[i][j] * yk[j]).sum::<f64>())
                        .collect();
                    let yhy: f64 = yk.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();

                    for (i, row) in inv_hess.iter_mut().enumerate() {
                        for (j, cell) in row.iter_mut().enumerate() {
                            *cell += (sy + yhy) * s[i] * s[j] / (sy * sy)
                                - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                        }
                    }
                }

                prev_grad = new_grad;
                best_val = val;
                best_params = candidate.clone();
                params = candidate;
                improved = true;
                break;
            }
            step *= 0.5;
        }

        if !improved {
            let mut candidate: Vec<f64> = params
                .iter()
                .zip(grad.iter())
                .map(|(p, g)| p - 1e-6 * g)
                .collect();
            constrain(&mut candidate);
            let val = neg_ll(&candidate);
            if val < best_val && val.is_finite() {
                best_val = val;
                best_params = candidate.clone();
                params = candidate;
                for (i, row) in inv_hess.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate() {
                        *cell = if i == j { 1.0 } else { 0.0 };
                    }
                }
            } else {
                return (best_params, true);
            }
        }
    }

    (best_params, true)
}

impl UnobservedComponents {
    /// Fit an Unobserved Components model to univariate time series data.
    pub fn fit(
        y: &Array1<f64>,
        level: UCLevel,
        seasonal: UCSeasonal,
    ) -> Result<UCResult, GreenersError> {
        let n = y.len();
        if n < 3 {
            return Err(GreenersError::InvalidOperation("Not enough data".to_string()));
        }

        // Determine state dimension and parameter layout
        let has_trend = matches!(level, UCLevel::LocalLinearTrend | UCLevel::SmoothTrend);
        let has_obs_noise = !matches!(level, UCLevel::RandomWalk);
        let seasonal_period = match &seasonal {
            UCSeasonal::None => 0,
            UCSeasonal::Deterministic(p) | UCSeasonal::Stochastic(p) => *p,
        };
        let has_seasonal = seasonal_period > 0;
        let stochastic_seasonal = matches!(seasonal, UCSeasonal::Stochastic(_));

        let n_seasonal_states = if has_seasonal { seasonal_period - 1 } else { 0 };
        let level_states = if has_trend { 2 } else { 1 };
        let n_states = level_states + n_seasonal_states;

        // Parameter indices: [sigma2_irregular, sigma2_level, sigma2_trend, sigma2_seasonal]
        // Only include parameters that are free
        let mut param_names = Vec::new();
        if has_obs_noise {
            param_names.push("sigma2.irregular".to_string());
        }
        // Level disturbance: not present for SmoothTrend
        let has_level_disturbance = !matches!(level, UCLevel::SmoothTrend);
        if has_level_disturbance {
            param_names.push("sigma2.level".to_string());
        }
        if has_trend {
            param_names.push("sigma2.trend".to_string());
        }
        if stochastic_seasonal {
            param_names.push("sigma2.seasonal".to_string());
        }
        let n_params = param_names.len();

        // Initial variance estimate from data
        let y_var = {
            let mean = y.mean().unwrap_or(0.0);
            y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64)
        };
        let init: Vec<f64> = vec![(y_var * 0.1).ln(); n_params];

        // Prepare observations for Kalman filter
        let obs: Vec<Array1<f64>> = y.iter().map(|v| Array1::from_vec(vec![*v])).collect();

        // Build state space model from log-params and evaluate negative log-likelihood
        let build_model = |log_params: &[f64]| -> StateSpaceModel {
            let params: Vec<f64> = log_params.iter().map(|p| p.exp()).collect();
            let mut idx = 0;

            let sigma2_irr = if has_obs_noise {
                let v = params[idx];
                idx += 1;
                v
            } else {
                0.0
            };
            let sigma2_level = if has_level_disturbance {
                let v = params[idx];
                idx += 1;
                v
            } else {
                0.0
            };
            let sigma2_trend = if has_trend {
                let v = params[idx];
                idx += 1;
                v
            } else {
                0.0
            };
            let sigma2_seasonal = if stochastic_seasonal {
                let v = params[idx];
                let _ = idx + 1;
                v
            } else {
                0.0
            };

            // Transition matrix F
            let mut f_mat = Array2::<f64>::zeros((n_states, n_states));
            // Level block
            f_mat[[0, 0]] = 1.0;
            if has_trend {
                f_mat[[0, 1]] = 1.0; // level gets trend
                f_mat[[1, 1]] = 1.0; // trend persists
            }
            // Seasonal block: rotation matrix
            if has_seasonal {
                let s_start = level_states;
                // First seasonal state = -sum of others
                for j in 0..n_seasonal_states {
                    f_mat[[s_start, s_start + j]] = -1.0;
                }
                // Shift: s_{i+1,t} = s_{i,t-1}
                for j in 1..n_seasonal_states {
                    f_mat[[s_start + j, s_start + j - 1]] = 1.0;
                }
            }

            // Observation matrix H: [1, 0, ..., seasonal_1, 0, ...]
            let mut h_mat = Array2::<f64>::zeros((1, n_states));
            h_mat[[0, 0]] = 1.0;
            if has_seasonal {
                h_mat[[0, level_states]] = 1.0;
            }

            // State noise selection matrix R and covariance Q
            // Determine number of shocks
            let mut n_shocks = 0;
            if has_level_disturbance {
                n_shocks += 1;
            }
            if has_trend {
                n_shocks += 1;
            }
            if stochastic_seasonal {
                n_shocks += 1;
            }
            // If no shocks at all, add a dummy zero shock
            let n_shocks = n_shocks.max(1);

            let mut r_mat = Array2::<f64>::zeros((n_states, n_shocks));
            let mut q_mat = Array2::<f64>::zeros((n_shocks, n_shocks));

            let mut shock_idx = 0;
            if has_level_disturbance {
                r_mat[[0, shock_idx]] = 1.0;
                q_mat[[shock_idx, shock_idx]] = sigma2_level;
                shock_idx += 1;
            }
            if has_trend {
                r_mat[[1, shock_idx]] = 1.0;
                q_mat[[shock_idx, shock_idx]] = sigma2_trend;
                shock_idx += 1;
            }
            if stochastic_seasonal {
                r_mat[[level_states, shock_idx]] = 1.0;
                q_mat[[shock_idx, shock_idx]] = sigma2_seasonal;
            }

            // Observation noise
            let r_obs = Array2::from_elem((1, 1), sigma2_irr);

            // Initial state
            let s0 = Array1::zeros(n_states);
            let p0 = Array2::from_diag(&Array1::from_elem(n_states, y_var * 10.0));

            StateSpaceModel {
                h: h_mat,
                f: f_mat,
                r: r_mat,
                q: q_mat,
                r_obs,
                s0,
                p0,
            }
        };

        let neg_ll = |log_params: &[f64]| -> f64 {
            let model = build_model(log_params);
            match KalmanFilter::filter(&model, &obs) {
                Ok(res) => -res.log_likelihood,
                Err(_) => 1e18,
            }
        };

        let constrain = |_params: &mut [f64]| {
            // log-parameterization ensures positivity; no extra constraints needed
        };

        let (opt_log_params, _converged) = optimize(neg_ll, &init, 500, constrain);

        // Build final model and run filter + smoother
        let final_model = build_model(&opt_log_params);
        let filter_result = KalmanFilter::filter(&final_model, &obs)?;
        let smooth_result = KalmanSmoother::smooth(&final_model, &filter_result)?;

        let ll = filter_result.log_likelihood;
        let aic = -2.0 * ll + 2.0 * n_params as f64;
        let bic = -2.0 * ll + (n_params as f64) * (n as f64).ln();

        // Extract components from smoothed states
        let mut level_comp = Array1::zeros(n);
        let mut trend_comp = if has_trend {
            Some(Array1::zeros(n))
        } else {
            Option::None
        };
        let mut seasonal_comp = if has_seasonal {
            Some(Array1::zeros(n))
        } else {
            Option::None
        };

        for t in 0..n {
            let st = &smooth_result.smoothed_states[t];
            level_comp[t] = st[0];
            if let Some(ref mut tr) = trend_comp {
                tr[t] = st[1];
            }
            if let Some(ref mut se) = seasonal_comp {
                se[t] = st[level_states];
            }
        }

        // Residuals
        let mut fitted = level_comp.clone();
        if let Some(ref tr) = trend_comp {
            fitted = &fitted + tr;
        }
        if let Some(ref se) = seasonal_comp {
            fitted = &fitted + se;
        }
        let residuals = y - &fitted;

        // Collect actual variance params
        let final_params: Vec<f64> = opt_log_params.iter().map(|p| p.exp()).collect();

        let last_state = filter_result
            .filtered_states
            .last()
            .cloned()
            .unwrap_or_else(|| Array1::zeros(n_states));

        Ok(UCResult {
            level: level_comp,
            trend: trend_comp,
            seasonal: seasonal_comp,
            residuals,
            params: final_params,
            param_names,
            log_likelihood: ll,
            aic,
            bic,
            n_obs: n,
            level_type: level,
            seasonal_type: seasonal,
            ssm: final_model,
            last_state,
        })
    }
}
