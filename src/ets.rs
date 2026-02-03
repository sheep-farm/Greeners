use crate::GreenersError;
use ndarray::Array1;
use std::fmt;

/// Exponential smoothing result.
#[derive(Debug, Clone)]
pub struct ETSResult {
    pub level: Array1<f64>,
    pub trend: Array1<f64>,
    pub seasonal: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub residuals: Array1<f64>,
    pub alpha: f64,
    pub beta: Option<f64>,
    pub gamma: Option<f64>,
    pub phi: Option<f64>,
    pub sse: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub last_level: f64,
    pub last_trend: f64,
    pub last_seasonal: Array1<f64>,
    pub seasonal_periods: usize,
    pub trend_type: String,
    pub seasonal_type: String,
    pub damped: bool,
}

impl ETSResult {
    /// Forecast `steps` ahead.
    pub fn predict(&self, steps: usize) -> Array1<f64> {
        let mut forecasts = Array1::<f64>::zeros(steps);
        let m = self.seasonal_periods;
        let phi = self.phi.unwrap_or(1.0);

        for h in 0..steps {
            let h1 = (h + 1) as f64;

            // Damped cumulative factor
            let phi_h = if self.damped {
                let mut s = 0.0;
                let mut p = 1.0;
                for _ in 0..h + 1 {
                    p *= phi;
                    s += p;
                }
                s
            } else {
                h1
            };

            let level = self.last_level;
            let trend_component = match self.trend_type.as_str() {
                "add" => self.last_trend * phi_h,
                "mul" => self.last_trend.powf(phi_h),
                _ => 0.0,
            };

            let seasonal_idx = if m > 0 { h % m } else { 0 };
            let s_val = if m > 0 {
                self.last_seasonal[seasonal_idx]
            } else {
                0.0
            };

            forecasts[h] = match (self.trend_type.as_str(), self.seasonal_type.as_str()) {
                ("none", "none") => level,
                ("add", "none") => level + trend_component,
                ("mul", "none") => level * trend_component,
                ("none", "add") => level + s_val,
                ("add", "add") => level + trend_component + s_val,
                ("add", "mul") => (level + trend_component) * s_val,
                ("mul", "add") => level * trend_component + s_val,
                ("mul", "mul") => level * trend_component * s_val,
                _ => level,
            };
        }

        forecasts
    }
}

impl fmt::Display for ETSResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let model_name = format!(
            "ETS({},{},{}{})",
            "A", // error is always additive in this implementation
            match self.trend_type.as_str() {
                "add" => "A",
                "mul" => "M",
                _ => "N",
            },
            match self.seasonal_type.as_str() {
                "add" => "A",
                "mul" => "M",
                _ => "N",
            },
            if self.damped { "d" } else { "" }
        );
        writeln!(f, "\n{:=^60}", format!(" {} ", model_name))?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.6}", "Alpha:", self.alpha)?;
        if let Some(b) = self.beta {
            writeln!(f, "{:<20} {:>10.6}", "Beta:", b)?;
        }
        if let Some(g) = self.gamma {
            writeln!(f, "{:<20} {:>10.6}", "Gamma:", g)?;
        }
        if let Some(p) = self.phi {
            writeln!(f, "{:<20} {:>10.6}", "Phi:", p)?;
        }
        writeln!(f, "{:<20} {:>10.4}", "SSE:", self.sse)?;
        writeln!(f, "{:<20} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>10.4}", "BIC:", self.bic)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Exponential smoothing models (Holt-Winters family).
pub struct ExponentialSmoothing;

impl ExponentialSmoothing {
    /// Fit an exponential smoothing model.
    ///
    /// * `y` — time series data
    /// * `trend` — `None`, `Some("add")`, or `Some("mul")`
    /// * `seasonal` — `None`, `Some("add")`, or `Some("mul")`
    /// * `seasonal_periods` — number of periods in a seasonal cycle (ignored if seasonal is None)
    /// * `damped` — whether to use damped trend
    pub fn fit(
        y: &Array1<f64>,
        trend: Option<&str>,
        seasonal: Option<&str>,
        seasonal_periods: usize,
        damped: bool,
    ) -> Result<ETSResult, GreenersError> {
        let n = y.len();
        let m = if seasonal.is_some() {
            seasonal_periods
        } else {
            0
        };

        if n < 4 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for exponential smoothing".into(),
            ));
        }
        if seasonal.is_some() && (m < 2 || n < 2 * m) {
            return Err(GreenersError::ShapeMismatch(
                "Need at least 2 full seasonal periods".into(),
            ));
        }

        let trend_type = trend.unwrap_or("none").to_string();
        let seasonal_type = seasonal.unwrap_or("none").to_string();

        // Grid search for optimal parameters
        let steps: Vec<f64> = (1..=19).map(|i| i as f64 * 0.05).collect(); // 0.05 to 0.95
        let phi_steps: Vec<f64> = if damped {
            (80..=99).map(|i| i as f64 * 0.01).collect()
        } else {
            vec![1.0]
        };

        let mut best_sse = f64::MAX;
        let mut best_params = (0.1, 0.01, 0.01, 1.0);

        let need_beta = trend.is_some();
        let need_gamma = seasonal.is_some();

        let beta_range: Vec<f64> = if need_beta { steps.clone() } else { vec![0.0] };
        let gamma_range: Vec<f64> = if need_gamma { steps.clone() } else { vec![0.0] };

        // Coarse grid search: sample a subset
        let coarse_alpha: Vec<f64> = (1..=9).map(|i| i as f64 * 0.1).collect();
        let coarse_beta: Vec<f64> = if need_beta {
            (1..=5).map(|i| i as f64 * 0.1).collect()
        } else {
            vec![0.0]
        };
        let coarse_gamma: Vec<f64> = if need_gamma {
            (1..=5).map(|i| i as f64 * 0.1).collect()
        } else {
            vec![0.0]
        };
        let coarse_phi: Vec<f64> = if damped {
            vec![0.85, 0.90, 0.95, 0.98]
        } else {
            vec![1.0]
        };

        for &a in &coarse_alpha {
            for &b in &coarse_beta {
                for &g in &coarse_gamma {
                    for &p in &coarse_phi {
                        let sse = compute_sse(
                            y,
                            &ETSParams {
                                alpha: a,
                                beta: b,
                                gamma: g,
                                phi: p,
                                trend_type: &trend_type,
                                seasonal_type: &seasonal_type,
                                m,
                            },
                        );
                        if sse < best_sse {
                            best_sse = sse;
                            best_params = (a, b, g, p);
                        }
                    }
                }
            }
        }

        // Fine grid: search around best coarse parameters
        let refine = |center: f64, range: &[f64]| -> Vec<f64> {
            range
                .iter()
                .copied()
                .filter(|&v| (v - center).abs() < 0.15 && v > 0.01 && v < 0.99)
                .collect::<Vec<f64>>()
        };

        let fine_alpha = refine(best_params.0, &steps);
        let fine_beta = if need_beta {
            refine(best_params.1, &beta_range)
        } else {
            vec![0.0]
        };
        let fine_gamma = if need_gamma {
            refine(best_params.2, &gamma_range)
        } else {
            vec![0.0]
        };
        let fine_phi = if damped {
            refine(best_params.3, &phi_steps)
        } else {
            vec![1.0]
        };

        let fa = if fine_alpha.is_empty() {
            coarse_alpha
        } else {
            fine_alpha
        };
        let fb = if fine_beta.is_empty() {
            coarse_beta
        } else {
            fine_beta
        };
        let fg = if fine_gamma.is_empty() {
            coarse_gamma
        } else {
            fine_gamma
        };
        let fp = if fine_phi.is_empty() {
            coarse_phi
        } else {
            fine_phi
        };

        for &a in &fa {
            for &b in &fb {
                for &g in &fg {
                    for &p in &fp {
                        let sse = compute_sse(
                            y,
                            &ETSParams {
                                alpha: a,
                                beta: b,
                                gamma: g,
                                phi: p,
                                trend_type: &trend_type,
                                seasonal_type: &seasonal_type,
                                m,
                            },
                        );
                        if sse < best_sse {
                            best_sse = sse;
                            best_params = (a, b, g, p);
                        }
                    }
                }
            }
        }

        let (alpha, beta, gamma, phi) = best_params;

        // Run final model with best parameters to get states
        let (level, trend_arr, seasonal_arr, fitted, residuals) = run_ets(
            y,
            &ETSParams {
                alpha,
                beta,
                gamma,
                phi,
                trend_type: &trend_type,
                seasonal_type: &seasonal_type,
                m,
            },
        );

        let n_params = 1
            + if need_beta { 1 } else { 0 }
            + if need_gamma { 1 } else { 0 }
            + if damped { 1 } else { 0 }
            + 1 // initial level
            + if need_beta { 1 } else { 0 } // initial trend
            + if need_gamma { m } else { 0 }; // initial seasonal

        let nf = n as f64;
        let aic = nf * (best_sse / nf).ln() + 2.0 * n_params as f64;
        let bic = nf * (best_sse / nf).ln() + n_params as f64 * nf.ln();

        let last_level = level[n - 1];
        let last_trend = if !trend_arr.is_empty() {
            trend_arr[n - 1]
        } else {
            0.0
        };
        let last_seasonal = if m > 0 {
            // Last m seasonal values
            Array1::from_vec((0..m).map(|j| seasonal_arr[n - m + j]).collect())
        } else {
            Array1::zeros(0)
        };

        Ok(ETSResult {
            level,
            trend: trend_arr,
            seasonal: seasonal_arr,
            fitted_values: fitted,
            residuals,
            alpha,
            beta: if need_beta { Some(beta) } else { None },
            gamma: if need_gamma { Some(gamma) } else { None },
            phi: if damped { Some(phi) } else { None },
            sse: best_sse,
            aic,
            bic,
            n_obs: n,
            last_level,
            last_trend,
            last_seasonal,
            seasonal_periods: m,
            trend_type,
            seasonal_type,
            damped,
        })
    }
}

struct ETSParams<'a> {
    alpha: f64,
    beta: f64,
    gamma: f64,
    phi: f64,
    trend_type: &'a str,
    seasonal_type: &'a str,
    m: usize,
}

type ETSState = (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
);

fn compute_sse(y: &Array1<f64>, p: &ETSParams) -> f64 {
    let (_, _, _, _, residuals) = run_ets(y, p);
    residuals.iter().map(|r| r * r).sum()
}

#[allow(clippy::too_many_lines)]
fn run_ets(y: &Array1<f64>, ep: &ETSParams) -> ETSState {
    let n = y.len();
    let (alpha, beta, gamma, phi) = (ep.alpha, ep.beta, ep.gamma, ep.phi);
    let (trend_type, seasonal_type, m) = (ep.trend_type, ep.seasonal_type, ep.m);
    let has_trend = trend_type != "none";
    let has_seasonal = seasonal_type != "none" && m > 0;

    let mut level = Array1::<f64>::zeros(n);
    let mut trend_arr = Array1::<f64>::zeros(n);
    let mut seasonal_arr = Array1::<f64>::zeros(n.max(m));
    let mut fitted = Array1::<f64>::zeros(n);
    let mut residuals = Array1::<f64>::zeros(n);

    // Initialize
    if has_seasonal {
        let first_period_mean: f64 = y.iter().take(m).sum::<f64>() / m as f64;
        level[0] = first_period_mean;
        if seasonal_type == "mul" {
            for j in 0..m {
                seasonal_arr[j] = if first_period_mean.abs() > 1e-15 {
                    y[j] / first_period_mean
                } else {
                    1.0
                };
            }
        } else {
            for j in 0..m {
                seasonal_arr[j] = y[j] - first_period_mean;
            }
        }
    } else {
        level[0] = y[0];
    }

    if has_trend {
        if n > 1 && has_seasonal && m > 0 {
            if trend_type == "mul" {
                trend_arr[0] = if level[0].abs() > 1e-15 {
                    (y[m.min(n - 1)] / level[0]).powf(1.0 / m as f64)
                } else {
                    1.0
                };
            } else {
                trend_arr[0] = (y[m.min(n - 1)] - y[0]) / m as f64;
            }
        } else if n > 1 {
            if trend_type == "mul" {
                trend_arr[0] = if y[0].abs() > 1e-15 { y[1] / y[0] } else { 1.0 };
            } else {
                trend_arr[0] = y[1] - y[0];
            }
        }
    }

    fitted[0] = ets_point(
        level[0],
        trend_arr[0],
        seasonal_arr[0],
        trend_type,
        seasonal_type,
        has_trend,
        has_seasonal,
    );
    residuals[0] = y[0] - fitted[0];

    for t in 1..n {
        let l_prev = level[t - 1];
        let b_prev = trend_arr[t - 1];
        let s_prev = if has_seasonal && t >= m {
            seasonal_arr[t - m]
        } else if has_seasonal {
            seasonal_arr[t]
        } else {
            0.0
        };

        let f_val = match (trend_type, seasonal_type) {
            ("none", "none") => l_prev,
            ("add", "none") => l_prev + phi * b_prev,
            ("mul", "none") => l_prev * b_prev.powf(phi),
            ("none", "add") => l_prev + s_prev,
            ("add", "add") => l_prev + phi * b_prev + s_prev,
            ("add", "mul") => (l_prev + phi * b_prev) * s_prev,
            ("mul", "add") => l_prev * b_prev.powf(phi) + s_prev,
            ("mul", "mul") => l_prev * b_prev.powf(phi) * s_prev,
            _ => l_prev,
        };
        fitted[t] = f_val;
        residuals[t] = y[t] - f_val;

        let y_t = y[t];
        let new_level = match (trend_type, seasonal_type) {
            ("none", "none") => alpha * y_t + (1.0 - alpha) * l_prev,
            ("add", "none") => alpha * y_t + (1.0 - alpha) * (l_prev + phi * b_prev),
            ("mul", "none") => alpha * y_t + (1.0 - alpha) * l_prev * b_prev.powf(phi),
            ("none", "add") => alpha * (y_t - s_prev) + (1.0 - alpha) * l_prev,
            ("add", "add") => alpha * (y_t - s_prev) + (1.0 - alpha) * (l_prev + phi * b_prev),
            ("add", "mul") if s_prev.abs() > 1e-15 => {
                alpha * (y_t / s_prev) + (1.0 - alpha) * (l_prev + phi * b_prev)
            }
            ("add", "mul") => alpha * y_t + (1.0 - alpha) * (l_prev + phi * b_prev),
            ("mul", "add") => alpha * (y_t - s_prev) + (1.0 - alpha) * l_prev * b_prev.powf(phi),
            ("mul", "mul") if s_prev.abs() > 1e-15 => {
                alpha * (y_t / s_prev) + (1.0 - alpha) * l_prev * b_prev.powf(phi)
            }
            ("mul", "mul") => alpha * y_t + (1.0 - alpha) * l_prev * b_prev.powf(phi),
            _ => alpha * y_t + (1.0 - alpha) * l_prev,
        };
        level[t] = new_level;

        if has_trend {
            trend_arr[t] = match trend_type {
                "add" => beta * (new_level - l_prev) + (1.0 - beta) * phi * b_prev,
                "mul" if l_prev.abs() > 1e-15 => {
                    beta * (new_level / l_prev) + (1.0 - beta) * b_prev.powf(phi)
                }
                "mul" => b_prev,
                _ => 0.0,
            };
        }

        if has_seasonal {
            let new_s = match seasonal_type {
                "add" => gamma * (y_t - new_level) + (1.0 - gamma) * s_prev,
                "mul" if new_level.abs() > 1e-15 => {
                    gamma * (y_t / new_level) + (1.0 - gamma) * s_prev
                }
                "mul" => s_prev,
                _ => 0.0,
            };
            seasonal_arr[t] = new_s;
        }
    }

    (
        level,
        trend_arr,
        seasonal_arr.slice(ndarray::s![..n]).to_owned(),
        fitted,
        residuals,
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full ETS (Error-Trend-Seasonal) framework with MLE estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Error component type for ETS models.
#[derive(Debug, Clone, PartialEq)]
pub enum ETSError {
    Additive,
    Multiplicative,
}

/// Trend component type for ETS models.
#[derive(Debug, Clone, PartialEq)]
pub enum ETSTrend {
    None,
    Additive,
    AdditiveDamped,
    Multiplicative,
    MultiplicativeDamped,
}

/// Seasonal component type for ETS models.
#[derive(Debug, Clone, PartialEq)]
pub enum ETSSeasonal {
    None,
    Additive(usize),
    Multiplicative(usize),
}

/// Full ETS model estimator using MLE.
pub struct ETSModel;

/// Result of fitting an ETS model.
#[derive(Debug)]
pub struct ETSModelResult {
    pub error: ETSError,
    pub trend: ETSTrend,
    pub seasonal: ETSSeasonal,
    pub alpha: f64,
    pub beta: Option<f64>,
    pub gamma: Option<f64>,
    pub phi: Option<f64>,
    pub level: Array1<f64>,
    pub trend_component: Option<Array1<f64>>,
    pub seasonal_component: Option<Array1<f64>>,
    pub fitted: Array1<f64>,
    pub residuals: Array1<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    // Store last states for forecasting
    last_level: f64,
    last_trend: Option<f64>,
    last_seasonal: Vec<f64>,
    seasonal_period: usize,
}

impl ETSModelResult {
    /// Forecast `steps` ahead.
    pub fn predict(&self, steps: usize) -> Array1<f64> {
        let mut forecasts = Array1::<f64>::zeros(steps);
        let m = self.seasonal_period;
        let has_trend = self.trend != ETSTrend::None;
        let damped = matches!(
            self.trend,
            ETSTrend::AdditiveDamped | ETSTrend::MultiplicativeDamped
        );
        let mul_trend = matches!(
            self.trend,
            ETSTrend::Multiplicative | ETSTrend::MultiplicativeDamped
        );
        let mul_seasonal = matches!(self.seasonal, ETSSeasonal::Multiplicative(_));
        let has_seasonal = m > 0;
        let phi = self.phi.unwrap_or(1.0);
        let l = self.last_level;
        let b = self.last_trend.unwrap_or(if mul_trend { 1.0 } else { 0.0 });

        for h in 0..steps {
            let j = h + 1;
            let phi_sum = if damped {
                (1..=j).fold(0.0, |acc, i| acc + phi.powi(i as i32))
            } else {
                j as f64
            };

            let trend_val = if !has_trend {
                if mul_trend {
                    1.0
                } else {
                    0.0
                }
            } else if mul_trend {
                b.powf(phi_sum)
            } else {
                b * phi_sum
            };

            let s_val = if has_seasonal {
                let idx = h % m;
                self.last_seasonal[idx]
            } else if mul_seasonal {
                1.0
            } else {
                0.0
            };

            forecasts[h] = match (mul_trend, mul_seasonal) {
                (false, false) => l + trend_val + s_val,
                (false, true) => (l + trend_val) * s_val,
                (true, false) => l * trend_val + s_val,
                (true, true) => l * trend_val * s_val,
            };
        }
        forecasts
    }
}

impl fmt::Display for ETSModelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let e = match self.error {
            ETSError::Additive => "A",
            ETSError::Multiplicative => "M",
        };
        let t = match self.trend {
            ETSTrend::None => "N",
            ETSTrend::Additive => "A",
            ETSTrend::AdditiveDamped => "Ad",
            ETSTrend::Multiplicative => "M",
            ETSTrend::MultiplicativeDamped => "Md",
        };
        let s = match self.seasonal {
            ETSSeasonal::None => "N",
            ETSSeasonal::Additive(_) => "A",
            ETSSeasonal::Multiplicative(_) => "M",
        };
        let name = format!(" ETS({},{},{}) ", e, t, s);
        writeln!(f, "\n{:=^60}", name)?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.6}", "Alpha:", self.alpha)?;
        if let Some(b) = self.beta {
            writeln!(f, "{:<20} {:>10.6}", "Beta:", b)?;
        }
        if let Some(g) = self.gamma {
            writeln!(f, "{:<20} {:>10.6}", "Gamma:", g)?;
        }
        if let Some(p) = self.phi {
            writeln!(f, "{:<20} {:>10.6}", "Phi:", p)?;
        }
        writeln!(f, "{:<20} {:>10.4}", "Log-Likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>10.4}", "BIC:", self.bic)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Return type for ets_full_recursion:
/// (level, trend, seasonal, fitted, errors, neg_log_likelihood).
type EtsRecursionResult = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64);

/// Internal: run the ETS recursion for the full framework.
/// `params` layout: [alpha, (beta), (gamma), (phi), l0, (b0), (s0..s_{m-1})]
#[allow(clippy::too_many_arguments)]
fn ets_full_recursion(
    y: &[f64],
    params: &[f64],
    mul_error: bool,
    has_trend: bool,
    mul_trend: bool,
    damped: bool,
    has_seasonal: bool,
    mul_seasonal: bool,
    m: usize,
) -> EtsRecursionResult {
    let n = y.len();
    let mut idx = 0;
    let alpha = params[idx];
    idx += 1;
    let beta = if has_trend {
        let v = params[idx];
        idx += 1;
        v
    } else {
        0.0
    };
    let gamma = if has_seasonal {
        let v = params[idx];
        idx += 1;
        v
    } else {
        0.0
    };
    let phi = if damped {
        let v = params[idx];
        idx += 1;
        v
    } else {
        1.0
    };
    let l0 = params[idx];
    idx += 1;
    let b0 = if has_trend {
        let v = params[idx];
        idx += 1;
        v
    } else if mul_trend {
        1.0
    } else {
        0.0
    };
    let mut s_init = vec![if mul_seasonal { 1.0 } else { 0.0 }; m.max(1)];
    if has_seasonal {
        s_init[..m].copy_from_slice(&params[idx..idx + m]);
    }

    let mut level = vec![0.0; n];
    let mut trend_v = vec![0.0; n];
    let mut seasonal_v = vec![if mul_seasonal { 1.0 } else { 0.0 }; n + m];
    let mut fitted = vec![0.0; n];
    let mut errors = vec![0.0; n];

    // Place initial seasonal values
    if has_seasonal {
        seasonal_v[..m].copy_from_slice(&s_init[..m]);
    }

    let mut l_prev = l0;
    let mut b_prev = b0;
    let mut neg_ll = 0.0;

    for t in 0..n {
        let s_prev = if has_seasonal {
            seasonal_v[t]
        } else if mul_seasonal {
            1.0
        } else {
            0.0
        };

        // One-step-ahead forecast (mean)
        let mu = match (mul_trend, mul_seasonal) {
            (false, false) => l_prev + phi * b_prev + s_prev,
            (false, true) => (l_prev + phi * b_prev) * s_prev,
            (true, false) => l_prev * b_prev.powf(phi) + s_prev,
            (true, true) => l_prev * b_prev.powf(phi) * s_prev,
        };
        fitted[t] = mu;

        let e_t = if mul_error {
            if mu.abs() < 1e-15 {
                return (level, trend_v, seasonal_v, fitted, errors, f64::MAX);
            }
            (y[t] - mu) / mu
        } else {
            y[t] - mu
        };
        errors[t] = e_t;

        // Update level
        let new_l = if mul_error {
            match (mul_trend, mul_seasonal) {
                (false, false) => (l_prev + phi * b_prev) * (1.0 + alpha * e_t),
                (false, true) => (l_prev + phi * b_prev) * (1.0 + alpha * e_t),
                (true, false) => l_prev * b_prev.powf(phi) * (1.0 + alpha * e_t),
                (true, true) => l_prev * b_prev.powf(phi) * (1.0 + alpha * e_t),
            }
        } else {
            match (mul_trend, mul_seasonal) {
                (false, false) => alpha * (y[t] - s_prev) + (1.0 - alpha) * (l_prev + phi * b_prev),
                (false, true) => {
                    if s_prev.abs() > 1e-15 {
                        alpha * (y[t] / s_prev) + (1.0 - alpha) * (l_prev + phi * b_prev)
                    } else {
                        return (level, trend_v, seasonal_v, fitted, errors, f64::MAX);
                    }
                }
                (true, false) => {
                    alpha * (y[t] - s_prev) + (1.0 - alpha) * l_prev * b_prev.powf(phi)
                }
                (true, true) => {
                    if s_prev.abs() > 1e-15 {
                        alpha * (y[t] / s_prev) + (1.0 - alpha) * l_prev * b_prev.powf(phi)
                    } else {
                        return (level, trend_v, seasonal_v, fitted, errors, f64::MAX);
                    }
                }
            }
        };

        // Update trend
        let new_b = if has_trend {
            if mul_error {
                if mul_trend {
                    if l_prev.abs() > 1e-15 {
                        b_prev.powf(phi) * (1.0 + beta * e_t) * new_l
                            / (l_prev * b_prev.powf(phi) + 1e-30)
                            * (l_prev * b_prev.powf(phi))
                            / new_l
                    } else {
                        b_prev
                    }
                } else {
                    phi * b_prev + beta * (new_l - l_prev - phi * b_prev)
                }
            } else if mul_trend {
                if l_prev.abs() > 1e-15 {
                    beta * (new_l / l_prev) + (1.0 - beta) * b_prev.powf(phi)
                } else {
                    b_prev
                }
            } else {
                beta * (new_l - l_prev) + (1.0 - beta) * phi * b_prev
            }
        } else {
            b_prev
        };

        // Update seasonal
        if has_seasonal {
            let new_s = if mul_error {
                if mul_seasonal {
                    s_prev * (1.0 + gamma * e_t)
                } else {
                    s_prev + gamma * mu * e_t
                }
            } else if mul_seasonal {
                if new_l.abs() > 1e-15 {
                    gamma * (y[t] / new_l) + (1.0 - gamma) * s_prev
                } else {
                    s_prev
                }
            } else {
                gamma * (y[t] - new_l) + (1.0 - gamma) * s_prev
            };
            seasonal_v[t + m] = new_s;
        }

        level[t] = new_l;
        trend_v[t] = new_b;
        l_prev = new_l;
        b_prev = new_b;

        // Accumulate negative log-likelihood
        if mul_error {
            // Multiplicative error: e_t ~ N(0, sigma^2), y_t = mu_t(1+e_t)
            // -ll contribution: 0.5*e_t^2/sigma^2 + log(|mu_t|) (the sigma part handled globally)
            neg_ll += e_t * e_t;
            if mu.abs() < 1e-15 {
                return (level, trend_v, seasonal_v, fitted, errors, f64::MAX);
            }
        } else {
            neg_ll += e_t * e_t;
        }
    }

    // Gaussian log-likelihood: -n/2 * ln(2*pi*sigma^2) - SSE/(2*sigma^2)
    // where sigma^2 = SSE/n. So ll = -n/2*(1 + ln(2*pi*SSE/n))
    let nf = n as f64;
    let sigma2 = neg_ll / nf;
    let ll = if sigma2 > 0.0 {
        -nf / 2.0 * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln())
    } else {
        0.0
    };
    // For multiplicative error, add sum of log|mu_t|
    let ll = if mul_error {
        ll - fitted.iter().map(|m| m.abs().ln()).sum::<f64>()
    } else {
        ll
    };

    (
        level,
        trend_v,
        seasonal_v[..n].to_vec(),
        fitted,
        errors,
        -ll, // return neg log-likelihood
    )
}

fn ets_numerical_gradient(f: &dyn Fn(&[f64]) -> f64, params: &[f64], eps: f64) -> Vec<f64> {
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

fn ets_optimize(
    neg_ll: impl Fn(&[f64]) -> f64,
    init: &[f64],
    max_iter: usize,
    constrain: impl Fn(&mut [f64]),
) -> Vec<f64> {
    let n = init.len();
    let mut params = init.to_vec();
    constrain(&mut params);
    let mut best_val = neg_ll(&params);
    let mut best_params = params.clone();

    let mut inv_hess = vec![vec![0.0; n]; n];
    for (i, row) in inv_hess.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    let mut _prev_grad = ets_numerical_gradient(&neg_ll, &params, 1e-5);

    for iter in 0..max_iter {
        let grad = if iter == 0 {
            _prev_grad.clone()
        } else {
            ets_numerical_gradient(&neg_ll, &params, 1e-5)
        };
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-6 {
            return params;
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
                let new_grad = ets_numerical_gradient(&neg_ll, &candidate, 1e-5);
                let s: Vec<f64> = candidate
                    .iter()
                    .zip(params.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let y_v: Vec<f64> = new_grad
                    .iter()
                    .zip(grad.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let sy: f64 = s.iter().zip(y_v.iter()).map(|(a, b)| a * b).sum();

                if sy > 1e-10 {
                    let hy: Vec<f64> = (0..n)
                        .map(|i| (0..n).map(|j| inv_hess[i][j] * y_v[j]).sum::<f64>())
                        .collect();
                    let yhy: f64 = y_v.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();
                    for (i, row) in inv_hess.iter_mut().enumerate() {
                        for (j, cell) in row.iter_mut().enumerate() {
                            *cell += (sy + yhy) * s[i] * s[j] / (sy * sy)
                                - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                        }
                    }
                }

                _prev_grad = new_grad;
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
                return best_params;
            }
        }

        let param_change: f64 = params
            .iter()
            .zip(best_params.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        if param_change < 1e-8 && iter > 10 {
            return params;
        }
    }
    best_params
}

impl ETSModel {
    /// Fit an ETS model via maximum likelihood estimation.
    pub fn fit(
        y: &Array1<f64>,
        error: ETSError,
        trend: ETSTrend,
        seasonal: ETSSeasonal,
    ) -> Result<ETSModelResult, GreenersError> {
        let n = y.len();
        if n < 4 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for ETS model".into(),
            ));
        }

        let mul_error = error == ETSError::Multiplicative;
        let has_trend = trend != ETSTrend::None;
        let mul_trend = matches!(
            trend,
            ETSTrend::Multiplicative | ETSTrend::MultiplicativeDamped
        );
        let damped = matches!(
            trend,
            ETSTrend::AdditiveDamped | ETSTrend::MultiplicativeDamped
        );
        let (has_seasonal, mul_seasonal, m) = match &seasonal {
            ETSSeasonal::None => (false, false, 0),
            ETSSeasonal::Additive(p) => (true, false, *p),
            ETSSeasonal::Multiplicative(p) => (true, true, *p),
        };

        if has_seasonal && (m < 2 || n < 2 * m) {
            return Err(GreenersError::ShapeMismatch(
                "Need at least 2 full seasonal periods".into(),
            ));
        }

        // Multiplicative error requires positive data
        if mul_error && y.iter().any(|&v| v <= 0.0) {
            return Err(GreenersError::ShapeMismatch(
                "Multiplicative error requires strictly positive data".into(),
            ));
        }

        let y_vec: Vec<f64> = y.to_vec();

        // Build initial parameter vector
        // Layout: [alpha, (beta), (gamma), (phi), l0, (b0), (s0..s_{m-1})]
        let _y_mean = y.iter().sum::<f64>() / n as f64;
        let mut init = Vec::new();
        init.push(0.3); // alpha
        if has_trend {
            init.push(0.05); // beta
        }
        if has_seasonal {
            init.push(0.05); // gamma
        }
        if damped {
            init.push(0.95); // phi
        }

        // Initial level
        let l0 = if has_seasonal {
            y.iter().take(m).sum::<f64>() / m as f64
        } else {
            y[0]
        };
        init.push(l0);

        // Initial trend
        if has_trend {
            let b0 = if mul_trend {
                if has_seasonal && m < n {
                    (y[m.min(n - 1)] / l0.max(1e-10)).powf(1.0 / m as f64)
                } else if n > 1 {
                    (y[1] / y[0].max(1e-10)).max(0.5)
                } else {
                    1.0
                }
            } else if has_seasonal && m < n {
                (y[m.min(n - 1)] - y[0]) / m as f64
            } else if n > 1 {
                y[1] - y[0]
            } else {
                0.0
            };
            init.push(b0);
        }

        // Initial seasonal
        if has_seasonal {
            let first_mean = y.iter().take(m).sum::<f64>() / m as f64;
            for j in 0..m {
                if mul_seasonal {
                    init.push(if first_mean.abs() > 1e-15 {
                        y[j] / first_mean
                    } else {
                        1.0
                    });
                } else {
                    init.push(y[j] - first_mean);
                }
            }
        }

        let n_smooth = 1 + if has_trend { 1 } else { 0 } + if has_seasonal { 1 } else { 0 };
        let n_damp = if damped { 1 } else { 0 };
        let n_init_states = 1 + if has_trend { 1 } else { 0 } + if has_seasonal { m } else { 0 };
        let n_params = n_smooth + n_damp + n_init_states;

        // Constraint function
        let constrain = move |p: &mut [f64]| {
            let mut idx = 0;
            // alpha
            p[idx] = p[idx].clamp(1e-4, 0.9999);
            idx += 1;
            // beta
            if has_trend {
                p[idx] = p[idx].clamp(1e-4, 0.9999);
                idx += 1;
            }
            // gamma
            if has_seasonal {
                p[idx] = p[idx].clamp(1e-4, 0.9999);
                idx += 1;
            }
            // phi
            if damped {
                p[idx] = p[idx].clamp(0.80, 0.98);
                idx += 1;
            }
            // l0 - no constraint beyond finiteness
            idx += 1;
            // b0
            if has_trend {
                if mul_trend {
                    p[idx] = p[idx].clamp(0.1, 5.0);
                }
                idx += 1;
            }
            // seasonal: if multiplicative, keep > 0
            if has_seasonal && mul_seasonal {
                for j in 0..m {
                    p[idx + j] = p[idx + j].max(0.01);
                }
            }
        };

        let neg_ll_fn = {
            let y_ref = y_vec.clone();
            move |p: &[f64]| -> f64 {
                let (_, _, _, _, _, nll) = ets_full_recursion(
                    &y_ref,
                    p,
                    mul_error,
                    has_trend,
                    mul_trend,
                    damped,
                    has_seasonal,
                    mul_seasonal,
                    m,
                );
                if nll.is_finite() {
                    nll
                } else {
                    f64::MAX
                }
            }
        };

        // Also do a coarse grid search on smoothing params to find a better starting point
        let alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9];
        let beta_grid = [0.01, 0.05, 0.1, 0.2];
        let gamma_grid = [0.01, 0.05, 0.1, 0.2];
        let phi_grid = [0.85, 0.90, 0.95, 0.98];

        let mut best_init = init.clone();
        let mut best_nll = neg_ll_fn(&init);

        let beta_vals: &[f64] = if has_trend { &beta_grid } else { &[0.0][..] };
        let gamma_vals: &[f64] = if has_seasonal {
            &gamma_grid
        } else {
            &[0.0][..]
        };
        let phi_vals: &[f64] = if damped { &phi_grid } else { &[1.0][..] };

        for &a in &alpha_grid {
            for &b in beta_vals {
                for &g in gamma_vals {
                    for &p in phi_vals {
                        let mut candidate = init.clone();
                        let mut idx = 0;
                        candidate[idx] = a;
                        idx += 1;
                        if has_trend {
                            candidate[idx] = b;
                            idx += 1;
                        }
                        if has_seasonal {
                            candidate[idx] = g;
                            idx += 1;
                        }
                        if damped {
                            candidate[idx] = p;
                        }
                        let nll = neg_ll_fn(&candidate);
                        if nll.is_finite() && nll < best_nll {
                            best_nll = nll;
                            best_init = candidate;
                        }
                    }
                }
            }
        }

        let best_params = ets_optimize(neg_ll_fn, &best_init, 200, constrain);

        // Run final recursion
        let (level_v, trend_v, seasonal_v, fitted_v, error_v, final_nll) = ets_full_recursion(
            &y_vec,
            &best_params,
            mul_error,
            has_trend,
            mul_trend,
            damped,
            has_seasonal,
            mul_seasonal,
            m,
        );

        let ll = -final_nll;
        let nf = n as f64;
        let k = n_params as f64;
        let aic = -2.0 * ll + 2.0 * k;
        let bic = -2.0 * ll + k * nf.ln();

        // Extract optimized parameters
        let mut idx = 0;
        let alpha = best_params[idx];
        idx += 1;
        let beta_opt = if has_trend {
            let v = best_params[idx];
            idx += 1;
            Some(v)
        } else {
            None
        };
        let gamma_opt = if has_seasonal {
            let v = best_params[idx];
            idx += 1;
            Some(v)
        } else {
            None
        };
        let phi_opt = if damped {
            let v = best_params[idx];
            // idx += 1;
            Some(v)
        } else {
            None
        };

        // Build last seasonal states for forecasting
        let last_seasonal = if has_seasonal && m > 0 {
            // Last m seasonal values
            let sv_len = seasonal_v.len();
            if sv_len >= m {
                seasonal_v[sv_len - m..].to_vec()
            } else {
                seasonal_v.clone()
            }
        } else {
            vec![]
        };

        Ok(ETSModelResult {
            error,
            trend,
            seasonal,
            alpha,
            beta: beta_opt,
            gamma: gamma_opt,
            phi: phi_opt,
            level: Array1::from_vec(level_v.clone()),
            trend_component: if has_trend {
                Some(Array1::from_vec(trend_v.clone()))
            } else {
                None
            },
            seasonal_component: if has_seasonal {
                Some(Array1::from_vec(seasonal_v))
            } else {
                None
            },
            fitted: Array1::from_vec(fitted_v),
            residuals: Array1::from_vec(error_v),
            log_likelihood: ll,
            aic,
            bic,
            n_obs: n,
            last_level: *level_v.last().unwrap(),
            last_trend: if has_trend {
                Some(*trend_v.last().unwrap())
            } else {
                None
            },
            last_seasonal,
            seasonal_period: m,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Original helper functions for ExponentialSmoothing
// ═══════════════════════════════════════════════════════════════════════════════

fn ets_point(
    level: f64,
    trend: f64,
    seasonal: f64,
    trend_type: &str,
    seasonal_type: &str,
    has_trend: bool,
    has_seasonal: bool,
) -> f64 {
    let base = if has_trend {
        match trend_type {
            "add" => level + trend,
            "mul" => level * trend,
            _ => level,
        }
    } else {
        level
    };
    if has_seasonal {
        match seasonal_type {
            "add" => base + seasonal,
            "mul" => base * seasonal,
            _ => base,
        }
    } else {
        base
    }
}
