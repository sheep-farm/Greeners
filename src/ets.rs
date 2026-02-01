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
