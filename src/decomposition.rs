use crate::GreenersError;
use ndarray::Array1;
use std::fmt;

/// Result of time series decomposition.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub observed: Array1<f64>,
    pub trend: Array1<f64>,
    pub seasonal: Array1<f64>,
    pub residual: Array1<f64>,
    pub model: String,
}

impl fmt::Display for DecompositionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^60}",
            format!(" Seasonal Decomposition ({}) ", self.model)
        )?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.observed.len())?;
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "Trend mean:",
            self.trend.mean().unwrap_or(f64::NAN)
        )?;
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "Seasonal std:",
            std_dev(&self.seasonal)
        )?;
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "Residual std:",
            std_dev(&self.residual)
        )?;
        writeln!(f, "{:=^60}", "")
    }
}

fn std_dev(arr: &Array1<f64>) -> f64 {
    let valid: Vec<f64> = arr.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.len() < 2 {
        return f64::NAN;
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (valid.len() - 1) as f64;
    var.sqrt()
}

/// Classical seasonal decomposition and STL.
pub struct Decomposition;

impl Decomposition {
    /// Classical seasonal decomposition using moving averages.
    ///
    /// * `series` — the time series
    /// * `period` — seasonal period (e.g., 12 for monthly, 4 for quarterly)
    /// * `model` — `"additive"` or `"multiplicative"`
    pub fn seasonal_decompose(
        series: &Array1<f64>,
        period: usize,
        model: &str,
    ) -> Result<DecompositionResult, GreenersError> {
        let n = series.len();
        if n < 2 * period {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for seasonal decomposition".into(),
            ));
        }
        if period < 2 {
            return Err(GreenersError::ShapeMismatch("Period must be >= 2".into()));
        }

        let multiplicative = model.starts_with('m') || model.starts_with('M');

        // Step 1: Compute trend using centered moving average
        let trend = centered_ma(series, period);

        // Step 2: Detrend
        let detrended = if multiplicative {
            Array1::from_vec(
                (0..n)
                    .map(|i| {
                        if trend[i].is_finite() && trend[i].abs() > 1e-15 {
                            series[i] / trend[i]
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
            )
        } else {
            Array1::from_vec(
                (0..n)
                    .map(|i| {
                        if trend[i].is_finite() {
                            series[i] - trend[i]
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
            )
        };

        // Step 3: Average seasonal component for each period position
        let mut seasonal_avg = vec![0.0f64; period];
        let mut counts = vec![0usize; period];
        for i in 0..n {
            let val = detrended[i];
            if val.is_finite() {
                seasonal_avg[i % period] += val;
                counts[i % period] += 1;
            }
        }
        for p in 0..period {
            if counts[p] > 0 {
                seasonal_avg[p] /= counts[p] as f64;
            }
        }

        // Normalize seasonal: subtract mean (additive) or divide by mean (multiplicative)
        if multiplicative {
            let smean: f64 = seasonal_avg.iter().sum::<f64>() / period as f64;
            if smean.abs() > 1e-15 {
                for v in &mut seasonal_avg {
                    *v /= smean;
                }
            }
        } else {
            let smean: f64 = seasonal_avg.iter().sum::<f64>() / period as f64;
            for v in &mut seasonal_avg {
                *v -= smean;
            }
        }

        let seasonal = Array1::from_vec((0..n).map(|i| seasonal_avg[i % period]).collect());

        // Step 4: Residual
        let residual = if multiplicative {
            Array1::from_vec(
                (0..n)
                    .map(|i| {
                        if trend[i].is_finite() && seasonal[i].abs() > 1e-15 {
                            series[i] / (trend[i] * seasonal[i])
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
            )
        } else {
            Array1::from_vec(
                (0..n)
                    .map(|i| {
                        if trend[i].is_finite() {
                            series[i] - trend[i] - seasonal[i]
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
            )
        };

        Ok(DecompositionResult {
            observed: series.clone(),
            trend,
            seasonal,
            residual,
            model: if multiplicative {
                "multiplicative".to_string()
            } else {
                "additive".to_string()
            },
        })
    }

    /// STL decomposition (Seasonal and Trend decomposition using LOESS).
    ///
    /// * `series` — the time series
    /// * `period` — seasonal period
    /// * `seasonal_window` — LOESS window for seasonal extraction (odd, >= 7)
    /// * `trend_window` — LOESS window for trend extraction (odd, >= period+1). If 0, auto-selected.
    pub fn stl(
        series: &Array1<f64>,
        period: usize,
        seasonal_window: usize,
        trend_window: usize,
    ) -> Result<DecompositionResult, GreenersError> {
        let n = series.len();
        if n < 2 * period {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for STL".into(),
            ));
        }
        if period < 2 {
            return Err(GreenersError::ShapeMismatch("Period must be >= 2".into()));
        }

        let s_win = if seasonal_window < 7 {
            7
        } else {
            seasonal_window | 1
        }; // ensure odd
        let t_win = if trend_window == 0 {
            // Auto: next odd >= ceil(1.5 * period / (1 - 1.5/s_win))
            let tw = (1.5 * period as f64 / (1.0 - 1.5 / s_win as f64)).ceil() as usize;
            tw | 1
        } else {
            trend_window | 1
        };

        let mut seasonal = Array1::<f64>::zeros(n);
        let mut trend = Array1::<f64>::zeros(n);
        let mut weights = Array1::from_elem(n, 1.0f64);

        // Outer robustness loop (2 iterations)
        for _outer in 0..2 {
            // Inner loop (2 iterations)
            for _inner in 0..2 {
                // Step 1: Detrend
                let detrended = series - &trend;

                // Step 2: Cycle-subseries smoothing
                // For each position in the period, extract subseries and smooth with LOESS
                let mut seasonal_raw = Array1::<f64>::zeros(n);
                for p in 0..period {
                    let indices: Vec<usize> = (p..n).step_by(period).collect();
                    let sub_x: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
                    let sub_y: Vec<f64> = indices.iter().map(|&i| detrended[i]).collect();
                    let sub_w: Vec<f64> = indices.iter().map(|&i| weights[i]).collect();

                    let smoothed = loess(&sub_x, &sub_y, &sub_w, &sub_x, s_win);

                    for (j, &idx) in indices.iter().enumerate() {
                        seasonal_raw[idx] = smoothed[j];
                    }
                }

                // Step 3: Low-pass filter on seasonal to remove trend leakage
                // Apply MA(period), MA(period), MA(3), then LOESS
                let lp = moving_average(
                    &moving_average(&moving_average(&seasonal_raw, period), period),
                    3,
                );
                // Smooth the low-pass with LOESS
                let lp_x: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let lp_y: Vec<f64> = lp.iter().copied().collect();
                let lp_w: Vec<f64> = lp
                    .iter()
                    .map(|v| if v.is_finite() { 1.0 } else { 0.0 })
                    .collect();
                let lp_smooth = loess(&lp_x, &lp_y, &lp_w, &lp_x, t_win);

                seasonal = &seasonal_raw - &Array1::from_vec(lp_smooth);

                // Step 4: Deseason and smooth trend
                let deseasoned = series - &seasonal;
                let ds_x: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let ds_y: Vec<f64> = deseasoned.iter().copied().collect();
                let ds_w: Vec<f64> = weights.iter().copied().collect();
                trend = Array1::from_vec(loess(&ds_x, &ds_y, &ds_w, &ds_x, t_win));
            }

            // Outer loop: update robustness weights
            let residual = series - &trend - &seasonal;
            let abs_resid: Vec<f64> = residual.iter().map(|v| v.abs()).collect();
            let mut sorted = abs_resid.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let h = sorted[sorted.len() * 6 / 10]; // ~median * 6

            if h > 1e-15 {
                for i in 0..n {
                    let u = abs_resid[i] / (6.0 * h);
                    weights[i] = if u >= 1.0 { 0.0 } else { (1.0 - u * u).powi(2) };
                }
            }
        }

        let residual = series - &trend - &seasonal;

        Ok(DecompositionResult {
            observed: series.clone(),
            trend,
            seasonal,
            residual,
            model: "STL".to_string(),
        })
    }
}

/// Centered moving average.
fn centered_ma(series: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = series.len();
    let mut result = Array1::from_elem(n, f64::NAN);

    if window % 2 == 1 {
        let half = window / 2;
        for i in half..n - half {
            let sum: f64 = (i - half..=i + half).map(|j| series[j]).sum();
            result[i] = sum / window as f64;
        }
    } else {
        // Even window: 2x(window) centered MA
        let half = window / 2;
        for i in half..n - half {
            let mut sum: f64 = (i - half + 1..i + half).map(|j| series[j]).sum();
            sum += 0.5 * series[i - half] + 0.5 * series[i + half];
            result[i] = sum / window as f64;
        }
    }

    result
}

/// Simple moving average (non-centered, for internal use).
fn moving_average(series: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = series.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    let half = window / 2;

    for i in half..n.saturating_sub(half + if window.is_multiple_of(2) { 1 } else { 0 }) {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let vals: Vec<f64> = (start..end)
            .map(|j| series[j])
            .filter(|v| v.is_finite())
            .collect();
        if !vals.is_empty() {
            result[i] = vals.iter().sum::<f64>() / vals.len() as f64;
        }
    }

    result
}

/// Weighted local regression (LOESS) helper.
///
/// Fits a local linear regression at each point in `x_pred` using
/// the nearest `span` points from `(x, y)` with weights `w`.
fn loess(x: &[f64], y: &[f64], w: &[f64], x_pred: &[f64], span: usize) -> Vec<f64> {
    let n = x.len();
    let h = span.min(n);

    x_pred
        .iter()
        .map(|&xp| {
            // Find distances and sort
            let mut dists: Vec<(usize, f64)> = (0..n).map(|i| (i, (x[i] - xp).abs())).collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let max_dist = dists[h - 1].1.max(1e-15);

            // Tricube kernel weights
            let mut sum_w = 0.0;
            let mut sum_wx = 0.0;
            let mut sum_wy = 0.0;
            let mut sum_wxx = 0.0;
            let mut sum_wxy = 0.0;

            for &(i, d) in dists.iter().take(h) {
                if !y[i].is_finite() || w[i] <= 0.0 {
                    continue;
                }
                let u = d / max_dist;
                let kernel = if u < 1.0 {
                    (1.0 - u.powi(3)).powi(3)
                } else {
                    0.0
                };
                let wi = kernel * w[i];
                let xi = x[i] - xp;

                sum_w += wi;
                sum_wx += wi * xi;
                sum_wy += wi * y[i];
                sum_wxx += wi * xi * xi;
                sum_wxy += wi * xi * y[i];
            }

            if sum_w < 1e-15 {
                let valid: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
                return if valid.is_empty() {
                    0.0
                } else {
                    valid.iter().sum::<f64>() / valid.len() as f64
                };
            }

            // Local linear fit: y = a + b*(x - xp)
            let det = sum_w * sum_wxx - sum_wx * sum_wx;
            if det.abs() < 1e-15 {
                sum_wy / sum_w
            } else {
                (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
            }
        })
        .collect()
}
