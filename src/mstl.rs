use crate::GreenersError;
use ndarray::Array1;
use std::fmt;

pub struct MSTL;

#[derive(Debug)]
pub struct MSTLResult {
    pub trend: Array1<f64>,
    pub seasonal: Vec<Array1<f64>>,
    pub resid: Array1<f64>,
    pub periods: Vec<usize>,
    pub n_obs: usize,
}

impl MSTLResult {
    /// Reconstruct the observed series: trend + sum(seasonal) + resid
    pub fn observed(&self) -> Array1<f64> {
        let mut out = self.trend.clone() + &self.resid;
        for s in &self.seasonal {
            out += s;
        }
        out
    }
}

impl fmt::Display for MSTLResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " MSTL Decomposition ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Seasonal periods:", format!("{:?}", self.periods))?;
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "Trend mean:",
            self.trend.mean().unwrap_or(f64::NAN)
        )?;
        for (i, s) in self.seasonal.iter().enumerate() {
            writeln!(
                f,
                "Seasonal[{}] std:     {:>10.4}",
                self.periods[i],
                std_dev(s)
            )?;
        }
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "Residual std:",
            std_dev(&self.resid)
        )?;
        writeln!(f, "{:=^60}", "")
    }
}

fn std_dev(arr: &Array1<f64>) -> f64 {
    let n = arr.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = arr.mean().unwrap_or(0.0);
    let var = arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

impl MSTL {
    /// Decompose a time series with multiple seasonal periods.
    ///
    /// `periods`: vector of seasonal periods, e.g., [7, 365] for daily data
    /// with weekly + yearly seasonality. Each period must be >= 2.
    pub fn fit(y: &Array1<f64>, periods: &[usize]) -> Result<MSTLResult, GreenersError> {
        let n = y.len();
        if n < 4 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for MSTL decomposition".into(),
            ));
        }
        if periods.is_empty() {
            return Err(GreenersError::ShapeMismatch(
                "At least one seasonal period must be provided".into(),
            ));
        }
        for &p in periods {
            if p < 2 {
                return Err(GreenersError::ShapeMismatch(
                    format!("Seasonal period must be >= 2, got {}", p),
                ));
            }
            if p > n {
                return Err(GreenersError::ShapeMismatch(
                    format!(
                        "Seasonal period {} exceeds series length {}",
                        p, n
                    ),
                ));
            }
        }

        // Sort periods ascending
        let mut sorted_periods = periods.to_vec();
        sorted_periods.sort();

        let n_seasonal = sorted_periods.len();
        let mut seasonals: Vec<Array1<f64>> = vec![Array1::zeros(n); n_seasonal];
        let mut trend = Array1::zeros(n);

        let max_iter = 5;

        for _iter in 0..max_iter {
            // For each seasonal period, extract seasonal component via STL
            for (si, &period) in sorted_periods.iter().enumerate() {
                // input = y - trend - sum(other seasonals)
                let mut input = y - &trend;
                for (sj, scomp) in seasonals.iter().enumerate() {
                    if sj != si {
                        input -= scomp;
                    }
                }

                // Apply single-period STL decomposition
                let (stl_seasonal, _stl_trend) = stl_decompose(&input, period)?;
                seasonals[si] = stl_seasonal;
            }

            // Compute trend from y - sum(seasonals) using LOESS-like moving average
            let mut deseasoned = y.clone();
            for s in &seasonals {
                deseasoned -= s;
            }
            // Use a robust trend extraction: centered moving average with window
            // proportional to the largest period
            let trend_window = sorted_periods.last().copied().unwrap_or(3);
            let window = (trend_window | 1).max(3); // ensure odd and >= 3
            trend = moving_average_trend(&deseasoned, window);
        }

        // Final residual
        let mut resid = y - &trend;
        for s in &seasonals {
            resid -= s;
        }

        Ok(MSTLResult {
            trend,
            seasonal: seasonals,
            resid,
            periods: sorted_periods,
            n_obs: n,
        })
    }
}

/// Simplified STL decomposition for a single seasonal period.
/// Returns (seasonal, trend).
fn stl_decompose(
    y: &Array1<f64>,
    period: usize,
) -> Result<(Array1<f64>, Array1<f64>), GreenersError> {
    let n = y.len();

    // Step 1: Initial trend via centered moving average
    let trend = moving_average_trend(y, period | 1); // ensure odd window

    // Step 2: Detrend
    let detrended = y - &trend;

    // Step 3: Extract seasonal by averaging each seasonal position
    let mut seasonal = Array1::zeros(n);
    let mut season_avgs = vec![0.0f64; period];
    let mut season_counts = vec![0usize; period];

    for i in 0..n {
        let pos = i % period;
        if detrended[i].is_finite() {
            season_avgs[pos] += detrended[i];
            season_counts[pos] += 1;
        }
    }

    for p in 0..period {
        if season_counts[p] > 0 {
            season_avgs[p] /= season_counts[p] as f64;
        }
    }

    // Center the seasonal component (subtract mean so it sums to ~0)
    let smean: f64 = season_avgs.iter().sum::<f64>() / period as f64;
    for avg in &mut season_avgs {
        *avg -= smean;
    }

    for i in 0..n {
        seasonal[i] = season_avgs[i % period];
    }

    // Refine: iterate the seasonal extraction a couple of times with LOESS smoothing
    // on the cycle-subseries
    for _inner in 0..2 {
        let new_detrended = y - &moving_average_trend(&(y - &seasonal), period | 1);

        let mut new_avgs = vec![0.0f64; period];
        let mut new_counts = vec![0usize; period];
        for i in 0..n {
            let pos = i % period;
            if new_detrended[i].is_finite() {
                new_avgs[pos] += new_detrended[i];
                new_counts[pos] += 1;
            }
        }
        for (p, avg) in new_avgs.iter_mut().enumerate().take(period) {
            if new_counts[p] > 0 {
                *avg /= new_counts[p] as f64;
            }
        }
        let avg_mean: f64 = new_avgs.iter().sum::<f64>() / period as f64;
        for avg in &mut new_avgs {
            *avg -= avg_mean;
        }

        // LOESS-smooth each cycle-subseries
        for p in 0..period {
            let indices: Vec<usize> = (p..n).step_by(period).collect();
            if indices.len() >= 3 {
                let sub_values: Vec<f64> = indices.iter().map(|&i| new_detrended[i]).collect();
                let smoothed = loess_smooth(&sub_values, 0.3_f64.max(3.0 / indices.len() as f64));
                for (j, &idx) in indices.iter().enumerate() {
                    seasonal[idx] = smoothed[j];
                }
            } else {
                for &idx in &indices {
                    seasonal[idx] = new_avgs[p];
                }
            }
        }

        // Re-center
        let smean2: f64 = seasonal.iter().sum::<f64>() / n as f64;
        seasonal.mapv_inplace(|v| v - smean2);
    }

    let final_trend = moving_average_trend(&(y - &seasonal), period | 1);

    Ok((seasonal, final_trend))
}

/// Centered moving average for trend extraction.
fn moving_average_trend(y: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = y.len();
    let w = window.max(1);
    let half = w / 2;
    let mut trend = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let count = end - start;
        let sum: f64 = (start..end).map(|j| y[j]).sum();
        trend[i] = sum / count as f64;
    }

    // Apply a second pass for even-period smoothing (2x moving average)
    if window.is_multiple_of(2) {
        let first = trend.clone();
        for i in 0..n {
            let start = i.saturating_sub(1);
            let end = (i + 2).min(n);
            let count = end - start;
            let sum: f64 = (start..end).map(|j| first[j]).sum();
            trend[i] = sum / count as f64;
        }
    }

    trend
}

/// Simple LOESS (local regression) smoother.
/// `span` is the fraction of data to use for each local regression (0..1].
fn loess_smooth(y: &[f64], span: f64) -> Vec<f64> {
    let n = y.len();
    if n <= 2 {
        return y.to_vec();
    }

    let span = span.clamp(0.1, 1.0);
    let h = ((n as f64 * span).ceil() as usize).max(3).min(n);

    let mut result = vec![0.0; n];

    for i in 0..n {
        let x_i = i as f64;

        // Find the h nearest neighbors
        let mut dists: Vec<(usize, f64)> = (0..n).map(|j| (j, (j as f64 - x_i).abs())).collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors = &dists[..h];
        let max_dist = neighbors.last().unwrap().1.max(1e-10);

        // Tricube weights
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        let mut sum_wy = 0.0;
        let mut sum_wxx = 0.0;
        let mut sum_wxy = 0.0;

        for &(j, d) in neighbors {
            let u = d / max_dist;
            let w = if u < 1.0 {
                let t = 1.0 - u * u * u;
                t * t * t
            } else {
                0.0
            };
            let xj = j as f64;
            sum_w += w;
            sum_wx += w * xj;
            sum_wy += w * y[j];
            sum_wxx += w * xj * xj;
            sum_wxy += w * xj * y[j];
        }

        // Weighted linear regression: y = a + b*x
        let denom = sum_w * sum_wxx - sum_wx * sum_wx;
        if denom.abs() > 1e-15 {
            let b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
            let a = (sum_wy - b * sum_wx) / sum_w;
            result[i] = a + b * x_i;
        } else if sum_w > 0.0 {
            result[i] = sum_wy / sum_w;
        } else {
            result[i] = y[i];
        }
    }

    result
}
