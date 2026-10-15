use greeners_core::GreenersError;
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
    /// * `period` —seasonal period (e.g., 12 for monthly, 4 for fourthly)
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

    /// Robust STL decomposition, with local-linear smoothers and no skipping.
    ///
    /// Runs two inner iterations, then one robustness-weighted refit (two more
    /// inner iterations). The low-pass window equals the effective trend window,
    /// not the statsmodels default. Seasonal windows are raised to at least 7;
    /// seasonal and trend windows are rounded up to odd. A zero trend window
    /// selects the next odd >= ceil(1.5 * period / (1 - 1.5 / seasonal_window)).
    ///
    /// Rejects non-finite inputs, fewer than two periods, effective trend windows
    /// <= period, dimensions/spans above i32::MAX, and non-finite arithmetic.
    /// Finite inputs over the full f64 range are not guaranteed to succeed.
    /// Robust residuals use abs(y - (trend + seasonal)); returned residuals
    /// retain the expression y - trend - seasonal.
    ///
    /// Adapted from statsmodels 0.14.6 `_stl.pyx`, (c) 2019 Kevin Sheppard,
    /// NCSA/BSD-3 Clause, based on NETLIB STL. See THIRD_PARTY_NOTICES.md.
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
        if period < 2 || period > n / 2 {
            return Err(GreenersError::ShapeMismatch(
                "STL requires period >= 2 and at least two complete periods".into(),
            ));
        }
        // Bound dimensions before arithmetic, indexing or allocation. Spans
        // need not fit the series, but must fit the reference's integer domain.
        let extended_len = period.checked_mul(2).and_then(|p| n.checked_add(p));
        if extended_len.is_none_or(|len| {
            len > i32::MAX as usize || len > isize::MAX as usize / std::mem::size_of::<f64>()
        }) || seasonal_window > i32::MAX as usize
            || trend_window > i32::MAX as usize
        {
            return Err(GreenersError::ShapeMismatch(
                "STL dimensions or windows too large".into(),
            ));
        }
        for &value in series {
            stl_finite(value)?;
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
        if t_win <= period || t_win > i32::MAX as usize {
            return Err(GreenersError::ShapeMismatch(
                "STL effective trend window must exceed period and fit i32".into(),
            ));
        }

        let mut seasonal = vec![0.0; n];
        let mut trend = vec![0.0; n];
        let mut weights = vec![1.0; n];

        for outer in 0..2 {
            let robustness = if outer == 0 {
                None
            } else {
                Some(weights.as_slice())
            };
            for _ in 0..2 {
                let detrended: Vec<f64> = (0..n)
                    .map(|i| stl_finite(series[i] - trend[i]))
                    .collect::<Result<_, _>>()?;
                let extended = stl_extend(&detrended, period, s_win, robustness)?;
                let first = stl_moving_average(&extended, period)?;
                let second = stl_moving_average(&first, period)?;
                let low_pass = stl_moving_average(&second, 3)?;
                let low_pass = stl_smooth(&low_pass, t_win, None)?;
                let mut deseasoned = vec![0.0; n];
                for i in 0..n {
                    seasonal[i] = stl_finite(extended[period + i] - low_pass[i])?;
                    deseasoned[i] = stl_finite(series[i] - seasonal[i])?;
                }
                trend = stl_smooth(&deseasoned, t_win, robustness)?;
            }
            if outer == 0 {
                let absolute_residuals: Vec<f64> = (0..n)
                    .map(|i| {
                        stl_finite(series[i] - stl_finite(trend[i] + seasonal[i])?).map(f64::abs)
                    })
                    .collect::<Result<_, _>>()?;
                stl_bisquare(&absolute_residuals, &mut weights)?;
            }
        }

        let residual: Vec<f64> = (0..n)
            .map(|i| stl_finite(stl_finite(series[i] - trend[i])? - seasonal[i]))
            .collect::<Result<_, _>>()?;

        Ok(DecompositionResult {
            observed: series.clone(),
            trend: Array1::from_vec(trend),
            seasonal: Array1::from_vec(seasonal),
            residual: Array1::from_vec(residual),
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

// STL helpers adapted from statsmodels 0.14.6 _stl.pyx, (c) 2019 Kevin
// Sheppard, NCSA/BSD-3 Clause; based on NETLIB STL. Full notice in this crate's
// THIRD_PARTY_NOTICES.md. Restricted to degree 1 / jump 1; arithmetic errors
// are explicit instead of sharing the reference's NaN no-support sentinel.
fn stl_finite(value: f64) -> Result<f64, GreenersError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GreenersError::InvalidOperation(
            "STL encountered non-finite input or arithmetic".into(),
        ))
    }
}

fn stl_moving_average(y: &[f64], window: usize) -> Result<Vec<f64>, GreenersError> {
    let mut sum = 0.0;
    for &v in &y[..window] {
        sum = stl_finite(sum + v)?;
    }
    let mut result = Vec::with_capacity(y.len() - window + 1);
    result.push(sum / window as f64);
    for i in window..y.len() {
        sum = stl_finite(sum + stl_finite(y[i] - y[i - window])?)?;
        result.push(sum / window as f64);
    }
    Ok(result)
}

// xs uses the reference's 1-based regular grid; bounds are a Rust half-open
// range. None means genuine zero support, not a failed arithmetic operation.
fn stl_estimate(
    y: &[f64],
    span: usize,
    xs: f64,
    bounds: std::ops::Range<usize>,
    robustness: Option<&[f64]>,
) -> Result<Option<f64>, GreenersError> {
    let mut h = (xs - (bounds.start + 1) as f64).max(bounds.end as f64 - xs);
    if span > y.len() {
        h += ((span - y.len()) / 2) as f64;
    }
    let mut w = Vec::with_capacity(bounds.len());
    let mut total = 0.0;
    for j in bounds.clone() {
        stl_finite(y[j])?;
        let r = ((j + 1) as f64 - xs).abs();
        let mut weight = if r > 0.999 * h {
            0.0
        } else if r <= 0.001 * h {
            1.0
        } else {
            (1.0 - (r / h).powi(3)).powi(3)
        };
        if let Some(rw) = robustness {
            weight = stl_finite(weight * rw[j])?;
        }
        total = stl_finite(total + weight)?;
        w.push(weight);
    }
    if total <= 0.0 {
        return Ok(None);
    }
    for weight in &mut w {
        *weight /= total;
    }
    if h > 0.0 {
        let mut mean = 0.0;
        for (j, &weight) in bounds.clone().zip(&w) {
            mean += weight * (j + 1) as f64;
        }
        let mut variance = 0.0;
        for (j, &weight) in bounds.clone().zip(&w) {
            variance += weight * ((j + 1) as f64 - mean).powi(2);
        }
        stl_finite(variance)?;
        if variance.sqrt() > 0.001 * (y.len() - 1) as f64 {
            let slope = (xs - mean) / variance;
            for (j, weight) in bounds.clone().zip(&mut w) {
                *weight = stl_finite(*weight * (slope * ((j + 1) as f64 - mean) + 1.0))?;
            }
        }
    }
    let mut estimate = 0.0;
    for (j, weight) in bounds.zip(w) {
        estimate = stl_finite(estimate + weight * y[j])?;
    }
    Ok(Some(estimate))
}

fn stl_smooth(
    y: &[f64],
    span: usize,
    robustness: Option<&[f64]>,
) -> Result<Vec<f64>, GreenersError> {
    let n = y.len();
    let width = span.min(n);
    (0..n)
        .map(|i| {
            let left = if span >= n {
                0
            } else {
                i.saturating_sub(span / 2).min(n - width)
            };
            Ok(
                stl_estimate(y, span, (i + 1) as f64, left..left + width, robustness)?
                    .unwrap_or(y[i]),
            )
        })
        .collect()
}

fn stl_extend(
    y: &[f64],
    period: usize,
    span: usize,
    robustness: Option<&[f64]>,
) -> Result<Vec<f64>, GreenersError> {
    let mut extended = vec![0.0; y.len() + 2 * period];
    for phase in 0..period {
        let sub: Vec<f64> = (phase..y.len()).step_by(period).map(|i| y[i]).collect();
        let rw: Option<Vec<f64>> = robustness.map(|weights| {
            (phase..y.len())
                .step_by(period)
                .map(|i| weights[i])
                .collect()
        });
        let smoothed = stl_smooth(&sub, span, rw.as_deref())?;
        let k = sub.len();
        let left =
            stl_estimate(&sub, span, 0.0, 0..span.min(k), rw.as_deref())?.unwrap_or(smoothed[0]);
        let right = stl_estimate(
            &sub,
            span,
            (k + 1) as f64,
            k.saturating_sub(span)..k,
            rw.as_deref(),
        )?
        .unwrap_or(smoothed[k - 1]);
        extended[phase] = left;
        for (j, &value) in smoothed.iter().enumerate() {
            extended[(j + 1) * period + phase] = value;
        }
        extended[(k + 1) * period + phase] = right;
    }
    Ok(extended)
}

fn stl_bisquare(residuals: &[f64], weights: &mut [f64]) -> Result<(), GreenersError> {
    for &r in residuals {
        stl_finite(r)?;
    }
    let mut sorted = residuals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    let scale = stl_finite(3.0 * stl_finite(sorted[mid] + sorted[sorted.len() - mid - 1])?)?;
    if scale == 0.0 {
        weights.fill(1.0);
        return Ok(());
    }
    for (&r, w) in residuals.iter().zip(weights) {
        *w = if r <= 0.001 * scale {
            1.0
        } else if r <= 0.999 * scale {
            (1.0 - (r / scale).powi(2)).powi(2)
        } else {
            0.0
        };
    }
    Ok(())
}

#[cfg(test)]
mod stl_tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!(a.is_finite() && (a - b).abs() <= 1e-12, "{a} != {b}");
    }

    #[test]
    fn stl_valid_ma_support_and_coefficients() {
        for window in [3, 12] {
            let n = 20;
            for impulse in 0..n {
                let mut y = vec![0.0; n];
                y[impulse] = 1.0;
                let result = stl_moving_average(&y, window).unwrap();
                assert_eq!(result.len(), n - window + 1);
                for (i, &value) in result.iter().enumerate() {
                    close(
                        value,
                        if (i..i + window).contains(&impulse) {
                            1.0 / window as f64
                        } else {
                            0.0
                        },
                    );
                }
            }
        }
    }

    #[test]
    fn stl_extended_cascade_alignment_and_extrapolation() {
        for (n, p) in [(48, 12), (53, 7)] {
            for slope in [0.0, 0.03125] {
                let y: Vec<f64> = (0..n).map(|i| 2.0 + slope * i as f64).collect();
                let extended = stl_extend(&y, p, 7, None).unwrap();
                assert_eq!(extended.len(), n + 2 * p);
                for (i, &v) in extended.iter().enumerate() {
                    close(v, 2.0 + slope * (i as f64 - p as f64));
                }
                for i in 0..n {
                    close(extended[p + i], y[i]);
                }
                let first = stl_moving_average(&extended, p).unwrap();
                assert_eq!(first.len(), n + p + 1);
                let second = stl_moving_average(&first, p).unwrap();
                assert_eq!(second.len(), n + 2);
                let third = stl_moving_average(&second, 3).unwrap();
                assert_eq!(third.len(), n);
                for i in 0..n {
                    close(third[i], y[i]);
                }
            }
        }
    }

    #[test]
    fn stl_zero_support_uses_observation_and_adjacent_endpoint() {
        let y = [2.0, 5.0, 11.0, 17.0, 23.0, 31.0];
        let w = [0.0; 6];
        assert_eq!(stl_smooth(&y, 7, Some(&w)).unwrap(), y);
        let extended = stl_extend(&y, 2, 7, Some(&w)).unwrap();
        assert_eq!(
            extended,
            [2.0, 5.0, 2.0, 5.0, 11.0, 17.0, 23.0, 31.0, 23.0, 31.0]
        );
    }

    #[test]
    fn stl_bisquare_exact_medians_reset_and_tiny_scale() {
        for r in [vec![1.0, 2.0, 3.0, 4.0, 100.0], vec![1.0, 2.0, 4.0, 100.0]] {
            let mut w = vec![0.25; r.len()];
            stl_bisquare(&r, &mut w).unwrap();
            for i in 0..r.len() {
                close(
                    w[i],
                    if r[i] > 0.999 * 18.0 {
                        0.0
                    } else {
                        (1.0 - (r[i] / 18.0).powi(2)).powi(2)
                    },
                );
            }
            let small: Vec<f64> = r.iter().map(|x| x * 1e-100).collect();
            let mut small_w = vec![0.1; r.len()];
            stl_bisquare(&small, &mut small_w).unwrap();
            for i in 0..r.len() {
                close(small_w[i], w[i]);
            }
        }
        for r in [vec![0.0; 5], vec![0.0, 0.0, 0.0, 1.0, 100.0]] {
            let mut w = vec![0.0, 0.2, 0.5, 0.7, 0.9];
            stl_bisquare(&r, &mut w).unwrap();
            assert_eq!(w, vec![1.0; 5]);
        }
        let r = [0.006, 1.0, 1.0, 0.999 * 6.0, 6.0];
        let mut w = [0.0; 5];
        stl_bisquare(&r, &mut w).unwrap();
        assert_eq!(w[0], 1.0);
        assert!(w[3] > 0.0);
        close(w[3], (1.0_f64 - 0.999_f64.powi(2)).powi(2));
        assert_eq!(w[4], 0.0);
    }

    #[test]
    fn stl_helpers_reject_nonfinite_arithmetic() {
        assert!(matches!(
            stl_bisquare(&[1e308; 5], &mut [0.5; 5]),
            Err(GreenersError::InvalidOperation(_))
        ));
        assert!(stl_bisquare(&[f64::NAN; 5], &mut [0.5; 5]).is_err());
        assert!(stl_moving_average(&[1e308; 5], 3).is_err());
        assert!(stl_smooth(&[f64::INFINITY, 1.0], 7, Some(&[0.0, 0.0])).is_err());
    }
}
