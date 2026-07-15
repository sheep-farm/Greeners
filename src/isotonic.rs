//! Isotonic Regression (Barlow, Bartholomew, Bremner & Brunk 1972).
//!
//! Monotone regression: find the best-fitting monotone (non-
//! decreasing) sequence to data. Solves:
//!
//!   min  sum((y_i - y_hat_i)^2)
//!   s.t. y_hat_1 <= y_hat_2 <= ... <= y_hat_n
//!
//! Algorithm: Pool Adjacent Violators Algorithm (PAVA).
//!
//! Also supports non-increasing order and weighted isotonic
//! regression. Reports fitted values, R-squared, and optionally
//! the step function (unique level transitions).

use crate::GreenersError;
use ndarray::Array1;
use std::fmt;

/// Result of isotonic regression.
#[derive(Debug)]
pub struct IsotonicResult {
    /// Fitted (monotone) values (n)
    pub fitted: Array1<f64>,
    /// Input x (sorted order)
    pub x: Array1<f64>,
    /// Input y (sorted order)
    pub y: Array1<f64>,
    /// Weights (sorted order)
    pub weights: Array1<f64>,
    /// Whether the fit is non-decreasing (true) or non-increasing (false)
    pub increasing: bool,
    /// Unique x values at step boundaries
    pub x_steps: Vec<f64>,
    /// Fitted values at step boundaries
    pub y_steps: Vec<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Mean squared error
    pub mse: f64,
    /// Number of unique blocks (steps)
    pub n_blocks: usize,
    /// Number of observations
    pub n_obs: usize,
}

impl fmt::Display for IsotonicResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let order = if self.increasing {
            "non-decreasing"
        } else {
            "non-increasing"
        };
        writeln!(f, "\n{:=^78}", " Isotonic Regression ")?;
        writeln!(f, "Barlow, Bartholomew, Bremner & Brunk (1972)")?;
        writeln!(f, "Pool Adjacent Violators Algorithm (PAVA)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Order:", order)?;
        writeln!(f, "{:<20} {:>12}", "Unique blocks:", self.n_blocks)?;
        writeln!(f, "{:<20} {:>12.6}", "R²:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "MSE:", self.mse)?;

        // Step function
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Step function:")?;
        writeln!(f, "  {:<10} {:>14}", "x", "y_hat")?;
        writeln!(f, "{:-^78}", "")?;
        let n_show = self.x_steps.len().min(15);
        for i in 0..n_show {
            writeln!(f, "  {:<10.4} {:>14.6}", self.x_steps[i], self.y_steps[i])?;
        }
        if self.x_steps.len() > 15 {
            writeln!(f, "  ... ({} steps total)", self.x_steps.len())?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct IsotonicRegression;

impl IsotonicRegression {
    /// Fit isotonic regression.
    ///
    /// # Arguments
    /// * `x` - Predictor (n), will be sorted
    /// * `y` - Response (n)
    /// * `increasing` - If true, fit non-decreasing; if false, non-increasing
    /// * `weights` - Optional weights (n), default uniform
    pub fn fit(
        x: &Array1<f64>,
        y: &Array1<f64>,
        increasing: bool,
        weights: Option<&Array1<f64>>,
    ) -> Result<IsotonicResult, GreenersError> {
        let n = x.len();
        if y.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "IsotonicRegression: x and y must have same length".into(),
            ));
        }
        if n < 2 {
            return Err(GreenersError::InvalidOperation(
                "IsotonicRegression: need at least 2 observations".into(),
            ));
        }

        let w = weights.cloned().unwrap_or_else(|| Array1::ones(n));
        if w.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "IsotonicRegression: weights must have same length as x".into(),
            ));
        }

        // Sort by x
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let x_sorted: Array1<f64> = indices.iter().map(|&i| x[i]).collect();
        let y_sorted: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        let w_sorted: Array1<f64> = indices.iter().map(|&i| w[i]).collect();

        // If non-increasing, negate y, fit non-decreasing, then negate result
        let y_work: Array1<f64> = if increasing {
            y_sorted.clone()
        } else {
            y_sorted.mapv(|v| -v)
        };

        // PAVA
        let fitted = Self::pava(&y_work, &w_sorted, n);
        let fitted_final: Array1<f64> = if increasing {
            fitted
        } else {
            fitted.mapv(|v| -v)
        };

        // Extract step function (unique blocks)
        let (x_steps, y_steps) = Self::extract_steps(&x_sorted, &fitted_final, n);

        // R-squared and MSE
        let y_mean = y_sorted.mean().unwrap_or(0.0);
        let tss: f64 = y_sorted.mapv(|v| (v - y_mean).powi(2)).sum();
        let sse: f64 = y_sorted
            .iter()
            .zip(fitted_final.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };
        let mse = sse / n as f64;

        Ok(IsotonicResult {
            fitted: fitted_final,
            x: x_sorted,
            y: y_sorted,
            weights: w_sorted,
            increasing,
            x_steps,
            y_steps,
            r_squared,
            mse,
            n_blocks: 0, // set below
            n_obs: n,
        })
    }

    /// Pool Adjacent Violators Algorithm.
    fn pava(y: &Array1<f64>, w: &Array1<f64>, n: usize) -> Array1<f64> {
        // Blocks: each block has (value, weight_sum, count, start_index)
        let mut block_values: Vec<f64> = y.to_vec();
        let mut block_weights: Vec<f64> = w.to_vec();
        let mut block_counts: Vec<usize> = vec![1; n];
        let mut block_starts: Vec<usize> = (0..n).collect();

        let mut n_blocks = n;

        // Forward pass: pool violators
        let mut i = 0;
        while i < n_blocks - 1 {
            if block_values[i] > block_values[i + 1] {
                // Pool blocks i and i+1
                let w_i = block_weights[i];
                let w_j = block_weights[i + 1];
                let pooled_value =
                    (block_values[i] * w_i + block_values[i + 1] * w_j) / (w_i + w_j);
                let pooled_weight = w_i + w_j;
                let pooled_count = block_counts[i] + block_counts[i + 1];

                block_values[i] = pooled_value;
                block_weights[i] = pooled_weight;
                block_counts[i] = pooled_count;

                // Remove block i+1
                block_values.remove(i + 1);
                block_weights.remove(i + 1);
                block_counts.remove(i + 1);
                block_starts.remove(i + 1);

                n_blocks -= 1;

                // Backtrack: check if previous block now violates
                if i > 0 {
                    i = i.saturating_sub(1);
                }
            } else {
                i += 1;
            }
        }

        // Expand blocks back to fitted values
        let mut fitted = Array1::zeros(n);
        let mut idx = 0;
        for b in 0..n_blocks {
            for _ in 0..block_counts[b] {
                fitted[idx] = block_values[b];
                idx += 1;
            }
        }
        fitted
    }

    fn extract_steps(x: &Array1<f64>, fitted: &Array1<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x_steps = Vec::new();
        let mut y_steps = Vec::new();

        if n == 0 {
            return (x_steps, y_steps);
        }

        x_steps.push(x[0]);
        y_steps.push(fitted[0]);

        for i in 1..n {
            if (fitted[i] - fitted[i - 1]).abs() > 1e-12 {
                x_steps.push(x[i]);
                y_steps.push(fitted[i]);
            }
        }

        (x_steps, y_steps)
    }

    /// Predict fitted values for new x values.
    /// Uses linear interpolation between step boundaries.
    pub fn predict(result: &IsotonicResult, x_new: &Array1<f64>) -> Array1<f64> {
        let n = x_new.len();
        let mut pred = Array1::zeros(n);

        for i in 0..n {
            let x = x_new[i];
            // Find the step that x falls into
            if x <= result.x_steps[0] {
                pred[i] = result.y_steps[0];
            } else if x >= *result.x_steps.last().unwrap() {
                pred[i] = *result.y_steps.last().unwrap();
            } else {
                // Find the interval
                let mut found = false;
                for j in 1..result.x_steps.len() {
                    if x < result.x_steps[j] {
                        // x is in [x_steps[j-1], x_steps[j])
                        // Isotonic is a step function: use y_steps[j-1]
                        pred[i] = result.y_steps[j - 1];
                        found = true;
                        break;
                    }
                }
                if !found {
                    pred[i] = *result.y_steps.last().unwrap();
                }
            }
        }

        pred
    }
}
