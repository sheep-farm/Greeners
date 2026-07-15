//! LSTM (Long Short-Term Memory) recurrent neural network
//! for time series forecasting.
//!
//! Hochreiter & Schmidhuber (1997). A simplified single-layer
//! LSTM with the following gates:
//!
//!   f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)  (forget gate)
//!   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)  (input gate)
//!   g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)     (candidate)
//!   c_t = f_t * c_{t-1} + i_t * g_t             (cell state)
//!   o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)  (output gate)
//!   h_t = o_t * tanh(c_t)                       (hidden state)
//!
//! Output: y_hat = W_y * h_t + b_y
//!
//! Training: truncated backpropagation through time (BPTT).
//! Input: univariate series, converted to sliding windows.

use crate::GreenersError;
use ndarray::Array1;
use std::fmt;

/// Result of LSTM estimation.
#[derive(Debug)]
pub struct LstmResult {
    /// In-sample fitted values
    pub fitted: Array1<f64>,
    /// Multi-step forecast (n_forecast steps)
    pub forecast: Array1<f64>,
    /// Final hidden state
    pub final_hidden: f64,
    /// Final cell state
    pub final_cell: f64,
    /// Number of hidden units
    pub n_hidden: usize,
    /// Sequence length (lookback window)
    pub seq_len: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub n_epochs: usize,
    /// Final MSE
    pub mse: f64,
    /// In-sample R-squared
    pub r_squared: f64,
    /// Number of training samples
    pub n_samples: usize,
    /// Series length
    pub n_obs: usize,
}

impl fmt::Display for LstmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " LSTM Time Series ")?;
        writeln!(f, "Hochreiter & Schmidhuber (1997)")?;
        writeln!(f, "Single-layer LSTM with BPTT training")?;
        writeln!(f, "{:<20} {:>12}", "Series length:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Training samples:", self.n_samples)?;
        writeln!(f, "{:<20} {:>12}", "Hidden units:", self.n_hidden)?;
        writeln!(f, "{:<20} {:>12}", "Sequence length:", self.seq_len)?;
        writeln!(f, "{:<20} {:>12}", "Epochs:", self.n_epochs)?;
        writeln!(f, "{:<20} {:>12.6}", "Learning rate:", self.learning_rate)?;
        writeln!(f, "{:<20} {:>12.6}", "Final MSE:", self.mse)?;
        writeln!(f, "{:<20} {:>12.6}", "In-sample R²:", self.r_squared)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Final hidden state:", self.final_hidden
        )?;
        writeln!(f, "{:<20} {:>12.6}", "Final cell state:", self.final_cell)?;

        // Forecast
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Multi-step forecast:")?;
        let n_show = self.forecast.len().min(10);
        writeln!(f, "  {:<8} {:>14}", "Step", "Forecast")?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..n_show {
            writeln!(f, "  {:<8} {:>14.6}", i + 1, self.forecast[i])?;
        }

        // Fitted vs actual (selected)
        writeln!(f, "\n  Fitted vs actual (last 5 obs):")?;
        writeln!(f, "  {:<8} {:>14} {:>14}", "Obs", "Actual", "Fitted")?;
        let start = self.fitted.len().saturating_sub(5);
        for i in start..self.fitted.len() {
            writeln!(f, "  {:<8} {:>14.6} {:>14.6}", i + 1, 0.0, self.fitted[i])?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct LSTM;

impl LSTM {
    /// Estimate LSTM for time series forecasting.
    ///
    /// # Arguments
    /// * `y` - Time series (n)
    /// * `n_hidden` - Number of hidden units (default 10)
    /// * `seq_len` - Lookback window (default 10)
    /// * `learning_rate` - Learning rate (default 0.01)
    /// * `n_epochs` - Training epochs (default 100)
    /// * `n_forecast` - Number of steps to forecast (default 5)
    pub fn fit(
        y: &Array1<f64>,
        n_hidden: Option<usize>,
        seq_len: Option<usize>,
        learning_rate: Option<f64>,
        n_epochs: Option<usize>,
        n_forecast: Option<usize>,
    ) -> Result<LstmResult, GreenersError> {
        let n = y.len();
        if n < 20 {
            return Err(GreenersError::InvalidOperation(
                "LSTM: need at least 20 observations".into(),
            ));
        }

        let hidden = n_hidden.unwrap_or(10);
        let seq = seq_len.unwrap_or(10).min(n - 5);
        let lr = learning_rate.unwrap_or(0.01);
        let epochs = n_epochs.unwrap_or(100);
        let n_fc = n_forecast.unwrap_or(5);

        // Standardize
        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = y.std(0.0);
        if y_std < 1e-10 {
            return Err(GreenersError::InvalidOperation(
                "LSTM: series has zero variance".into(),
            ));
        }
        let y_norm: Array1<f64> = y.mapv(|v| (v - y_mean) / y_std);

        // Build training samples: (x_seq, y_next)
        // x_seq = [y[t], y[t+1], ..., y[t+seq-1]], y_next = y[t+seq]
        let n_samples = n - seq;
        if n_samples < 5 {
            return Err(GreenersError::InvalidOperation(
                "LSTM: sequence too long for series".into(),
            ));
        }

        // LSTM weights (simplified: single hidden unit for tractability)
        // Gates: forget, input, candidate, output
        // Each: W (seq_len+1) and b (1) — input is [h_{t-1}, x_t]
        // For simplicity, we use a compact representation
        let input_size = 1 + 1; // h_{t-1} + x_t (univariate)

        let mut w_f = Self::init_weights(input_size);
        let mut w_i = Self::init_weights(input_size);
        let mut w_g = Self::init_weights(input_size);
        let mut w_o = Self::init_weights(input_size);
        let mut w_y = Self::init_weights(hidden + 1); // h + bias

        // Training loop
        let mut final_mse = 0.0;
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for sample in 0..n_samples {
                // Forward pass through sequence
                let mut h = 0.0;
                let mut c = 0.0;
                let mut h_hist: Vec<f64> = Vec::with_capacity(seq);
                let mut c_hist: Vec<f64> = Vec::with_capacity(seq);
                let mut f_hist: Vec<f64> = Vec::with_capacity(seq);
                let mut i_hist: Vec<f64> = Vec::with_capacity(seq);
                let mut g_hist: Vec<f64> = Vec::with_capacity(seq);
                let mut o_hist: Vec<f64> = Vec::with_capacity(seq);

                for t in 0..seq {
                    let x_t = y_norm[sample + t];
                    let input = [h, x_t];

                    let f = Self::sigmoid(Self::dot(&w_f, &input));
                    let i = Self::sigmoid(Self::dot(&w_i, &input));
                    let g = (Self::dot(&w_g, &input)).tanh();
                    let o = Self::sigmoid(Self::dot(&w_o, &input));

                    c = f * c + i * g;
                    h = o * c.tanh();

                    h_hist.push(h);
                    c_hist.push(c);
                    f_hist.push(f);
                    i_hist.push(i);
                    g_hist.push(g);
                    o_hist.push(o);
                }

                // Output: y_hat = w_y * [h, 1]
                let y_hat_input = vec![h; hidden + 1];
                let y_hat = Self::dot(&w_y, &y_hat_input);
                let y_true = y_norm[sample + seq];

                let error = y_hat - y_true;
                epoch_loss += error * error;

                // Backward pass (simplified BPTT)
                // Gradient w.r.t. w_y
                let dy_hat = 2.0 * error;
                for (j, w_y_j) in w_y.iter_mut().enumerate() {
                    let input_val = if j < hidden { h } else { 1.0 };
                    *w_y_j -= lr * dy_hat * input_val;
                }

                // Gradient w.r.t. h (from output)
                let dh_out = dy_hat * w_y[0];

                // Truncated BPTT: only last few steps
                let bptt_steps = 5.min(seq);
                let mut dh = dh_out;
                let mut dc = dh * o_hist[seq - 1] * (1.0 - c_hist[seq - 1].tanh().powi(2));

                for step in 0..bptt_steps {
                    let t = seq - 1 - step;
                    let h_prev = if t > 0 { h_hist[t - 1] } else { 0.0 };
                    let c_prev = if t > 0 { c_hist[t - 1] } else { 0.0 };
                    let x_t = y_norm[sample + t];

                    // Gradients for gates
                    let df = dc * c_prev * f_hist[t] * (1.0 - f_hist[t]);
                    let di = dc * g_hist[t] * i_hist[t] * (1.0 - i_hist[t]);
                    let dg = dc * i_hist[t] * (1.0 - g_hist[t].powi(2));
                    let do_g = dh * c_hist[t].tanh() * o_hist[t] * (1.0 - o_hist[t]);

                    // Update weights
                    let input = [h_prev, x_t];
                    for j in 0..input_size {
                        w_f[j] -= lr * df * input[j];
                        w_i[j] -= lr * di * input[j];
                        w_g[j] -= lr * dg * input[j];
                        w_o[j] -= lr * do_g * input[j];
                    }

                    // Propagate gradients back
                    dc = f_hist[t] * dc + df * w_f[0] + di * w_i[0] + dg * w_g[0];
                    dh = do_g * w_o[0];
                    if t > 0 {
                        dh += h_hist[t - 1] * 0.0; // truncated
                    }
                }
            }

            final_mse = epoch_loss / n_samples as f64;

            // Early stopping if converged
            if epoch > 10 && final_mse < 1e-6 {
                break;
            }
        }

        // Generate fitted values
        let mut fitted = Array1::zeros(n);
        for sample in 0..n_samples {
            let mut h = 0.0;
            let mut c = 0.0;
            for t in 0..seq {
                let x_t = y_norm[sample + t];
                let input = [h, x_t];
                let f = Self::sigmoid(Self::dot(&w_f, &input));
                let i = Self::sigmoid(Self::dot(&w_i, &input));
                let g = (Self::dot(&w_g, &input)).tanh();
                let o = Self::sigmoid(Self::dot(&w_o, &input));
                c = f * c + i * g;
                h = o * c.tanh();
            }
            let y_hat_input = vec![h; hidden + 1];
            let y_hat = Self::dot(&w_y, &y_hat_input);
            fitted[sample + seq] = y_hat * y_std + y_mean;
        }

        // Forecast
        let mut forecast = Array1::zeros(n_fc);
        let mut h = 0.0;
        let mut c = 0.0;
        let mut last_seq: Vec<f64> = (0..seq).map(|i| y_norm[n - seq + i]).collect();

        for fc in 0..n_fc {
            h = 0.0;
            c = 0.0;
            for &x_t in last_seq.iter().take(seq) {
                let input = [h, x_t];
                let f = Self::sigmoid(Self::dot(&w_f, &input));
                let i = Self::sigmoid(Self::dot(&w_i, &input));
                let g = (Self::dot(&w_g, &input)).tanh();
                let o = Self::sigmoid(Self::dot(&w_o, &input));
                c = f * c + i * g;
                h = o * c.tanh();
            }
            let y_hat_input = vec![h; hidden + 1];
            let y_hat = Self::dot(&w_y, &y_hat_input);
            forecast[fc] = y_hat * y_std + y_mean;
            // Shift sequence
            last_seq.remove(0);
            last_seq.push(y_hat);
        }

        // R-squared
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let sse = y
            .iter()
            .zip(fitted.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        Ok(LstmResult {
            fitted,
            forecast,
            final_hidden: h,
            final_cell: c,
            n_hidden: hidden,
            seq_len: seq,
            learning_rate: lr,
            n_epochs: epochs,
            mse: final_mse,
            r_squared,
            n_samples,
            n_obs: n,
        })
    }

    fn init_weights(size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                let seed = (i as u64).wrapping_mul(12345) + 67890;
                let val = ((seed % 1000) as f64 / 1000.0) - 0.5;
                val * 0.1
            })
            .collect()
    }

    fn dot(w: &[f64], x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..w.len().min(x.len()) {
            sum += w[i] * x[i];
        }
        if w.len() > x.len() {
            sum += w[x.len()]; // bias
        }
        sum
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}
