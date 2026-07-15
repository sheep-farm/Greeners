//! Transformer for time series forecasting (Vaswani et al. 2017).
//!
//! A simplified single-head, single-layer transformer encoder
//! adapted for univariate time series:
//!
//! 1. Input embedding: sliding window of length `seq_len` mapped
//!    to hidden dimension via linear projection
//! 2. Positional encoding: sinusoidal (sin/cos) as in the original
//! 3. Self-attention: scaled dot-product attention
//!    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
//! 4. Feed-forward network: 2-layer MLP with ReLU
//! 5. Output: linear projection to scalar forecast
//!
//! Training: gradient descent on MSE loss. Truncated backprop.
//! Input: univariate series, converted to sliding windows.

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of Transformer estimation.
#[derive(Debug)]
pub struct TransformerResult {
    /// In-sample fitted values
    pub fitted: Array1<f64>,
    /// Multi-step forecast
    pub forecast: Array1<f64>,
    /// Number of attention heads (always 1 in this impl)
    pub n_heads: usize,
    /// Hidden dimension
    pub d_model: usize,
    /// Sequence length
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

impl fmt::Display for TransformerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Transformer Time Series ")?;
        writeln!(f, "Vaswani et al. (2017)")?;
        writeln!(f, "Single-head, single-layer encoder")?;
        writeln!(f, "{:<20} {:>12}", "Series length:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Training samples:", self.n_samples)?;
        writeln!(f, "{:<20} {:>12}", "d_model:", self.d_model)?;
        writeln!(f, "{:<20} {:>12}", "Attention heads:", self.n_heads)?;
        writeln!(f, "{:<20} {:>12}", "Sequence length:", self.seq_len)?;
        writeln!(f, "{:<20} {:>12}", "Epochs:", self.n_epochs)?;
        writeln!(f, "{:<20} {:>12.6}", "Learning rate:", self.learning_rate)?;
        writeln!(f, "{:<20} {:>12.6}", "Final MSE:", self.mse)?;
        writeln!(f, "{:<20} {:>12.6}", "In-sample R²:", self.r_squared)?;

        // Forecast
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Multi-step forecast:")?;
        let n_show = self.forecast.len().min(10);
        writeln!(f, "  {:<8} {:>14}", "Step", "Forecast")?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..n_show {
            writeln!(f, "  {:<8} {:>14.6}", i + 1, self.forecast[i])?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct Transformer;

impl Transformer {
    /// Estimate Transformer for time series forecasting.
    ///
    /// # Arguments
    /// * `y` - Time series (n)
    /// * `d_model` - Hidden dimension (default 8)
    /// * `seq_len` - Lookback window (default 10)
    /// * `learning_rate` - Learning rate (default 0.001)
    /// * `n_epochs` - Training epochs (default 100)
    /// * `n_forecast` - Number of steps to forecast (default 5)
    pub fn fit(
        y: &Array1<f64>,
        d_model: Option<usize>,
        seq_len: Option<usize>,
        learning_rate: Option<f64>,
        n_epochs: Option<usize>,
        n_forecast: Option<usize>,
    ) -> Result<TransformerResult, GreenersError> {
        let n = y.len();
        if n < 20 {
            return Err(GreenersError::InvalidOperation(
                "Transformer: need at least 20 observations".into(),
            ));
        }

        let d = d_model.unwrap_or(8);
        let seq = seq_len.unwrap_or(10).min(n - 5);
        let lr = learning_rate.unwrap_or(0.001);
        let epochs = n_epochs.unwrap_or(100);
        let n_fc = n_forecast.unwrap_or(5);

        // Standardize
        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = y.std(0.0);
        if y_std < 1e-10 {
            return Err(GreenersError::InvalidOperation(
                "Transformer: series has zero variance".into(),
            ));
        }
        let y_norm: Array1<f64> = y.mapv(|v| (v - y_mean) / y_std);

        let n_samples = n - seq;
        if n_samples < 5 {
            return Err(GreenersError::InvalidOperation(
                "Transformer: sequence too long for series".into(),
            ));
        }

        // Parameters:
        // W_embed: seq_len x d_model (input projection)
        // W_q, W_k, W_v: d_model x d_model (attention)
        // W_ff1: d_model x d_model (feed-forward)
        // W_ff2: d_model x 1 (output)
        // Plus biases

        let mut w_embed = Array2::zeros((seq, d));
        let mut w_q = Array2::zeros((d, d));
        let mut w_k = Array2::zeros((d, d));
        let mut w_v = Array2::zeros((d, d));
        let mut w_ff1 = Array2::zeros((d, d));
        let mut w_out = Array1::zeros(d);

        // Initialize with small random values
        Self::init_matrix(&mut w_embed, seq, d);
        Self::init_matrix(&mut w_q, d, d);
        Self::init_matrix(&mut w_k, d, d);
        Self::init_matrix(&mut w_v, d, d);
        Self::init_matrix(&mut w_ff1, d, d);
        for i in 0..d {
            w_out[i] = Self::rand_uniform() * 0.1 - 0.05;
        }

        let scale = 1.0 / (d as f64).sqrt();

        // Training loop
        let mut final_mse = 0.0;
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for sample in 0..n_samples {
                // Input: [y_norm[sample], ..., y_norm[sample+seq-1]]
                let x_seq: Vec<f64> = (0..seq).map(|t| y_norm[sample + t]).collect();

                // 1. Input embedding: x_seq * W_embed -> (1 x d_model)
                let mut embedded = vec![0.0_f64; d];
                for j in 0..d {
                    for t in 0..seq {
                        embedded[j] += x_seq[t] * w_embed[(t, j)];
                    }
                }

                // 2. Positional encoding (simplified: add sin/cos of position)
                for (j, emb) in embedded.iter_mut().enumerate().take(d) {
                    let pos = j as f64;
                    let pe = if j % 2 == 0 {
                        (pos * 0.1).sin()
                    } else {
                        (pos * 0.1).cos()
                    };
                    *emb += pe * 0.1;
                }

                // 3. Self-attention (single token attending to itself)
                // Q = embedded * W_q, K = embedded * W_k, V = embedded * W_v
                let q = Self::matvec(&w_q, &embedded);
                let k = Self::matvec(&w_k, &embedded);
                let v = Self::matvec(&w_v, &embedded);

                // Attention score (self): q . k / sqrt(d)
                let attn_score = Self::dot(&q, &k) * scale;
                let attn_weight = Self::softmax_scalar(attn_score);

                // Context = attn_weight * v
                let context: Vec<f64> = v.iter().map(|&vi| attn_weight * vi).collect();

                // 4. Feed-forward: ReLU(context * W_ff1) -> W_out
                let mut ff_hidden = vec![0.0_f64; d];
                for j in 0..d {
                    let mut s = 0.0;
                    for i in 0..d {
                        s += context[i] * w_ff1[(i, j)];
                    }
                    ff_hidden[j] = s.max(0.0); // ReLU
                }

                // Output: ff_hidden * W_out
                let y_hat = Self::dot(&ff_hidden, w_out.as_slice().unwrap());
                let y_true = y_norm[sample + seq];

                let error = y_hat - y_true;
                epoch_loss += error * error;

                // Backward pass (simplified gradient descent)
                let dy = 2.0 * error;

                // Gradient w.r.t. w_out
                for j in 0..d {
                    w_out[j] -= lr * dy * ff_hidden[j];
                }

                // Gradient w.r.t. w_ff1 (via ff_hidden)
                let d_ff_hidden: Vec<f64> = (0..d)
                    .map(|j| {
                        if ff_hidden[j] > 0.0 {
                            dy * w_out[j]
                        } else {
                            0.0
                        }
                    })
                    .collect();

                for i in 0..d {
                    for j in 0..d {
                        w_ff1[(i, j)] -= lr * d_ff_hidden[j] * context[i];
                    }
                }

                // Gradient w.r.t. context
                let d_context: Vec<f64> = (0..d)
                    .map(|j| {
                        let mut s = 0.0;
                        for jj in 0..d {
                            s += d_ff_hidden[jj] * w_ff1[(j, jj)];
                        }
                        s
                    })
                    .collect();

                // Gradient w.r.t. w_v
                for i in 0..d {
                    for j in 0..d {
                        w_v[(i, j)] -= lr * d_context[j] * embedded[i] * attn_weight;
                    }
                }

                // Gradient w.r.t. embedded (via attention path)
                let d_embedded: Vec<f64> =
                    (0..d).map(|j| d_context[j] * attn_weight * v[j]).collect();

                // Gradient w.r.t. w_embed
                for t in 0..seq {
                    for j in 0..d {
                        w_embed[(t, j)] -= lr * d_embedded[j] * x_seq[t];
                    }
                }
            }

            final_mse = epoch_loss / n_samples as f64;
            if epoch > 10 && final_mse < 1e-6 {
                break;
            }
        }

        // Generate fitted values
        let mut fitted = Array1::zeros(n);
        for sample in 0..n_samples {
            let x_seq: Vec<f64> = (0..seq).map(|t| y_norm[sample + t]).collect();
            let y_hat = Self::forward(
                &x_seq, &w_embed, &w_q, &w_k, &w_v, &w_ff1, &w_out, d, seq, scale,
            );
            fitted[sample + seq] = y_hat * y_std + y_mean;
        }

        // Forecast
        let mut forecast = Array1::zeros(n_fc);
        let mut last_seq: Vec<f64> = (0..seq).map(|i| y_norm[n - seq + i]).collect();

        for fc in 0..n_fc {
            let y_hat = Self::forward(
                &last_seq, &w_embed, &w_q, &w_k, &w_v, &w_ff1, &w_out, d, seq, scale,
            );
            forecast[fc] = y_hat * y_std + y_mean;
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

        Ok(TransformerResult {
            fitted,
            forecast,
            n_heads: 1,
            d_model: d,
            seq_len: seq,
            learning_rate: lr,
            n_epochs: epochs,
            mse: final_mse,
            r_squared,
            n_samples,
            n_obs: n,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        x_seq: &[f64],
        w_embed: &Array2<f64>,
        w_q: &Array2<f64>,
        w_k: &Array2<f64>,
        w_v: &Array2<f64>,
        w_ff1: &Array2<f64>,
        w_out: &Array1<f64>,
        d: usize,
        seq: usize,
        scale: f64,
    ) -> f64 {
        // Embed
        let mut embedded = vec![0.0_f64; d];
        for j in 0..d {
            for t in 0..seq {
                embedded[j] += x_seq[t] * w_embed[(t, j)];
            }
        }

        // Positional encoding
        for (j, emb) in embedded.iter_mut().enumerate().take(d) {
            let pos = j as f64;
            let pe = if j % 2 == 0 {
                (pos * 0.1).sin()
            } else {
                (pos * 0.1).cos()
            };
            *emb += pe * 0.1;
        }

        // Attention
        let q = Self::matvec(w_q, &embedded);
        let k = Self::matvec(w_k, &embedded);
        let v = Self::matvec(w_v, &embedded);
        let attn_score = Self::dot(&q, &k) * scale;
        let attn_weight = Self::softmax_scalar(attn_score);
        let context: Vec<f64> = v.iter().map(|&vi| attn_weight * vi).collect();

        // Feed-forward
        let mut ff_hidden = vec![0.0_f64; d];
        for j in 0..d {
            let mut s = 0.0;
            for i in 0..d {
                s += context[i] * w_ff1[(i, j)];
            }
            ff_hidden[j] = s.max(0.0);
        }

        Self::dot(&ff_hidden, w_out.as_slice().unwrap())
    }

    fn init_matrix(m: &mut Array2<f64>, rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                m[(i, j)] = Self::rand_uniform() * 0.1 - 0.05;
            }
        }
    }

    fn matvec(m: &Array2<f64>, v: &[f64]) -> Vec<f64> {
        let rows = m.nrows();
        let cols = m.ncols();
        let mut result = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..cols.min(v.len()) {
                result[i] += m[(i, j)] * v[j];
            }
        }
        result
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn softmax_scalar(x: f64) -> f64 {
        // For single element, softmax = 1.0, but we use sigmoid-like
        // to allow gradient flow
        1.0 / (1.0 + (-x).exp())
    }

    fn rand_uniform() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(2360679774) };
        }
        STATE.with(|s| {
            let mut state = s.get();
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.set(state);
            ((state >> 11) as f64) / (1u64 << 53) as f64
        })
    }
}
