//! Multilayer Perceptron (MLP) regression via backpropagation.
//!
//! Single hidden layer with sigmoid activation:
//!
//!   y_hat = W2 * sigmoid(W1 * x + b1) + b2
//!
//! Training: gradient descent with momentum on squared loss.
//! Xavier initialization for weights. Mini-batch optional.
//!
//! Rumelhart, Hinton & Williams (1986). Goodfellow et al. (2016).

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of MLP regression.
#[derive(Debug)]
pub struct MlpResult {
    /// In-sample fitted values
    pub fitted: Array1<f64>,
    /// Input-to-hidden weights (n_hidden x n_features)
    pub w1: Array2<f64>,
    /// Hidden biases (n_hidden)
    pub b1: Array1<f64>,
    /// Hidden-to-output weights (n_features_output x n_hidden) — here 1 x n_hidden
    pub w2: Array2<f64>,
    /// Output bias
    pub b2: f64,
    /// Number of hidden units
    pub n_hidden: usize,
    /// Learning rate used
    pub learning_rate: f64,
    /// Number of epochs
    pub n_epochs: usize,
    /// Final loss (MSE)
    pub final_mse: f64,
    /// In-sample R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for MlpResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Neural Network (MLP) Regression ")?;
        writeln!(f, "Rumelhart-Hinton-Williams (1986)")?;
        writeln!(f, "Single hidden layer, sigmoid activation")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Hidden units:", self.n_hidden)?;
        writeln!(f, "{:<20} {:>12}", "Epochs:", self.n_epochs)?;
        writeln!(f, "{:<20} {:>12.6}", "Learning rate:", self.learning_rate)?;
        writeln!(f, "{:<20} {:>12.6}", "Final MSE:", self.final_mse)?;
        writeln!(f, "{:<20} {:>12.6}", "In-sample R²:", self.r_squared)?;

        // Weight statistics
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Weight statistics:")?;
        let w1_mean = self.w1.mean().unwrap_or(0.0);
        let w1_std = self
            .w1
            .iter()
            .map(|w| (w - w1_mean).powi(2))
            .sum::<f64>()
            .sqrt()
            / (self.w1.len() as f64).sqrt();
        let w2_mean = self.w2.mean().unwrap_or(0.0);
        let w2_std = self
            .w2
            .iter()
            .map(|w| (w - w2_mean).powi(2))
            .sum::<f64>()
            .sqrt()
            / (self.w2.len() as f64).sqrt();
        writeln!(
            f,
            "  W1 (input->hidden):  mean={:.6}, std={:.6}",
            w1_mean, w1_std
        )?;
        writeln!(
            f,
            "  W2 (hidden->output): mean={:.6}, std={:.6}",
            w2_mean, w2_std
        )?;
        writeln!(f, "  b2 (output bias):    {:.6}", self.b2)?;

        write!(f, "{:=^78}", "")
    }
}

pub struct MLP;

impl MLP {
    /// Estimate MLP regression via backpropagation.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Features (n x k)
    /// * `n_hidden` - Number of hidden units
    /// * `learning_rate` - Gradient descent step size
    /// * `n_epochs` - Number of training epochs
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        n_hidden: usize,
        learning_rate: Option<f64>,
        n_epochs: Option<usize>,
        variable_names: Option<Vec<String>>,
    ) -> Result<MlpResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n < 5 || k == 0 {
            return Err(GreenersError::InvalidOperation(
                "MLP: too few observations or features".into(),
            ));
        }
        if n_hidden == 0 {
            return Err(GreenersError::InvalidOperation(
                "MLP: n_hidden must be >= 1".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let lr = learning_rate.unwrap_or(0.01);
        let epochs = n_epochs.unwrap_or(200);

        // Standardize x and y for training stability
        let x_mean: Array1<f64> = (0..k).map(|j| x.column(j).mean().unwrap_or(0.0)).collect();
        let x_std: Array1<f64> = (0..k).map(|j| x.column(j).std(0.0).max(1e-10)).collect();
        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = y.std(0.0).max(1e-10);

        let mut x_std_mat = Array2::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                x_std_mat[(i, j)] = (x[(i, j)] - x_mean[j]) / x_std[j];
            }
        }
        let y_std_arr: Array1<f64> = y.mapv(|v| (v - y_mean) / y_std);

        // Xavier initialization
        let limit1 = (6.0 / (k + n_hidden) as f64).sqrt();
        let mut w1 = Array2::zeros((n_hidden, k));
        let mut b1 = Array1::zeros(n_hidden);
        for i in 0..n_hidden {
            b1[i] = (Self::rand_uniform() * 2.0 - 1.0) * 0.1;
            for j in 0..k {
                w1[(i, j)] = (Self::rand_uniform() * 2.0 - 1.0) * limit1;
            }
        }

        let limit2 = (6.0 / (n_hidden + 1) as f64).sqrt();
        let mut w2 = Array2::zeros((1, n_hidden));
        let mut b2 = 0.0_f64;
        for j in 0..n_hidden {
            w2[(0, j)] = (Self::rand_uniform() * 2.0 - 1.0) * limit2;
        }

        // Momentum
        let mut v_w1 = Array2::zeros((n_hidden, k));
        let mut v_b1 = Array1::zeros(n_hidden);
        let mut v_w2 = Array2::zeros((1, n_hidden));
        let mut v_b2 = 0.0_f64;
        let momentum = 0.9;

        let mut final_mse = 0.0_f64;

        for _epoch in 0..epochs {
            let mut grad_w1 = Array2::zeros((n_hidden, k));
            let mut grad_b1 = Array1::zeros(n_hidden);
            let mut grad_w2 = Array2::zeros((1, n_hidden));
            let mut grad_b2 = 0.0_f64;
            let mut epoch_loss = 0.0_f64;

            for i in 0..n {
                // Forward pass
                let xi = x_std_mat.row(i).to_owned();

                // Hidden layer: h = sigmoid(W1 * x + b1)
                let mut h = Array1::zeros(n_hidden);
                for j in 0..n_hidden {
                    let mut sum = b1[j];
                    for l in 0..k {
                        sum += w1[(j, l)] * xi[l];
                    }
                    h[j] = Self::sigmoid(sum);
                }

                // Output: y_hat = W2 * h + b2
                let mut y_hat = b2;
                for j in 0..n_hidden {
                    y_hat += w2[(0, j)] * h[j];
                }

                // Loss: 0.5 * (y - y_hat)^2
                let err = y_std_arr[i] - y_hat;
                epoch_loss += err * err;

                // Backward pass
                // dL/dy_hat = -(y - y_hat) = -err
                let dy_hat = -err;

                // Grad w2, b2
                for j in 0..n_hidden {
                    grad_w2[(0, j)] += dy_hat * h[j];
                }
                grad_b2 += dy_hat;

                // Grad w1, b1 (chain rule through sigmoid)
                for j in 0..n_hidden {
                    let dh = dy_hat * w2[(0, j)] * h[j] * (1.0 - h[j]);
                    grad_b1[j] += dh;
                    for l in 0..k {
                        grad_w1[(j, l)] += dh * xi[l];
                    }
                }
            }

            // Average gradients
            let nf = n as f64;
            grad_w1 /= nf;
            grad_b1 /= nf;
            grad_w2 /= nf;
            grad_b2 /= nf;

            // Update with momentum
            v_w1 = v_w1 * momentum + &grad_w1 * lr;
            v_b1 = v_b1 * momentum + &grad_b1 * lr;
            v_w2 = v_w2 * momentum + &grad_w2 * lr;
            v_b2 = v_b2 * momentum + grad_b2 * lr;

            w1 += &v_w1;
            b1 += &v_b1;
            w2 += &v_w2;
            b2 += v_b2;

            final_mse = epoch_loss / nf;
        }

        // Compute fitted values (de-standardized)
        let mut fitted = Array1::zeros(n);
        for i in 0..n {
            let xi = x_std_mat.row(i).to_owned();
            let mut h = Array1::zeros(n_hidden);
            for j in 0..n_hidden {
                let mut sum = b1[j];
                for l in 0..k {
                    sum += w1[(j, l)] * xi[l];
                }
                h[j] = Self::sigmoid(sum);
            }
            let mut y_hat = b2;
            for j in 0..n_hidden {
                y_hat += w2[(0, j)] * h[j];
            }
            fitted[i] = y_hat * y_std + y_mean;
        }

        // R-squared
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let sse = y
            .iter()
            .zip(fitted.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        Ok(MlpResult {
            fitted,
            w1,
            b1,
            w2,
            b2,
            n_hidden,
            learning_rate: lr,
            n_epochs: epochs,
            final_mse,
            r_squared,
            n_obs: n,
            n_features: k,
            variable_names: names,
        })
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn rand_uniform() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(5566778899) };
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
