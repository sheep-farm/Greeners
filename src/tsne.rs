//! t-SNE: t-Distributed Stochastic Neighbor Embedding
//! (van der Maaten & Hinton 2008).
//!
//! Nonlinear dimensionality reduction for visualization.
//! Maps high-dimensional data to 2D (or 3D) preserving
//! local structure.
//!
//! Algorithm:
//!   1. Compute pairwise affinities in high-D (Gaussian kernel)
//!   2. Compute pairwise affinities in low-D (Student-t kernel)
//!   3. Minimize KL divergence via gradient descent
//!
//! This simplified implementation uses:
//!   - Fixed perplexity for bandwidth
//!   - Momentum-based gradient descent
//!   - Early exaggeration

use crate::GreenersError;
use ndarray::Array2;
use std::fmt;

/// Result of t-SNE.
#[derive(Debug)]
pub struct TsneResult {
    /// 2D embedding (n x 2)
    pub embedding: Array2<f64>,
    /// Final KL divergence
    pub kl_divergence: f64,
    /// Number of iterations
    pub n_iter: usize,
    /// Perplexity
    pub perplexity: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of original features
    pub n_features: usize,
    /// Number of output dimensions
    pub n_components: usize,
    /// Iterations array (for plotting)
    pub costs: Vec<f64>,
}

impl fmt::Display for TsneResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " t-SNE ")?;
        writeln!(f, "van der Maaten & Hinton (2008)")?;
        writeln!(f, "t-Distributed Stochastic Neighbor Embedding")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Input features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Output dims:", self.n_components)?;
        writeln!(f, "{:<20} {:>12.2}", "Perplexity:", self.perplexity)?;
        writeln!(f, "{:<20} {:>12.2}", "Learning rate:", self.learning_rate)?;
        writeln!(f, "{:<20} {:>12}", "Iterations:", self.n_iter)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Final KL divergence:", self.kl_divergence
        )?;

        // Embedding (first 10 points)
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  2D embedding (first 10 points):")?;
        writeln!(f, "  {:<6} {:>12} {:>12}", "Obs", "Dim 1", "Dim 2")?;
        writeln!(f, "{:-^78}", "")?;
        let n_show = 10.min(self.n_obs);
        for i in 0..n_show {
            writeln!(
                f,
                "  {:<6} {:>12.4} {:>12.4}",
                i + 1,
                self.embedding[(i, 0)],
                self.embedding[(i, 1)]
            )?;
        }

        // Cost history (every 50 iterations)
        if !self.costs.is_empty() {
            writeln!(f, "\n  KL divergence history:")?;
            writeln!(f, "  {:<10} {:>12}", "Iter", "KL")?;
            writeln!(f, "{:-^78}", "")?;
            for (i, cost) in self.costs.iter().enumerate() {
                if i % 50 == 0 || i == self.costs.len() - 1 {
                    writeln!(f, "  {:<10} {:>12.6}", i + 1, cost)?;
                }
            }
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct TSNE;

impl TSNE {
    /// Run t-SNE dimensionality reduction.
    ///
    /// # Arguments
    /// * `x` - High-dimensional data (n x d)
    /// * `perplexity` - Target perplexity (default 30.0)
    /// * `n_components` - Output dimensions (default 2)
    /// * `max_iter` - Max iterations (default 500)
    /// * `learning_rate` - Gradient descent learning rate (default 200.0)
    pub fn fit(
        x: &Array2<f64>,
        perplexity: Option<f64>,
        n_components: Option<usize>,
        max_iter: Option<usize>,
        learning_rate: Option<f64>,
    ) -> Result<TsneResult, GreenersError> {
        let n = x.nrows();
        let d = x.ncols();
        if n < 5 {
            return Err(GreenersError::InvalidOperation(
                "TSNE: need at least 5 observations".into(),
            ));
        }

        let perp = perplexity.unwrap_or(30.0).min(n as f64 / 3.0).max(5.0);
        let n_comp = n_components.unwrap_or(2);
        let max_iterations = max_iter.unwrap_or(500);
        let lr = learning_rate.unwrap_or(200.0);

        // Step 1: Compute pairwise distances in high-D
        let mut h_dists = Array2::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let dist: f64 = (0..d)
                    .map(|f| (x[(i, f)] - x[(j, f)]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                h_dists[(i, j)] = dist;
                h_dists[(j, i)] = dist;
            }
        }

        // Step 2: Compute high-D affinities P (symmetric Gaussian)
        let mut p = Array2::zeros((n, n));
        let target_entropy = (perp * 2.0_f64.ln()).max(1e-10);

        for i in 0..n {
            // Binary search for sigma_i
            let mut sigma = 1.0;
            let mut sigma_min = 0.1;
            let mut sigma_max = 100.0;

            for _ in 0..50 {
                let mut sum_p = 0.0;
                let mut entropy = 0.0;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let p_ij = (-h_dists[(i, j)].powi(2) / (2.0 * sigma * sigma)).exp();
                    p[(i, j)] = p_ij;
                    sum_p += p_ij;
                }
                if sum_p < 1e-15 {
                    sigma = (sigma_min + sigma_max) / 2.0;
                    continue;
                }
                for j in 0..n {
                    if i != j {
                        p[(i, j)] /= sum_p;
                        if p[(i, j)] > 1e-15 {
                            entropy -= p[(i, j)] * p[(i, j)].ln();
                        }
                    }
                }
                if entropy > target_entropy {
                    sigma_max = sigma;
                    sigma = (sigma_min + sigma) / 2.0;
                } else {
                    sigma_min = sigma;
                    sigma = (sigma + sigma_max) / 2.0;
                }
            }
        }

        // Symmetrize P
        for i in 0..n {
            for j in 0..n {
                p[(i, j)] = (p[(i, j)] + p[(j, i)]) / (2.0 * n as f64);
            }
        }
        // Clamp P to avoid log(0)
        for i in 0..n {
            for j in 0..n {
                p[(i, j)] = p[(i, j)].max(1e-12);
            }
        }

        // Step 3: Initialize low-D embedding randomly
        let mut y = Array2::zeros((n, n_comp));
        for i in 0..n {
            for j in 0..n_comp {
                y[(i, j)] = (Self::rand_normal() * 1e-4).clamp(-1e-4, 1e-4);
            }
        }

        // Gradient descent with momentum
        let mut prev_grad = Array2::<f64>::zeros((n, n_comp));
        let mut costs = Vec::new();
        let mut kl = 0.0;
        let mut momentum = 0.5;
        let early_exaggeration = 4.0;
        let exaggeration_end = 100;

        for iter in 0..max_iterations {
            let exaggeration = if iter < exaggeration_end {
                early_exaggeration
            } else {
                1.0
            };

            // Compute low-D affinities Q (Student-t)
            let mut q = Array2::zeros((n, n));
            let mut sum_q = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let dist: f64 = (0..n_comp)
                        .map(|f| (y[(i, f)] - y[(j, f)]).powi(2))
                        .sum::<f64>();
                    let q_ij = 1.0 / (1.0 + dist);
                    q[(i, j)] = q_ij;
                    sum_q += q_ij;
                }
            }
            sum_q = sum_q.max(1e-12);
            for i in 0..n {
                for j in 0..n {
                    q[(i, j)] /= sum_q;
                    q[(i, j)] = q[(i, j)].max(1e-12);
                }
            }

            // KL divergence
            kl = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        kl += p[(i, j)] * (p[(i, j)] / q[(i, j)]).ln();
                    }
                }
            }
            costs.push(kl);

            // Gradient
            let mut grad = Array2::zeros((n, n_comp));
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let pq = exaggeration * p[(i, j)] - q[(i, j)];
                    let dist: f64 = (0..n_comp)
                        .map(|f| (y[(i, f)] - y[(j, f)]).powi(2))
                        .sum::<f64>();
                    let factor = pq * 4.0 / (1.0 + dist);
                    for f in 0..n_comp {
                        grad[(i, f)] += factor * (y[(i, f)] - y[(j, f)]);
                    }
                }
                // Scale gradient
                for f in 0..n_comp {
                    grad[(i, f)] *= -lr;
                }
            }

            // Update with momentum
            for i in 0..n {
                for f in 0..n_comp {
                    y[(i, f)] += grad[(i, f)] + momentum * prev_grad[(i, f)];
                    prev_grad[(i, f)] = grad[(i, f)];
                }
            }

            // Re-center
            for f in 0..n_comp {
                let mean: f64 = (0..n).map(|i| y[(i, f)]).sum::<f64>() / n as f64;
                for i in 0..n {
                    y[(i, f)] -= mean;
                }
            }

            // Update momentum
            if iter == 250 {
                momentum = 0.8;
            }

            // Check convergence
            if iter > 100 && (costs[iter] - costs[iter - 1]).abs() < 1e-7 {
                break;
            }
        }

        Ok(TsneResult {
            embedding: y,
            kl_divergence: kl,
            n_iter: costs.len(),
            perplexity: perp,
            learning_rate: lr,
            n_obs: n,
            n_features: d,
            n_components: n_comp,
            costs,
        })
    }

    fn rand_normal() -> f64 {
        let u1 = Self::rand_uniform().max(1e-10);
        let u2 = Self::rand_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn rand_uniform() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(7766554433) };
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
