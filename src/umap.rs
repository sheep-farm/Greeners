//! UMAP: Uniform Manifold Approximation and Projection
//! (McInnes, Healy & Melville 2018).
//!
//! Nonlinear dimensionality reduction based on Riemannian
//! geometry and algebraic topology. Preserves both local and
//! global structure better than t-SNE.
//!
//! Algorithm (simplified):
//!   1. Build fuzzy simplicial set (k-NN graph with weights)
//!   2. Initialize low-D embedding (spectral or random)
//!   3. Optimize via cross-entropy minimization
//!
//! This simplified implementation uses:
//!   - k-NN graph with Gaussian kernel
//!   - Fuzzy union for graph construction
//!   - Gradient descent for embedding optimization

use crate::GreenersError;
use ndarray::Array2;
use std::fmt;

/// Result of UMAP.
#[derive(Debug)]
pub struct UmapResult {
    /// Low-D embedding (n x n_components)
    pub embedding: Array2<f64>,
    /// Number of iterations
    pub n_iter: usize,
    /// Final cross-entropy loss
    pub loss: f64,
    /// Number of neighbors (k)
    pub n_neighbors: usize,
    /// Minimum distance
    pub min_dist: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of original features
    pub n_features: usize,
    /// Number of output dimensions
    pub n_components: usize,
    /// Loss history
    pub losses: Vec<f64>,
}

impl fmt::Display for UmapResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " UMAP ")?;
        writeln!(f, "McInnes, Healy & Melville (2018)")?;
        writeln!(f, "Uniform Manifold Approximation and Projection")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Input features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Output dims:", self.n_components)?;
        writeln!(f, "{:<20} {:>12}", "Neighbors (k):", self.n_neighbors)?;
        writeln!(f, "{:<20} {:>12.4}", "Min distance:", self.min_dist)?;
        writeln!(f, "{:<20} {:>12}", "Iterations:", self.n_iter)?;
        writeln!(f, "{:<20} {:>12.6}", "Final loss:", self.loss)?;

        // Embedding (first 10 points)
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Embedding (first 10 points):")?;
        write!(f, "  {:<6}", "Obs")?;
        for j in 0..self.n_components {
            write!(f, " {:>10}", format!("Dim {}", j + 1))?;
        }
        writeln!(f)?;
        writeln!(f, "{:-^78}", "")?;
        let n_show = 10.min(self.n_obs);
        for i in 0..n_show {
            write!(f, "  {:<6}", i + 1)?;
            for j in 0..self.n_components {
                write!(f, " {:>10.4}", self.embedding[(i, j)])?;
            }
            writeln!(f)?;
        }

        // Loss history
        if !self.losses.is_empty() {
            writeln!(f, "\n  Loss history (selected):")?;
            writeln!(f, "  {:<10} {:>12}", "Iter", "Loss")?;
            writeln!(f, "{:-^78}", "")?;
            for (i, &l) in self.losses.iter().enumerate() {
                if i % 50 == 0 || i == self.losses.len() - 1 {
                    writeln!(f, "  {:<10} {:>12.6}", i + 1, l)?;
                }
            }
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct UMAP;

impl UMAP {
    /// Run UMAP dimensionality reduction.
    ///
    /// # Arguments
    /// * `x` - High-dimensional data (n x d)
    /// * `n_neighbors` - k for k-NN graph (default 15)
    /// * `n_components` - Output dimensions (default 2)
    /// * `min_dist` - Minimum embedding distance (default 0.1)
    /// * `max_iter` - Max optimization iterations (default 300)
    pub fn fit(
        x: &Array2<f64>,
        n_neighbors: Option<usize>,
        n_components: Option<usize>,
        min_dist: Option<f64>,
        max_iter: Option<usize>,
    ) -> Result<UmapResult, GreenersError> {
        let n = x.nrows();
        let d = x.ncols();
        if n < 5 {
            return Err(GreenersError::InvalidOperation(
                "UMAP: need at least 5 observations".into(),
            ));
        }

        let k = n_neighbors.unwrap_or(15).min(n - 1).max(2);
        let n_comp = n_components.unwrap_or(2);
        let md = min_dist.unwrap_or(0.1);
        let max_iterations = max_iter.unwrap_or(300);

        // Step 1: Compute pairwise distances
        let mut dists = Array2::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let dist: f64 = (0..d)
                    .map(|f| (x[(i, f)] - x[(j, f)]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                dists[(i, j)] = dist;
                dists[(j, i)] = dist;
            }
        }

        // Step 2: Build fuzzy simplicial set (k-NN graph)
        // For each point, find k nearest neighbors and compute weights
        let mut graph = Array2::zeros((n, n));
        for i in 0..n {
            // Find k nearest neighbors
            let mut neighbors: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, dists[(i, j)]))
                .collect();
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Compute sigma_i (local connectivity)
            let rho_i = neighbors.first().map(|(_, d)| *d).unwrap_or(0.0);

            // Find sigma via binary search (target k-th neighbor distance)
            let target = neighbors[k - 1].1;
            let mut sigma = (target - rho_i).max(1e-5);
            let mut sigma_min = 1e-5;
            let mut sigma_max = 100.0;

            for _ in 0..30 {
                let mut sum_w = 0.0;
                for (_j, dist) in &neighbors {
                    if *dist > rho_i {
                        sum_w += (-(dist - rho_i) / sigma).exp();
                    }
                }
                let log_sum = if sum_w > 0.0 { sum_w.ln() } else { -100.0 };
                let target_log = (k as f64).ln();
                if log_sum < target_log {
                    sigma_min = sigma;
                    sigma = (sigma + sigma_max) / 2.0;
                } else {
                    sigma_max = sigma;
                    sigma = (sigma_min + sigma) / 2.0;
                }
            }

            // Set weights
            for (j, dist) in &neighbors {
                let w = if *dist > rho_i {
                    (-(dist - rho_i) / sigma).exp()
                } else {
                    1.0
                };
                graph[(i, *j)] = w;
            }
        }

        // Symmetrize via fuzzy union: w_sym = w_ij + w_ji - w_ij * w_ji
        let mut sym_graph = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let w_ij = graph[(i, j)];
                let w_ji = graph[(j, i)];
                sym_graph[(i, j)] = w_ij + w_ji - w_ij * w_ji;
            }
        }

        // Step 3: Initialize embedding (random)
        let mut y = Array2::zeros((n, n_comp));
        for i in 0..n {
            for j in 0..n_comp {
                y[(i, j)] = Self::rand_normal() * 0.01;
            }
        }

        // Step 4: Optimize via gradient descent (cross-entropy)
        let mut losses = Vec::new();
        let mut loss = 0.0;
        let lr = 1.0;
        let mut prev_grad = Array2::<f64>::zeros((n, n_comp));

        for iter in 0..max_iterations {
            // Compute low-D distances
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
                    // a, b curve approximation: 1 / (a * dist^b + 1)
                    let a = 1.929;
                    let b = 0.7915;
                    let q_ij = 1.0 / (a * dist.powf(b) + 1.0);
                    q[(i, j)] = q_ij;
                    sum_q += q_ij;
                }
            }
            sum_q = sum_q.max(1e-12);

            // Cross-entropy loss
            loss = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let p_ij = sym_graph[(i, j)].max(1e-12);
                        let q_ij = (q[(i, j)] / sum_q).max(1e-12);
                        loss += p_ij * (p_ij / q_ij).ln();
                    }
                }
            }
            losses.push(loss);

            // Gradient
            let mut grad = Array2::zeros((n, n_comp));
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let p_ij = sym_graph[(i, j)];
                    let q_ij = q[(i, j)] / sum_q;
                    let dist: f64 = (0..n_comp)
                        .map(|f| (y[(i, f)] - y[(j, f)]).powi(2))
                        .sum::<f64>();
                    let a = 1.929;
                    let b = 0.7915;
                    let denom = a * dist.powf(b) + 1.0;
                    let factor = (p_ij - q_ij) * 2.0 * a * b * dist.powf(b - 1.0) / (denom * denom);
                    for f in 0..n_comp {
                        grad[(i, f)] += factor * (y[(i, f)] - y[(j, f)]);
                    }
                }
                for f in 0..n_comp {
                    grad[(i, f)] *= -lr;
                }
            }

            // Update with momentum
            let momentum = if iter < 100 { 0.5 } else { 0.8 };
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
        }

        Ok(UmapResult {
            embedding: y,
            n_iter: losses.len(),
            loss,
            n_neighbors: k,
            min_dist: md,
            n_obs: n,
            n_features: d,
            n_components: n_comp,
            losses,
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
            static STATE: Cell<u64> = const { Cell::new(6655443322) };
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
