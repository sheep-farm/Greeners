//! Gaussian Mixture Model via Expectation-Maximization
//! (Dempster, Laird & Rubin 1977).
//!
//! Probabilistic clustering assuming data is generated from
//! a mixture of K Gaussian distributions:
//!
//!   p(x) = sum_{k=1}^K pi_k * N(x | mu_k, Sigma_k)
//!
//! EM algorithm:
//!   E-step: Compute responsibilities gamma_ik = P(z=k | x_i)
//!   M-step: Update pi_k, mu_k, Sigma_k
//!
//! Reports cluster labels, means, covariances, log-likelihood,
//! BIC, and AIC.

use crate::linalg::{LinalgDeterminant as _, LinalgInverse as _};
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of GMM clustering.
#[derive(Debug)]
pub struct GmmResult {
    /// Cluster assignments (n), values 0..k-1
    pub labels: Vec<usize>,
    /// Cluster means (k x d)
    pub means: Array2<f64>,
    /// Cluster covariances (k x d x d), stored as k matrices
    pub covariances: Vec<Array2<f64>>,
    /// Mixing weights (k)
    pub weights: Array1<f64>,
    /// Responsibilities (n x k)
    pub responsibilities: Array2<f64>,
    /// Final log-likelihood
    pub log_likelihood: f64,
    /// BIC
    pub bic: f64,
    /// AIC
    pub aic: f64,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of EM iterations
    pub n_iter: usize,
    /// Whether converged
    pub converged: bool,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
}

impl fmt::Display for GmmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Gaussian Mixture Model ")?;
        writeln!(f, "Dempster, Laird & Rubin (1977)")?;
        writeln!(f, "Expectation-Maximization")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Clusters:", self.n_clusters)?;
        writeln!(f, "{:<20} {:>12}", "Iterations:", self.n_iter)?;
        writeln!(f, "{:<20} {:>12}", "Converged:", self.converged)?;
        writeln!(f, "{:<20} {:>12.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        // Weights
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Mixing weights:")?;
        writeln!(f, "  {:<10} {:>12}", "Cluster", "Weight")?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.n_clusters {
            writeln!(f, "  {:<10} {:>12.4}", i, self.weights[i])?;
        }

        // Means
        writeln!(f, "\n  Cluster means:")?;
        write!(f, "  {:<10}", "Cluster")?;
        for j in 0..self.n_features {
            write!(f, " {:>10}", format!("x{}", j + 1))?;
        }
        writeln!(f)?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.n_clusters {
            write!(f, "  {:<10}", i)?;
            for j in 0..self.n_features {
                write!(f, " {:>10.4}", self.means[(i, j)])?;
            }
            writeln!(f)?;
        }

        // Cluster sizes
        writeln!(f, "\n  Cluster sizes:")?;
        for i in 0..self.n_clusters {
            let size = self.labels.iter().filter(|&&l| l == i).count();
            writeln!(f, "  Cluster {}: {} obs", i, size)?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct GmmClustering;

impl GmmClustering {
    /// Fit GMM via EM algorithm.
    ///
    /// # Arguments
    /// * `x` - Data matrix (n x d)
    /// * `n_clusters` - Number of mixture components k
    /// * `max_iter` - Max EM iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-6)
    pub fn fit(
        x: &Array2<f64>,
        n_clusters: usize,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> Result<GmmResult, GreenersError> {
        let n = x.nrows();
        let d = x.ncols();
        if n < n_clusters * 2 {
            return Err(GreenersError::InvalidOperation(
                "GMM: need more observations".into(),
            ));
        }
        if n_clusters < 1 {
            return Err(GreenersError::InvalidOperation(
                "GMM: need at least 1 cluster".into(),
            ));
        }

        let max_iterations = max_iter.unwrap_or(100);
        let tolerance = tol.unwrap_or(1e-6);
        let k = n_clusters;

        // Initialize means via k-means++ style initialization
        let mut means = Array2::<f64>::zeros((k, d));
        let first = Self::rand_int(n);
        for j in 0..d {
            means[(0, j)] = x[(first, j)];
        }
        for c in 1..k {
            let mut dists = vec![f64::INFINITY; n];
            for i in 0..n {
                for cc in 0..c {
                    let dist: f64 = (0..d).map(|j| (x[(i, j)] - means[(cc, j)]).powi(2)).sum();
                    if dist < dists[i] {
                        dists[i] = dist;
                    }
                }
            }
            let total: f64 = dists.iter().sum();
            if total < 1e-15 {
                let idx = Self::rand_int(n);
                for j in 0..d {
                    means[(c, j)] = x[(idx, j)];
                }
                continue;
            }
            let r = Self::rand_uniform() * total;
            let mut cumsum = 0.0;
            let mut chosen = 0;
            for (i, &di) in dists.iter().enumerate().take(n) {
                cumsum += di;
                if cumsum >= r {
                    chosen = i;
                    break;
                }
            }
            for j in 0..d {
                means[(c, j)] = x[(chosen, j)];
            }
        }

        // Initialize covariances as identity, weights as uniform
        let mut covariances: Vec<Array2<f64>> =
            (0..k).map(|_| Array2::<f64>::eye(d) * 0.1).collect();
        let mut weights = Array1::from_elem(k, 1.0 / k as f64);

        // Compute global variance for initialization
        let global_mean: Array1<f64> = (0..d)
            .map(|j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64)
            .collect();
        let global_var: f64 = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| (x[(i, j)] - global_mean[j]).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / (n * d) as f64;
        let init_var = global_var.max(1e-4);
        for cov in covariances.iter_mut() {
            *cov = Array2::<f64>::eye(d) * init_var;
        }

        let mut log_likelihood = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0;

        let mut resp = Array2::zeros((n, k));

        for iter in 0..max_iterations {
            n_iter = iter + 1;

            // E-step: compute responsibilities
            let mut ll = 0.0;

            for i in 0..n {
                let mut probs = vec![0.0; k];
                let mut sum = 0.0;
                for c in 0..k {
                    let prob = Self::gaussian_pdf(
                        &x.row(i).to_owned(),
                        &means.row(c).to_owned(),
                        &covariances[c],
                    );
                    probs[c] = weights[c] * prob;
                    sum += probs[c];
                }
                if sum < 1e-300 {
                    // Assign uniformly if degenerate
                    for c in 0..k {
                        resp[(i, c)] = 1.0 / k as f64;
                    }
                } else {
                    for c in 0..k {
                        resp[(i, c)] = probs[c] / sum;
                    }
                }
                ll += sum.ln().max(-300.0);
            }

            // Check convergence
            if (ll - log_likelihood).abs() < tolerance {
                log_likelihood = ll;
                converged = true;
                break;
            }
            log_likelihood = ll;

            // M-step: update parameters
            for c in 0..k {
                let n_c: f64 = (0..n).map(|i| resp[(i, c)]).sum();
                if n_c < 1e-10 {
                    continue;
                }

                // Update weights
                weights[c] = n_c / n as f64;

                // Update means
                for j in 0..d {
                    means[(c, j)] = (0..n).map(|i| resp[(i, c)] * x[(i, j)]).sum::<f64>() / n_c;
                }

                // Update covariances
                let mut cov = Array2::zeros((d, d));
                for i in 0..n {
                    let diff: Array1<f64> = (0..d).map(|j| x[(i, j)] - means[(c, j)]).collect();
                    for a in 0..d {
                        for b in 0..d {
                            cov[(a, b)] += resp[(i, c)] * diff[a] * diff[b];
                        }
                    }
                }
                for a in 0..d {
                    for b in 0..d {
                        cov[(a, b)] /= n_c;
                    }
                }
                // Add regularization
                for a in 0..d {
                    cov[(a, a)] += 1e-6;
                }
                covariances[c] = cov;
            }
        }

        // Assign labels (hard assignment)
        let labels: Vec<usize> = (0..n)
            .map(|i| {
                let mut best_c = 0;
                let mut best_resp = 0.0;
                for c in 0..k {
                    if resp[(i, c)] > best_resp {
                        best_resp = resp[(i, c)];
                        best_c = c;
                    }
                }
                best_c
            })
            .collect();

        // BIC and AIC
        // Number of free parameters: k-1 (weights) + k*d (means) + k*d*(d+1)/2 (covariances)
        let n_params = (k - 1) + k * d + k * d * (d + 1) / 2;
        let bic = -2.0 * log_likelihood + n_params as f64 * (n as f64).ln();
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;

        Ok(GmmResult {
            labels,
            means,
            covariances,
            weights,
            responsibilities: resp,
            log_likelihood,
            bic,
            aic,
            n_clusters: k,
            n_iter,
            converged,
            n_obs: n,
            n_features: d,
        })
    }

    fn gaussian_pdf(x: &Array1<f64>, mean: &Array1<f64>, cov: &Array2<f64>) -> f64 {
        let d = x.len();
        let diff = x - mean;
        let cov_inv = match cov.inv() {
            Ok(v) => v,
            Err(_) => return 1e-300,
        };
        let det = cov.det().unwrap_or(1e-300).max(1e-300);

        // exp(-0.5 * diff' * cov_inv * diff)
        let quad = diff.dot(&cov_inv.dot(&diff));
        let norm = (2.0 * std::f64::consts::PI).powi(d as i32) * det;
        (norm).sqrt().recip() * (-0.5 * quad).exp()
    }

    fn rand_int(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (Self::rand_uniform() * n as f64) as usize
    }

    fn rand_uniform() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(9988776655) };
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
