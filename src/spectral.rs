//! Spectral Clustering (Shi-Malik 2000, Ng-Jordan-Weiss 2002).
//!
//! Clustering via the eigenstructure of a similarity graph.
//! Constructs a similarity graph from data, computes the graph
//! Laplacian, and uses its eigenvectors for clustering.
//!
//! Algorithm (Ng-Jordan-Weiss):
//!   1. Compute affinity matrix A (Gaussian kernel)
//!   2. Compute degree matrix D, normalized Laplacian
//!      L_sym = I - D^{-1/2} A D^{-1/2}
//!   3. Find k smallest eigenvectors of L_sym
//!   4. Normalize rows of eigenvector matrix
//!   5. Cluster rows via k-means
//!
//! This implementation uses a simplified eigenvalue computation
//! via the power iteration / QR approach on small matrices.

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of spectral clustering.
#[derive(Debug)]
pub struct SpectralResult {
    /// Cluster assignments (n), values 0..k-1
    pub labels: Vec<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Affinity matrix (n x n)
    pub affinity: Array2<f64>,
    /// Eigenvalues of Laplacian (k smallest)
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors of Laplacian (n x k)
    pub eigenvectors: Array2<f64>,
    /// Cluster centroids in eigenspace (k x k)
    pub centroids: Array2<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Sigma (Gaussian kernel width)
    pub sigma: f64,
    /// Within-cluster sum of squares
    pub inertia: f64,
}

impl fmt::Display for SpectralResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Spectral Clustering ")?;
        writeln!(f, "Ng-Jordan-Weiss (2002)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Clusters:", self.n_clusters)?;
        writeln!(f, "{:<20} {:>12.6}", "Sigma:", self.sigma)?;
        writeln!(f, "{:<20} {:>12.6}", "Inertia:", self.inertia)?;

        // Eigenvalues
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Smallest eigenvalues of Laplacian:")?;
        writeln!(f, "  {:<8} {:>14}", "Idx", "Eigenvalue")?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.eigenvalues.len() {
            writeln!(f, "  {:<8} {:>14.6}", i + 1, self.eigenvalues[i])?;
        }

        // Cluster sizes
        writeln!(f, "\n  Cluster sizes:")?;
        let mut sizes = vec![0_usize; self.n_clusters];
        for &label in &self.labels {
            if label < self.n_clusters {
                sizes[label] += 1;
            }
        }
        for (i, &size) in sizes.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} obs", i, size)?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct SpectralClustering;

impl SpectralClustering {
    /// Perform spectral clustering.
    ///
    /// # Arguments
    /// * `x` - Data matrix (n x d)
    /// * `n_clusters` - Number of clusters k
    /// * `sigma` - Gaussian kernel width (default: median pairwise distance)
    /// * `max_iter` - Max k-means iterations (default 100)
    pub fn fit(
        x: &Array2<f64>,
        n_clusters: usize,
        sigma: Option<f64>,
        max_iter: Option<usize>,
    ) -> Result<SpectralResult, GreenersError> {
        let n = x.nrows();
        let d = x.ncols();
        if n < n_clusters + 1 {
            return Err(GreenersError::InvalidOperation(
                "SpectralClustering: need more observations than clusters".into(),
            ));
        }
        if n_clusters < 2 {
            return Err(GreenersError::InvalidOperation(
                "SpectralClustering: need at least 2 clusters".into(),
            ));
        }

        // 1. Compute affinity matrix (Gaussian kernel)
        let dists = Self::compute_dists(x, n, d);
        let sigma_val = sigma.unwrap_or_else(|| {
            // Median of non-zero pairwise distances
            let mut all_dists = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    all_dists.push(dists[(i, j)].sqrt());
                }
            }
            all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if all_dists.is_empty() {
                1.0
            } else {
                all_dists[all_dists.len() / 2].max(1e-10)
            }
        });

        let mut affinity = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    affinity[(i, j)] = 0.0;
                } else {
                    affinity[(i, j)] = (-dists[(i, j)] / (2.0 * sigma_val * sigma_val)).exp();
                }
            }
        }

        // 2. Compute normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
        let mut degree = Array1::zeros(n);
        for i in 0..n {
            degree[i] = affinity.row(i).sum();
            if degree[i] < 1e-10 {
                degree[i] = 1e-10;
            }
        }

        let mut d_inv_sqrt = Array1::zeros(n);
        for i in 0..n {
            d_inv_sqrt[i] = 1.0 / degree[i].sqrt();
        }

        let mut laplacian = Array2::eye(n);
        for i in 0..n {
            for j in 0..n {
                laplacian[(i, j)] -= d_inv_sqrt[i] * affinity[(i, j)] * d_inv_sqrt[j];
            }
        }

        // 3. Find k smallest eigenvectors
        // For small matrices, use Jacobi eigenvalue algorithm
        let (eigenvalues, eigenvectors) = Self::smallest_eigvecs(&laplacian, n, n_clusters)?;

        // 4. Normalize rows of eigenvector matrix
        let mut u_norm = Array2::zeros((n, n_clusters));
        for i in 0..n {
            let row_norm: f64 = (0..n_clusters)
                .map(|j| eigenvectors[(i, j)].powi(2))
                .sum::<f64>()
                .sqrt();
            let norm = if row_norm < 1e-10 { 1e-10 } else { row_norm };
            for j in 0..n_clusters {
                u_norm[(i, j)] = eigenvectors[(i, j)] / norm;
            }
        }

        // 5. K-means on normalized eigenvectors
        let iterations = max_iter.unwrap_or(100);
        let (labels, centroids, inertia) = Self::kmeans(&u_norm, n, n_clusters, iterations)?;

        Ok(SpectralResult {
            labels,
            n_clusters,
            affinity,
            eigenvalues,
            eigenvectors: u_norm,
            centroids,
            n_obs: n,
            n_features: d,
            sigma: sigma_val,
            inertia,
        })
    }

    fn compute_dists(x: &Array2<f64>, n: usize, d: usize) -> Array2<f64> {
        let mut dists = Array2::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let mut dist = 0.0;
                for f in 0..d {
                    dist += (x[(i, f)] - x[(j, f)]).powi(2);
                }
                dists[(i, j)] = dist;
                dists[(j, i)] = dist;
            }
        }
        dists
    }

    /// Compute k smallest eigenvalues/vectors via Jacobi rotation.
    /// Returns all eigenvalues, then we pick the k smallest.
    fn smallest_eigvecs(
        a: &Array2<f64>,
        n: usize,
        k: usize,
    ) -> Result<(Array1<f64>, Array2<f64>), GreenersError> {
        // Jacobi eigenvalue algorithm for symmetric matrices
        let mut work = a.clone();
        let mut v = Array2::eye(n);

        let max_sweeps = 100;
        for _ in 0..max_sweeps {
            // Find off-diagonal element with largest magnitude
            let mut max_val = 0.0;
            let mut max_i = 0;
            let mut max_j = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    if work[(i, j)].abs() > max_val {
                        max_val = work[(i, j)].abs();
                        max_i = i;
                        max_j = j;
                    }
                }
            }

            if max_val < 1e-12 {
                break;
            }

            // Jacobi rotation
            let aii = work[(max_i, max_i)];
            let ajj = work[(max_j, max_j)];
            let aij = work[(max_i, max_j)];

            let theta = (ajj - aii) / (2.0 * aij);
            let t = if theta >= 0.0 {
                1.0 / (theta + (1.0 + theta * theta).sqrt())
            } else {
                -1.0 / (-theta + (1.0 + theta * theta).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Apply rotation
            for l in 0..n {
                let ail = work[(max_i, l)];
                let ajl = work[(max_j, l)];
                work[(max_i, l)] = c * ail - s * ajl;
                work[(max_j, l)] = s * ail + c * ajl;
            }
            for l in 0..n {
                let ali = work[(l, max_i)];
                let alj = work[(l, max_j)];
                work[(l, max_i)] = c * ali - s * alj;
                work[(l, max_j)] = s * ali + c * alj;
            }
            for l in 0..n {
                let vli = v[(l, max_i)];
                let vlj = v[(l, max_j)];
                v[(l, max_i)] = c * vli - s * vlj;
                v[(l, max_j)] = s * vli + c * vlj;
            }
        }

        // Extract eigenvalues (diagonal) and sort
        let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (work[(i, i)], i)).collect();
        eig_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let eigenvalues = Array1::from_vec(eig_pairs.iter().take(k).map(|(v, _)| *v).collect());
        let mut eigenvectors = Array2::zeros((n, k));
        for (j, (_, idx)) in eig_pairs.iter().take(k).enumerate() {
            for i in 0..n {
                eigenvectors[(i, j)] = v[(i, *idx)];
            }
        }

        Ok((eigenvalues, eigenvectors))
    }

    fn kmeans(
        x: &Array2<f64>,
        n: usize,
        k: usize,
        max_iter: usize,
    ) -> Result<(Vec<usize>, Array2<f64>, f64), GreenersError> {
        let d = x.ncols();

        // Initialize centroids via k-means++
        let mut centroids = Array2::zeros((k, d));
        let first = Self::rand_int(n);
        for j in 0..d {
            centroids[(0, j)] = x[(first, j)];
        }

        for c in 1..k {
            // Compute distances to nearest centroid
            let mut dists = vec![f64::INFINITY; n];
            for i in 0..n {
                for cc in 0..c {
                    let mut dist = 0.0;
                    for j in 0..d {
                        dist += (x[(i, j)] - centroids[(cc, j)]).powi(2);
                    }
                    if dist < dists[i] {
                        dists[i] = dist;
                    }
                }
            }
            // Weighted random selection
            let total: f64 = dists.iter().sum();
            if total < 1e-15 {
                // All points are at centroids, pick random
                let idx = Self::rand_int(n);
                for j in 0..d {
                    centroids[(c, j)] = x[(idx, j)];
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
                centroids[(c, j)] = x[(chosen, j)];
            }
        }

        // K-means iterations
        let mut labels = vec![0_usize; n];
        let mut inertia = 0.0;

        for _ in 0..max_iter {
            // Assign
            let mut new_labels = vec![0_usize; n];
            let mut new_inertia = 0.0;
            for i in 0..n {
                let mut best_dist = f64::INFINITY;
                let mut best_c = 0;
                for c in 0..k {
                    let mut dist = 0.0;
                    for j in 0..d {
                        dist += (x[(i, j)] - centroids[(c, j)]).powi(2);
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                new_labels[i] = best_c;
                new_inertia += best_dist;
            }

            // Check convergence
            if new_labels == labels {
                inertia = new_inertia;
                break;
            }
            labels = new_labels;
            inertia = new_inertia;

            // Update centroids
            let mut sums: Array2<f64> = Array2::zeros((k, d));
            let mut counts = vec![0_usize; k];
            for i in 0..n {
                let c = labels[i];
                counts[c] += 1;
                for j in 0..d {
                    sums[(c, j)] += x[(i, j)];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..d {
                        centroids[(c, j)] = sums[(c, j)] / counts[c] as f64;
                    }
                }
            }
        }

        Ok((labels, centroids, inertia))
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
            static STATE: Cell<u64> = const { Cell::new(1357902468) };
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
