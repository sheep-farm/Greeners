//! K-Means Clustering (MacQueen 1967; Lloyd 1982).
//!
//! Partition n observations into k clusters by minimizing
//! within-cluster sum of squares (WCSS / inertia):
//!
//!   min  sum_{i=1}^n ||x_i - mu_{c_i}||^2
//!
//! Algorithm (Lloyd's algorithm):
//!   1. Initialize centroids (k-means++)
//!   2. Assign each point to nearest centroid
//!   3. Update centroids as mean of assigned points
//!   4. Repeat until convergence
//!
//! Reports cluster labels, centroids, inertia, and silhouette-like
//! separation metric.

use crate::GreenersError;
use ndarray::Array2;
use std::fmt;

/// Result of K-Means clustering.
#[derive(Debug)]
pub struct KmeansResult {
    /// Cluster assignments (n), values 0..k-1
    pub labels: Vec<usize>,
    /// Cluster centroids (k x d)
    pub centroids: Array2<f64>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Within-cluster sum of squares (inertia)
    pub inertia: f64,
    /// Number of iterations until convergence
    pub n_iter: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Between-cluster sum of squares
    pub between_ss: f64,
    /// Total sum of squares
    pub total_ss: f64,
}

impl fmt::Display for KmeansResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " K-Means Clustering ")?;
        writeln!(f, "MacQueen (1967); Lloyd (1982)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Clusters:", self.n_clusters)?;
        writeln!(f, "{:<20} {:>12}", "Iterations:", self.n_iter)?;
        writeln!(f, "{:<20} {:>12.6}", "Inertia (WCSS):", self.inertia)?;
        writeln!(f, "{:<20} {:>12.6}", "Between SS:", self.between_ss)?;
        writeln!(f, "{:<20} {:>12.6}", "Total SS:", self.total_ss)?;
        let pct = if self.total_ss > 1e-15 {
            self.between_ss / self.total_ss * 100.0
        } else {
            0.0
        };
        writeln!(f, "{:<20} {:>12.2}%", "% explained:", pct)?;

        // Cluster sizes
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Cluster sizes:")?;
        writeln!(f, "  {:<10} {:>8}", "Cluster", "Size")?;
        writeln!(f, "{:-^78}", "")?;
        for (i, &size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  {:<10} {:>8}", i, size)?;
        }

        // Centroids
        writeln!(f, "\n  Centroids:")?;
        write!(f, "  {:<10}", "Cluster")?;
        for j in 0..self.n_features {
            write!(f, " {:>10}", format!("x{}", j + 1))?;
        }
        writeln!(f)?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.n_clusters {
            write!(f, "  {:<10}", i)?;
            for j in 0..self.n_features {
                write!(f, " {:>10.4}", self.centroids[(i, j)])?;
            }
            writeln!(f)?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct KMeans;

impl KMeans {
    /// Perform K-Means clustering.
    ///
    /// # Arguments
    /// * `x` - Data matrix (n x d)
    /// * `n_clusters` - Number of clusters k
    /// * `max_iter` - Max iterations (default 300)
    /// * `n_init` - Number of random restarts (default 10)
    pub fn fit(
        x: &Array2<f64>,
        n_clusters: usize,
        max_iter: Option<usize>,
        n_init: Option<usize>,
    ) -> Result<KmeansResult, GreenersError> {
        let n = x.nrows();
        let d = x.ncols();
        if n < n_clusters {
            return Err(GreenersError::InvalidOperation(
                "KMeans: need more observations than clusters".into(),
            ));
        }
        if n_clusters < 1 {
            return Err(GreenersError::InvalidOperation(
                "KMeans: need at least 1 cluster".into(),
            ));
        }

        let max_iterations = max_iter.unwrap_or(300);
        let n_restarts = n_init.unwrap_or(10);

        // Overall mean for between SS
        let overall_mean = (0..d)
            .map(|j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64)
            .collect::<Vec<f64>>();

        let total_ss: f64 = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| (x[(i, j)] - overall_mean[j]).powi(2))
                    .sum::<f64>()
            })
            .sum();

        // Run k-means multiple times, keep best
        let mut best_labels = Vec::new();
        let mut best_centroids = Array2::zeros((n_clusters, d));
        let mut best_inertia = f64::INFINITY;
        let mut best_n_iter = 0;

        for _ in 0..n_restarts {
            let (labels, centroids, inertia, n_iter) =
                Self::kmeans_once(x, n, d, n_clusters, max_iterations);

            if inertia < best_inertia {
                best_inertia = inertia;
                best_labels = labels;
                best_centroids = centroids;
                best_n_iter = n_iter;
            }
        }

        // Cluster sizes
        let mut cluster_sizes = vec![0_usize; n_clusters];
        for &label in &best_labels {
            if label < n_clusters {
                cluster_sizes[label] += 1;
            }
        }

        // Between-cluster SS
        let between_ss: f64 = (0..n_clusters)
            .map(|c| {
                if cluster_sizes[c] == 0 {
                    return 0.0;
                }
                (0..d)
                    .map(|j| {
                        cluster_sizes[c] as f64 * (best_centroids[(c, j)] - overall_mean[j]).powi(2)
                    })
                    .sum::<f64>()
            })
            .sum();

        Ok(KmeansResult {
            labels: best_labels,
            centroids: best_centroids,
            n_clusters,
            inertia: best_inertia,
            n_iter: best_n_iter,
            n_obs: n,
            n_features: d,
            cluster_sizes,
            between_ss,
            total_ss,
        })
    }

    fn kmeans_once(
        x: &Array2<f64>,
        n: usize,
        d: usize,
        k: usize,
        max_iter: usize,
    ) -> (Vec<usize>, Array2<f64>, f64, usize) {
        // k-means++ initialization
        let mut centroids = Array2::<f64>::zeros((k, d));
        let first = Self::rand_int(n);
        for j in 0..d {
            centroids[(0, j)] = x[(first, j)];
        }

        for c in 1..k {
            let mut dists = vec![f64::INFINITY; n];
            for i in 0..n {
                for cc in 0..c {
                    let dist: f64 = (0..d)
                        .map(|j| (x[(i, j)] - centroids[(cc, j)]).powi(2))
                        .sum();
                    if dist < dists[i] {
                        dists[i] = dist;
                    }
                }
            }
            let total: f64 = dists.iter().sum();
            if total < 1e-15 {
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

        // Lloyd's algorithm
        let mut labels = vec![0_usize; n];
        let mut inertia = 0.0;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            n_iter = iter + 1;
            // Assign
            let mut new_labels = vec![0_usize; n];
            let mut new_inertia = 0.0;
            for i in 0..n {
                let mut best_dist = f64::INFINITY;
                let mut best_c = 0;
                for c in 0..k {
                    let dist: f64 = (0..d)
                        .map(|j| (x[(i, j)] - centroids[(c, j)]).powi(2))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                new_labels[i] = best_c;
                new_inertia += best_dist;
            }

            if new_labels == labels {
                inertia = new_inertia;
                break;
            }
            labels = new_labels;
            inertia = new_inertia;

            // Update
            let mut sums = Array2::<f64>::zeros((k, d));
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

        (labels, centroids, inertia, n_iter)
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
            static STATE: Cell<u64> = const { Cell::new(1122334455) };
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
