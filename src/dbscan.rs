//! DBSCAN: Density-Based Spatial Clustering of Applications
//! with Noise (Ester, Kriegel, Sander & Xu 1996).
//!
//! Density-based clustering that groups points that are closely
//! packed together, marking points in low-density regions as
//! outliers (noise).
//!
//! Algorithm:
//!   1. For each point, find all points within epsilon distance
//!   2. If a point has >= min_pts neighbors, it's a core point
//!   3. Expand clusters from core points (BFS/DFS)
//!   4. Points not in any cluster are noise (label = -1)
//!
//! Advantages over k-means:
//!   - No need to specify k
//!   - Finds arbitrary-shaped clusters
//!   - Robust to outliers

use crate::GreenersError;
use ndarray::Array2;
use std::collections::HashSet;
use std::fmt;

/// Result of DBSCAN clustering.
#[derive(Debug)]
pub struct DbscanResult {
    /// Cluster assignments (n), -1 = noise, 0..k-1 = cluster
    pub labels: Vec<i64>,
    /// Number of clusters (excluding noise)
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
    /// Epsilon (neighborhood radius)
    pub eps: f64,
    /// Minimum points to form a core point
    pub min_pts: usize,
    /// Cluster sizes (including noise at index -1)
    pub cluster_sizes: Vec<usize>,
    /// Number of core points
    pub n_core: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
}

impl fmt::Display for DbscanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " DBSCAN ")?;
        writeln!(f, "Ester, Kriegel, Sander & Xu (1996)")?;
        writeln!(f, "Density-Based Spatial Clustering")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12.6}", "Epsilon:", self.eps)?;
        writeln!(f, "{:<20} {:>12}", "Min pts:", self.min_pts)?;
        writeln!(f, "{:<20} {:>12}", "Clusters:", self.n_clusters)?;
        writeln!(f, "{:<20} {:>12}", "Core points:", self.n_core)?;
        writeln!(f, "{:<20} {:>12}", "Noise points:", self.n_noise)?;

        // Cluster sizes
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Cluster sizes:")?;
        writeln!(f, "  {:<10} {:>8}", "Cluster", "Size")?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(f, "  {:<10} {:>8}", "Noise (-1)", self.n_noise)?;
        for i in 0..self.n_clusters {
            let size = self.labels.iter().filter(|&&l| l == i as i64).count();
            writeln!(f, "  {:<10} {:>8}", i, size)?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct DBSCAN;

impl DBSCAN {
    /// Perform DBSCAN clustering.
    ///
    /// # Arguments
    /// * `x` - Data matrix (n x d)
    /// * `eps` - Neighborhood radius
    /// * `min_pts` - Minimum points to form a core point
    pub fn fit(x: &Array2<f64>, eps: f64, min_pts: usize) -> Result<DbscanResult, GreenersError> {
        let n = x.nrows();
        let d = x.ncols();
        if n < 2 {
            return Err(GreenersError::InvalidOperation(
                "DBSCAN: need at least 2 observations".into(),
            ));
        }
        if eps <= 0.0 {
            return Err(GreenersError::InvalidOperation(
                "DBSCAN: eps must be positive".into(),
            ));
        }
        if min_pts < 2 {
            return Err(GreenersError::InvalidOperation(
                "DBSCAN: min_pts must be at least 2".into(),
            ));
        }

        // Compute pairwise distances
        let dists = Self::compute_dists(x, n, d);

        // Find neighbors for each point
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| i != j && dists[(i, j)] <= eps).collect())
            .collect();

        // Identify core points (>= min_pts neighbors, counting self)
        let is_core: Vec<bool> = neighbors
            .iter()
            .map(|nb| nb.len() + 1 >= min_pts) // +1 for self
            .collect();

        let n_core = is_core.iter().filter(|&&c| c).count();

        // Clustering
        let mut labels = vec![-1_i64; n]; // -1 = unvisited/noise
        let mut cluster_id = 0_i64;

        for i in 0..n {
            if labels[i] != -1 || !is_core[i] {
                continue;
            }

            // Start new cluster
            labels[i] = cluster_id;
            let mut queue = vec![i];
            let mut visited = HashSet::new();
            visited.insert(i);

            while let Some(&p) = queue.last() {
                queue.pop();

                for &nb in &neighbors[p] {
                    if visited.contains(&nb) {
                        continue;
                    }
                    visited.insert(nb);

                    // Assign to cluster (even if not core)
                    if labels[nb] == -1 || labels[nb] != cluster_id {
                        labels[nb] = cluster_id;
                    }

                    // If core point, expand further
                    if is_core[nb] {
                        queue.push(nb);
                    }
                }
            }

            cluster_id += 1;
        }

        let n_clusters = cluster_id as usize;
        let n_noise = labels.iter().filter(|&&l| l == -1).count();

        // Cluster sizes
        let mut cluster_sizes = vec![0_usize; n_clusters];
        for &l in &labels {
            if l >= 0 {
                cluster_sizes[l as usize] += 1;
            }
        }

        Ok(DbscanResult {
            labels,
            n_clusters,
            n_noise,
            eps,
            min_pts,
            cluster_sizes,
            n_core,
            n_obs: n,
            n_features: d,
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
                let dist = dist.sqrt();
                dists[(i, j)] = dist;
                dists[(j, i)] = dist;
            }
        }
        dists
    }
}
