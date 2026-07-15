//! Hierarchical Clustering: Agglomerative (Ward, Single, Complete,
//! Average linkage) (Ward 1963; Johnson 1967).
//!
//! Bottom-up clustering that builds a tree (dendrogram) by
//! successively merging closest clusters.
//!
//! Algorithm:
//!   1. Start with each point as its own cluster
//!   2. Find the two closest clusters (by linkage criterion)
//!   3. Merge them
//!   4. Repeat until one cluster remains
//!
//! Linkage criteria:
//!   - Ward: minimize variance increase
//!   - Single: min pairwise distance
//!   - Complete: max pairwise distance
//!   - Average: average pairwise distance
//!
//! Reports dendrogram (merge order), cluster labels at specified
//! cut height, and cophenetic distances.

use crate::GreenersError;
use ndarray::Array2;
use std::fmt;

/// Linkage method.
#[derive(Debug, Clone, Copy)]
pub enum Linkage {
    Ward,
    Single,
    Complete,
    Average,
}

impl Linkage {
    fn as_str(&self) -> &str {
        match self {
            Linkage::Ward => "Ward",
            Linkage::Single => "Single",
            Linkage::Complete => "Complete",
            Linkage::Average => "Average",
        }
    }
}

/// A merge step in the dendrogram.
#[derive(Debug, Clone)]
pub struct Merge {
    /// Index of first cluster merged
    pub cluster_a: usize,
    /// Index of second cluster merged
    pub cluster_b: usize,
    /// Distance (height) at which merge occurs
    pub distance: f64,
    /// Size of resulting cluster
    pub size: usize,
}

/// Result of hierarchical clustering.
#[derive(Debug)]
pub struct HierarchicalResult {
    /// Merge sequence (n-1 merges)
    pub merges: Vec<Merge>,
    /// Cluster labels at cut height (n)
    pub labels: Vec<usize>,
    /// Number of clusters at cut
    pub n_clusters: usize,
    /// Cut height used
    pub cut_height: f64,
    /// Linkage method
    pub linkage: Linkage,
    /// Cophenetic correlation coefficient
    pub cophenetic_corr: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Cluster sizes at cut
    pub cluster_sizes: Vec<usize>,
}

impl fmt::Display for HierarchicalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Hierarchical Clustering ")?;
        writeln!(f, "Agglomerative ({} linkage)", self.linkage.as_str())?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Clusters (at cut):", self.n_clusters)?;
        writeln!(f, "{:<20} {:>12.6}", "Cut height:", self.cut_height)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Cophenetic corr.:", self.cophenetic_corr
        )?;

        // Dendrogram (first 10 and last 10 merges)
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Dendrogram merges:")?;
        writeln!(
            f,
            "  {:<8} {:>8} {:>8} {:>10} {:>8}",
            "Step", "Clust A", "Clust B", "Distance", "Size"
        )?;
        writeln!(f, "{:-^78}", "")?;
        let n_merges = self.merges.len();
        let n_show = 10.min(n_merges);
        for i in 0..n_show {
            let m = &self.merges[i];
            writeln!(
                f,
                "  {:<8} {:>8} {:>8} {:>10.4} {:>8}",
                i + 1,
                m.cluster_a,
                m.cluster_b,
                m.distance,
                m.size
            )?;
        }
        if n_merges > n_show * 2 {
            writeln!(f, "  {:>8}", "...")?;
        }
        for i in n_merges.saturating_sub(n_show)..n_merges {
            let m = &self.merges[i];
            writeln!(
                f,
                "  {:<8} {:>8} {:>8} {:>10.4} {:>8}",
                i + 1,
                m.cluster_a,
                m.cluster_b,
                m.distance,
                m.size
            )?;
        }

        // Cluster sizes at cut
        writeln!(f, "\n  Cluster sizes at cut:")?;
        writeln!(f, "  {:<10} {:>8}", "Cluster", "Size")?;
        writeln!(f, "{:-^78}", "")?;
        for (i, &size) in self.cluster_sizes.iter().enumerate() {
            writeln!(f, "  {:<10} {:>8}", i, size)?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct HierarchicalClustering;

impl HierarchicalClustering {
    /// Perform agglomerative hierarchical clustering.
    ///
    /// # Arguments
    /// * `x` - Data matrix (n x d)
    /// * `linkage` - Linkage method
    /// * `cut_height` - Height to cut dendrogram (None = auto, uses 70% of max)
    pub fn fit(
        x: &Array2<f64>,
        linkage: Linkage,
        cut_height: Option<f64>,
    ) -> Result<HierarchicalResult, GreenersError> {
        let n = x.nrows();
        let d = x.ncols();
        if n < 2 {
            return Err(GreenersError::InvalidOperation(
                "HierarchicalClustering: need at least 2 observations".into(),
            ));
        }

        // Compute initial pairwise distances
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

        // Track active clusters: each cluster has members and size
        let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut cluster_active: Vec<bool> = vec![true; n];
        let mut cluster_sizes: Vec<usize> = vec![1; n];
        let mut merges: Vec<Merge> = Vec::with_capacity(n - 1);
        let mut next_cluster_id = n;

        // Lance-Williams style update
        let mut current_dists = dists.clone();
        let mut n_active = n;

        while n_active > 1 {
            // Find minimum distance among active clusters
            let mut min_dist = f64::INFINITY;
            let mut min_i = 0;
            let mut min_j = 1;
            for i in 0..next_cluster_id {
                if !cluster_active[i] {
                    continue;
                }
                for j in i..next_cluster_id {
                    if !cluster_active[j] || i == j {
                        continue;
                    }
                    if current_dists[(i, j)] < min_dist {
                        min_dist = current_dists[(i, j)];
                        min_i = i;
                        min_j = j;
                    }
                }
            }

            // Merge min_i and min_j into new cluster
            let new_id = next_cluster_id;
            let size_a = cluster_sizes[min_i];
            let size_b = cluster_sizes[min_j];
            let new_size = size_a + size_b;

            // Record merge
            merges.push(Merge {
                cluster_a: min_i,
                cluster_b: min_j,
                distance: min_dist,
                size: new_size,
            });

            // Merge members
            let mut new_members = cluster_members[min_i].clone();
            new_members.append(&mut cluster_members[min_j].clone());
            cluster_members.push(new_members);
            cluster_sizes.push(new_size);
            cluster_active.push(true);
            cluster_active[min_i] = false;
            cluster_active[min_j] = false;

            // Update distances for new cluster (Lance-Williams)
            // Extend distance matrix
            let mut new_dists = Array2::zeros((next_cluster_id + 1, next_cluster_id + 1));
            for i in 0..next_cluster_id {
                for j in 0..next_cluster_id {
                    new_dists[(i, j)] = current_dists[(i, j)];
                }
            }
            for k in 0..next_cluster_id {
                if !cluster_active[k] || k == new_id {
                    continue;
                }
                let d_ak = current_dists[(min_i, k)];
                let d_bk = current_dists[(min_j, k)];
                let n_a = size_a as f64;
                let n_b = size_b as f64;
                let n_k = cluster_sizes[k] as f64;

                let new_dist = match linkage {
                    Linkage::Ward => {
                        // Ward: sqrt of variance increase
                        let n_new = n_a + n_b;
                        ((n_a + n_k) * d_ak.powi(2) + (n_b + n_k) * d_bk.powi(2)
                            - n_k * min_dist.powi(2))
                            / (n_new + n_k)
                    }
                    Linkage::Single => d_ak.min(d_bk),
                    Linkage::Complete => d_ak.max(d_bk),
                    Linkage::Average => (n_a * d_ak + n_b * d_bk) / (n_a + n_b),
                };
                new_dists[(new_id, k)] = new_dist;
                new_dists[(k, new_id)] = new_dist;
            }
            new_dists[(new_id, new_id)] = 0.0;
            current_dists = new_dists;

            next_cluster_id += 1;
            n_active -= 1;
        }

        // Determine cut height
        let max_dist = merges.last().map(|m| m.distance).unwrap_or(1.0);
        let cut = cut_height.unwrap_or(max_dist * 0.7);

        // Assign labels by cutting dendrogram
        let labels = Self::cut_dendrogram(&merges, n, cut);
        let n_clusters = labels.iter().max().map(|&m| m + 1).unwrap_or(1);

        // Cluster sizes
        let mut cl_sizes = vec![0_usize; n_clusters];
        for &l in &labels {
            if l < n_clusters {
                cl_sizes[l] += 1;
            }
        }

        // Cophenetic correlation
        let cophenetic_corr = Self::cophenetic_correlation(&dists, &merges, n);

        Ok(HierarchicalResult {
            merges,
            labels,
            n_clusters,
            cut_height: cut,
            linkage,
            cophenetic_corr,
            n_obs: n,
            n_features: d,
            cluster_sizes: cl_sizes,
        })
    }

    fn cut_dendrogram(merges: &[Merge], n: usize, cut: f64) -> Vec<usize> {
        // Start with each point in its own cluster
        let mut cluster_id: Vec<i64> = (0..n as i64).collect();
        let mut next_id = n as i64;

        for m in merges {
            if m.distance > cut {
                break;
            }
            // Merge clusters a and b into new cluster
            let a = m.cluster_a as i64;
            let b = m.cluster_b as i64;
            for label in cluster_id.iter_mut() {
                if *label == a || *label == b {
                    *label = next_id;
                }
            }
            next_id += 1;
        }

        // Relabel to 0..k-1
        let mut label_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut next = 0;
        for label in cluster_id.iter_mut() {
            if let Some(&mapped) = label_map.get(label) {
                *label = mapped as i64;
            } else {
                label_map.insert(*label, next);
                *label = next as i64;
                next += 1;
            }
        }

        cluster_id.iter().map(|&l| l as usize).collect()
    }

    fn cophenetic_correlation(dists: &Array2<f64>, merges: &[Merge], n: usize) -> f64 {
        // Cophenetic distance: height at which two points first merge
        let mut cophenetic = Array2::zeros((n, n));
        let mut cluster_id: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        for m in merges {
            let a = m.cluster_a;
            let b = m.cluster_b;
            let mut merged = cluster_id[a].clone();
            merged.append(&mut cluster_id[b].clone());

            // Set cophenetic distance for all pairs across the two clusters
            for &i in &cluster_id[a] {
                for &j in &cluster_id[b] {
                    cophenetic[(i, j)] = m.distance;
                    cophenetic[(j, i)] = m.distance;
                }
            }

            cluster_id.push(merged);
        }

        // Pearson correlation between original and cophenetic distances
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        let mut count = 0.0;

        for i in 0..n {
            for j in i..n {
                if i == j {
                    continue;
                }
                let x = dists[(i, j)];
                let y = cophenetic[(i, j)];
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
                sum_y2 += y * y;
                count += 1.0;
            }
        }

        if count < 2.0 {
            return 0.0;
        }
        let mean_x = sum_x / count;
        let mean_y = sum_y / count;
        let cov = sum_xy / count - mean_x * mean_y;
        let var_x = sum_x2 / count - mean_x * mean_x;
        let var_y = sum_y2 / count - mean_y * mean_y;
        let denom = (var_x * var_y).sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            cov / denom
        }
    }
}
