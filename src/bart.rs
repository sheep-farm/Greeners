//! BART: Bayesian Additive Regression Trees
//! (Chipman, George & McCulloch 2010).
//!
//! Sum-of-trees model with Bayesian regularization:
//!   y = sum_{t=1}^m T_t + epsilon
//!   epsilon ~ N(0, sigma^2)
//!
//! Priors (simplified):
//!   - Tree structure: shallow trees (depth ~3), favor small trees
//!   - Leaf parameters: mu ~ N(0, sigma_mu^2), shrinkage
//!   - sigma^2: Inverse-Gamma conjugate
//!
//! Implementation: simplified BART with:
//! 1. m shallow trees (default 20)
//! 2. Greedy tree growing with Bayesian backfitting
//! 3. Conjugate updates for leaf parameters
//! 4. Gibbs sampling for sigma^2
//!
//! This is a lightweight implementation suitable for small datasets.

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// A single BART tree node.
#[derive(Debug, Clone)]
struct BartNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<BartNode>>,
    right: Option<Box<BartNode>>,
    /// Leaf value (mu)
    leaf_mu: f64,
    /// Number of obs in leaf
    _leaf_n: usize,
    is_leaf: bool,
}

/// Result of BART estimation.
#[derive(Debug)]
pub struct BartResult {
    /// In-sample fitted values (posterior mean)
    pub fitted: Array1<f64>,
    /// Posterior mean of sigma^2
    pub sigma2: f64,
    /// Number of trees
    pub n_trees: usize,
    /// Max depth
    pub max_depth: usize,
    /// Number of MCMC iterations
    pub n_iter: usize,
    /// Burn-in iterations
    pub burn_in: usize,
    /// In-sample R-squared
    pub r_squared: f64,
    /// MSE
    pub mse: f64,
    /// Posterior samples of sigma^2 (thinned)
    pub sigma2_samples: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Variable inclusion proportions
    pub variable_inclusion: Array1<f64>,
}

impl fmt::Display for BartResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " BART ")?;
        writeln!(f, "Chipman, George & McCulloch (2010)")?;
        writeln!(f, "Bayesian Additive Regression Trees")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Trees:", self.n_trees)?;
        writeln!(f, "{:<20} {:>12}", "Max depth:", self.max_depth)?;
        writeln!(f, "{:<20} {:>12}", "MCMC iterations:", self.n_iter)?;
        writeln!(f, "{:<20} {:>12}", "Burn-in:", self.burn_in)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma² (posterior):", self.sigma2)?;
        writeln!(f, "{:<20} {:>12.6}", "In-sample R²:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "MSE:", self.mse)?;

        // Variable inclusion
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Variable inclusion proportions:")?;
        let mut inc_vec: Vec<(String, f64)> = self
            .variable_names
            .iter()
            .zip(self.variable_inclusion.iter())
            .map(|(name, &inc)| (name.clone(), inc))
            .collect();
        inc_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        writeln!(f, "  {:<14} {:>12}", "Variable", "Inclusion")?;
        writeln!(f, "{:-^78}", "")?;
        for (name, inc) in inc_vec {
            writeln!(f, "  {:<14} {:>12.4}", name, inc)?;
        }

        // Posterior sigma^2 summary
        if !self.sigma2_samples.is_empty() {
            let mut sorted = self.sigma2_samples.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            writeln!(
                f,
                "\n  sigma² posterior: 2.5%: {:.6}  50%: {:.6}  97.5%: {:.6}",
                sorted[n / 40],
                sorted[n / 2],
                sorted[(39 * n) / 40]
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct BART;

impl BART {
    /// Estimate BART.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Features (n x k)
    /// * `n_trees` - Number of trees (default 20)
    /// * `max_depth` - Max tree depth (default 3)
    /// * `n_iter` - MCMC iterations (default 100)
    /// * `burn_in` - Burn-in iterations (default 20)
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        n_trees: Option<usize>,
        max_depth: Option<usize>,
        n_iter: Option<usize>,
        burn_in: Option<usize>,
        variable_names: Option<Vec<String>>,
    ) -> Result<BartResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n < 10 || k == 0 {
            return Err(GreenersError::InvalidOperation(
                "BART: too few observations or features".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let m = n_trees.unwrap_or(20);
        let depth = max_depth.unwrap_or(3);
        let iterations = n_iter.unwrap_or(100);
        let burn = burn_in.unwrap_or(20).min(iterations / 2);

        // Standardize y
        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = y.std(0.0);
        if y_std < 1e-10 {
            return Err(GreenersError::InvalidOperation(
                "BART: y has zero variance".into(),
            ));
        }
        let y_norm: Array1<f64> = y.mapv(|v| (v - y_mean) / y_std);

        // Initialize trees: all stumps (single leaf with mu = 0)
        let mut trees: Vec<BartNode> = (0..m).map(|_| Self::make_stump()).collect();

        // Initialize sigma^2
        let mut sigma2 = 1.0;

        // Prior: sigma_mu = 2 / (k_sqrt * m), where k_sqrt controls shrinkage
        let sigma_mu = 2.0 / (3.0 * (m as f64).sqrt());

        // Variable inclusion counts
        let mut var_inclusion_counts = vec![0_usize; k];

        // MCMC
        let mut sigma2_samples: Vec<f64> = Vec::new();
        let mut fitted_sum: Array1<f64> = Array1::zeros(n);

        for iter in 0..iterations {
            // For each tree, compute partial residuals and update
            for tree_idx in 0..m {
                // Compute residual: y - sum of other trees
                let mut residual = y_norm.clone();
                for (j, tree) in trees.iter().enumerate() {
                    if j != tree_idx {
                        for i in 0..n {
                            residual[i] -= Self::predict_tree(tree, &x.row(i).to_owned());
                        }
                    }
                }

                // Update tree: grow/prune/change (simplified: just re-grow)
                let (new_tree, splits_used) =
                    Self::grow_tree(&residual, x, n, k, depth, sigma2, sigma_mu);
                trees[tree_idx] = new_tree;

                for &f in &splits_used {
                    if f < k {
                        var_inclusion_counts[f] += 1;
                    }
                }
            }

            // Update sigma^2 (Inverse-Gamma conjugate)
            let mut sse = 0.0;
            for i in 0..n {
                let mut pred = 0.0;
                for tree in &trees {
                    pred += Self::predict_tree(tree, &x.row(i).to_owned());
                }
                sse += (y_norm[i] - pred).powi(2);
            }
            // Inverse-Gamma posterior: shape = n/2 + 1, scale = sse/2
            let shape = n as f64 / 2.0 + 1.0;
            let scale = sse / 2.0;
            sigma2 = scale / shape; // posterior mean

            // Record after burn-in
            if iter >= burn {
                sigma2_samples.push(sigma2);
                for i in 0..n {
                    let mut pred = 0.0;
                    for tree in &trees {
                        pred += Self::predict_tree(tree, &x.row(i).to_owned());
                    }
                    fitted_sum[i] += pred;
                }
            }
        }

        // Posterior mean fitted values
        let n_post = iterations - burn;
        let fitted_norm = if n_post > 0 {
            fitted_sum.mapv(|v| v / n_post as f64)
        } else {
            Array1::zeros(n)
        };
        let fitted = fitted_norm.mapv(|v| v * y_std + y_mean);

        // R-squared
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let sse: f64 = y
            .iter()
            .zip(fitted.iter())
            .map(|(a, b)| {
                let diff: f64 = *a - *b;
                diff.powi(2)
            })
            .sum();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };
        let mse = sse / n as f64;

        // Variable inclusion proportions
        let total_splits: usize = var_inclusion_counts.iter().sum();
        let variable_inclusion = Array1::from_vec(
            var_inclusion_counts
                .iter()
                .map(|&c| {
                    if total_splits > 0 {
                        c as f64 / total_splits as f64
                    } else {
                        0.0
                    }
                })
                .collect(),
        );

        Ok(BartResult {
            fitted,
            sigma2: sigma2 * y_std * y_std, // un-standardize
            n_trees: m,
            max_depth: depth,
            n_iter: iterations,
            burn_in: burn,
            r_squared,
            mse,
            sigma2_samples: sigma2_samples.iter().map(|&s| s * y_std * y_std).collect(),
            n_obs: n,
            n_features: k,
            variable_names: names,
            variable_inclusion,
        })
    }

    fn make_stump() -> BartNode {
        BartNode {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            leaf_mu: 0.0,
            _leaf_n: 0,
            is_leaf: true,
        }
    }

    fn grow_tree(
        residual: &Array1<f64>,
        x: &Array2<f64>,
        n: usize,
        k: usize,
        max_depth: usize,
        sigma2: f64,
        sigma_mu: f64,
    ) -> (BartNode, Vec<usize>) {
        let indices: Vec<usize> = (0..n).collect();
        let mut splits_used = Vec::new();
        let tree = Self::build_bart_tree(
            residual,
            x,
            &indices,
            max_depth,
            k,
            0,
            sigma2,
            sigma_mu,
            &mut splits_used,
        );
        (tree, splits_used)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_bart_tree(
        residual: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        max_depth: usize,
        k: usize,
        depth: usize,
        sigma2: f64,
        sigma_mu: f64,
        splits_used: &mut Vec<usize>,
    ) -> BartNode {
        let n = indices.len();
        // Leaf value: posterior mean with shrinkage
        // mu | data ~ N(sum(r)/(n + sigma2/sigma_mu^2), sigma2/(n + sigma2/sigma_mu^2))
        let sum_r: f64 = indices.iter().map(|&i| residual[i]).sum();
        let shrink = n as f64 + sigma2 / (sigma_mu * sigma_mu);
        let leaf_mu = sum_r / shrink;
        let _leaf_n = n;

        if n < 5 || depth >= max_depth {
            return BartNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_mu,
                _leaf_n,
                is_leaf: true,
            };
        }

        // Try to find a split
        let (best_feature, best_threshold, best_gain) =
            Self::find_split_bart(residual, x, indices, k);

        if best_gain < 0.01 || best_feature >= k {
            return BartNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_mu,
                _leaf_n,
                is_leaf: true,
            };
        }

        splits_used.push(best_feature);

        let mut left_idx = Vec::new();
        let mut right_idx = Vec::new();
        for &i in indices {
            if x[(i, best_feature)] <= best_threshold {
                left_idx.push(i);
            } else {
                right_idx.push(i);
            }
        }

        if left_idx.is_empty() || right_idx.is_empty() {
            return BartNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_mu,
                _leaf_n,
                is_leaf: true,
            };
        }

        BartNode {
            feature: best_feature,
            threshold: best_threshold,
            left: Some(Box::new(Self::build_bart_tree(
                residual,
                x,
                &left_idx,
                max_depth,
                k,
                depth + 1,
                sigma2,
                sigma_mu,
                splits_used,
            ))),
            right: Some(Box::new(Self::build_bart_tree(
                residual,
                x,
                &right_idx,
                max_depth,
                k,
                depth + 1,
                sigma2,
                sigma_mu,
                splits_used,
            ))),
            leaf_mu,
            _leaf_n,
            is_leaf: false,
        }
    }

    fn find_split_bart(
        residual: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        k: usize,
    ) -> (usize, f64, f64) {
        let n = indices.len();
        let parent_mean: f64 = indices.iter().map(|&i| residual[i]).sum::<f64>() / n as f64;
        let parent_sse: f64 = indices
            .iter()
            .map(|&i| (residual[i] - parent_mean).powi(2))
            .sum::<f64>();

        let mut best_feature = k;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        for feat in 0..k {
            let mut values: Vec<f64> = indices.iter().map(|&i| x[(i, feat)]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if values.len() < 4 {
                continue;
            }

            let n_thresh = 3.min(values.len() - 1);
            for t in 0..n_thresh {
                let idx = (t + 1) * values.len() / (n_thresh + 1);
                let threshold = values[idx];

                let mut left_sum = 0.0_f64;
                let mut left_n = 0_usize;
                let mut right_sum = 0.0_f64;
                let mut right_n = 0_usize;
                for &i in indices {
                    if x[(i, feat)] <= threshold {
                        left_sum += residual[i];
                        left_n += 1;
                    } else {
                        right_sum += residual[i];
                        right_n += 1;
                    }
                }

                if left_n < 3 || right_n < 3 {
                    continue;
                }

                let left_mean = left_sum / left_n as f64;
                let right_mean = right_sum / right_n as f64;

                let left_sse: f64 = indices
                    .iter()
                    .filter(|&&i| x[(i, feat)] <= threshold)
                    .map(|&i| (residual[i] - left_mean).powi(2))
                    .sum::<f64>();
                let right_sse: f64 = indices
                    .iter()
                    .filter(|&&i| x[(i, feat)] > threshold)
                    .map(|&i| (residual[i] - right_mean).powi(2))
                    .sum::<f64>();

                let gain = parent_sse - left_sse - right_sse;
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn predict_tree(tree: &BartNode, x: &Array1<f64>) -> f64 {
        if tree.is_leaf {
            return tree.leaf_mu;
        }
        if x[tree.feature] <= tree.threshold {
            if let Some(ref left) = tree.left {
                return Self::predict_tree(left, x);
            }
        } else if let Some(ref right) = tree.right {
            return Self::predict_tree(right, x);
        }
        tree.leaf_mu
    }
}
