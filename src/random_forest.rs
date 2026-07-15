//! Random Forest regression.
//!
//! Breiman (2001). Ensemble of decision trees for regression.
//! Each tree is trained on a bootstrap sample with random feature
//! subsampling at each split. Predictions are averaged across trees.
//!
//! Implementation: CART-style decision trees with:
//!   - Bootstrap sampling (bagging)
//!   - mtry = sqrt(n_features) features per split
//!   - MSE-based split criterion
//!   - Max depth control
//!   - OOB (out-of-bag) error estimation

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// A single decision tree node.
#[derive(Debug, Clone)]
struct TreeNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    prediction: f64,
    is_leaf: bool,
}

/// Result of Random Forest estimation.
#[derive(Debug)]
pub struct RandomForestResult {
    /// Predictions (in-sample fitted values)
    pub fitted: Array1<f64>,
    /// Out-of-bag (OOB) predictions
    pub oob_predictions: Array1<f64>,
    /// Feature importance (sum of impurity decrease per feature)
    pub feature_importance: Array1<f64>,
    /// OOB R-squared
    pub oob_r_squared: f64,
    /// In-sample R-squared
    pub r_squared: f64,
    /// MSE
    pub mse: f64,
    /// Number of trees
    pub n_trees: usize,
    /// Max depth
    pub max_depth: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for RandomForestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Random Forest Regression ")?;
        writeln!(f, "Breiman (2001) — ensemble of decision trees")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Trees:", self.n_trees)?;
        writeln!(f, "{:<20} {:>12}", "Max depth:", self.max_depth)?;
        writeln!(f, "{:<20} {:>12.6}", "In-sample R²:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "OOB R²:", self.oob_r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "MSE:", self.mse)?;

        // Feature importance
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Feature importance:")?;
        let total: f64 = self.feature_importance.sum().max(1e-10);
        let mut imp_vec: Vec<(String, f64, f64)> = self
            .variable_names
            .iter()
            .zip(self.feature_importance.iter())
            .map(|(name, &imp)| (name.clone(), imp, imp / total * 100.0))
            .collect();
        imp_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        writeln!(
            f,
            "  {:<14} {:>12} {:>10}",
            "Variable", "Importance", "% Total"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for (name, imp, pct) in imp_vec {
            writeln!(f, "  {:<14} {:>12.6} {:>9.4}%", name, imp, pct)?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct RandomForest;

impl RandomForest {
    /// Estimate Random Forest regression.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Features (n x k)
    /// * `n_trees` - Number of trees in the forest
    /// * `max_depth` - Maximum tree depth
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        n_trees: usize,
        max_depth: usize,
        variable_names: Option<Vec<String>>,
    ) -> Result<RandomForestResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n < 5 || k == 0 {
            return Err(GreenersError::InvalidOperation(
                "RandomForest: too few observations or features".into(),
            ));
        }
        if n_trees == 0 {
            return Err(GreenersError::InvalidOperation(
                "RandomForest: n_trees must be >= 1".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let mtry = (k as f64).sqrt().ceil() as usize;
        let mtry = mtry.max(1).min(k);

        let mut trees: Vec<TreeNode> = Vec::with_capacity(n_trees);
        let mut feature_importance = Array1::zeros(k);
        let mut oob_sum = Array1::<f64>::zeros(n);
        let mut oob_count = Array1::<f64>::zeros(n);

        for _tree in 0..n_trees {
            // Bootstrap sample
            let mut boot_indices = Vec::with_capacity(n);
            let mut oob_indices = Vec::with_capacity(n);
            let mut in_boot = vec![false; n];
            for _ in 0..n {
                let idx = Self::rand_int(n);
                boot_indices.push(idx);
                in_boot[idx] = true;
            }
            for (i, &in_b) in in_boot.iter().enumerate().take(n) {
                if !in_b {
                    oob_indices.push(i);
                }
            }

            // Build tree
            let tree = Self::build_tree(
                y,
                x,
                &boot_indices,
                max_depth,
                mtry,
                k,
                0,
                &mut feature_importance,
            );

            // OOB predictions
            for &i in &oob_indices {
                let pred = Self::predict_single(&tree, &x.row(i).to_owned());
                oob_sum[i] += pred;
                oob_count[i] += 1.0;
            }

            trees.push(tree);
        }

        // In-sample predictions (average across all trees)
        let mut fitted = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for tree in &trees {
                sum += Self::predict_single(tree, &x.row(i).to_owned());
            }
            fitted[i] = sum / n_trees as f64;
        }

        // OOB predictions
        let mut oob_predictions = Array1::zeros(n);
        for i in 0..n {
            if oob_count[i] > 0.0 {
                oob_predictions[i] = oob_sum[i] / oob_count[i];
            } else {
                oob_predictions[i] = y.mean().unwrap_or(0.0);
            }
        }

        // R-squared
        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let in_sample_sse = y
            .iter()
            .zip(fitted.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 1e-15 {
            1.0 - in_sample_sse / tss
        } else {
            0.0
        };
        let mse = in_sample_sse / n as f64;

        // OOB R-squared (only for observations with OOB predictions)
        let oob_sse: f64 = y
            .iter()
            .zip(oob_predictions.iter())
            .zip(oob_count.iter())
            .filter(|(_, &c)| c > 0.0)
            .map(|((&yv, &pred), _)| (yv - pred).powi(2))
            .sum();
        let oob_r_squared = if tss > 1e-15 {
            1.0 - oob_sse / tss
        } else {
            0.0
        };

        Ok(RandomForestResult {
            fitted,
            oob_predictions,
            feature_importance,
            oob_r_squared,
            r_squared,
            mse,
            n_trees,
            max_depth,
            n_obs: n,
            n_features: k,
            variable_names: names,
        })
    }

    /// Build a decision tree recursively.
    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        y: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        max_depth: usize,
        mtry: usize,
        k: usize,
        depth: usize,
        importance: &mut Array1<f64>,
    ) -> TreeNode {
        let n = indices.len();

        // Leaf prediction: mean of y
        let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n.max(1) as f64;

        // Stopping criteria
        if n < 5 || depth >= max_depth {
            return TreeNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                prediction: mean,
                is_leaf: true,
            };
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) =
            Self::find_best_split(y, x, indices, mtry, k);

        if best_gain < 1e-10 || best_feature >= k {
            return TreeNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                prediction: mean,
                is_leaf: true,
            };
        }

        // Update feature importance
        importance[best_feature] += best_gain;

        // Split
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
            return TreeNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                prediction: mean,
                is_leaf: true,
            };
        }

        TreeNode {
            feature: best_feature,
            threshold: best_threshold,
            left: Some(Box::new(Self::build_tree(
                y,
                x,
                &left_idx,
                max_depth,
                mtry,
                k,
                depth + 1,
                importance,
            ))),
            right: Some(Box::new(Self::build_tree(
                y,
                x,
                &right_idx,
                max_depth,
                mtry,
                k,
                depth + 1,
                importance,
            ))),
            prediction: mean,
            is_leaf: false,
        }
    }

    /// Find the best split using MSE criterion.
    fn find_best_split(
        y: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        mtry: usize,
        k: usize,
    ) -> (usize, f64, f64) {
        let n = indices.len();
        let parent_mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
        let parent_mse: f64 = indices
            .iter()
            .map(|&i| (y[i] - parent_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        // Random feature subset
        let mut features: Vec<usize> = (0..k).collect();
        // Shuffle and take first mtry
        for i in 0..features.len() {
            let j = i + Self::rand_int(features.len() - i);
            features.swap(i, j);
        }
        let features = &features[..mtry.min(features.len())];

        let mut best_feature = k; // invalid
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        for &feat in features {
            // Candidate thresholds: percentiles of x[:, feat]
            let mut values: Vec<f64> = indices.iter().map(|&i| x[(i, feat)]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if values.len() < 2 {
                continue;
            }

            // Try a few thresholds
            let n_thresh = 5.min(values.len() - 1);
            for t in 0..n_thresh {
                let idx = (t + 1) * values.len() / (n_thresh + 1);
                let threshold = values[idx];

                let mut left_sum = 0.0;
                let mut left_n = 0;
                let mut right_sum = 0.0;
                let mut right_n = 0;
                for &i in indices {
                    if x[(i, feat)] <= threshold {
                        left_sum += y[i];
                        left_n += 1;
                    } else {
                        right_sum += y[i];
                        right_n += 1;
                    }
                }

                if left_n == 0 || right_n == 0 {
                    continue;
                }

                let left_mean = left_sum / left_n as f64;
                let right_mean = right_sum / right_n as f64;

                let left_mse: f64 = indices
                    .iter()
                    .filter(|&&i| x[(i, feat)] <= threshold)
                    .map(|&i| (y[i] - left_mean).powi(2))
                    .sum::<f64>()
                    / n as f64;
                let right_mse: f64 = indices
                    .iter()
                    .filter(|&&i| x[(i, feat)] > threshold)
                    .map(|&i| (y[i] - right_mean).powi(2))
                    .sum::<f64>()
                    / n as f64;

                let gain = parent_mse - left_mse - right_mse;
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    /// Predict for a single observation.
    fn predict_single(tree: &TreeNode, x: &Array1<f64>) -> f64 {
        if tree.is_leaf {
            return tree.prediction;
        }
        if x[tree.feature] <= tree.threshold {
            if let Some(ref left) = tree.left {
                return Self::predict_single(left, x);
            }
        } else if let Some(ref right) = tree.right {
            return Self::predict_single(right, x);
        }
        tree.prediction
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
            static STATE: Cell<u64> = const { Cell::new(987654321) };
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
