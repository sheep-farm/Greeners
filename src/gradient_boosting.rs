//! Gradient Boosting regression (Friedman 2001).
//!
//! Sequential additive model of weak learners (regression trees):
//!
//!   F_m(x) = F_{m-1}(x) + nu * h_m(x)
//!
//! where h_m is a regression tree fit to the pseudo-residuals
//! r_i = y_i - F_{m-1}(x_i), and nu is the learning rate (shrinkage).
//!
//! Features:
//!   - MSE-based split criterion
//!   - Learning rate (shrinkage) for regularization
//!   - Max depth control per tree
//!   - Subsample ratio for stochastic GBM
//!   - Feature importance via impurity decrease accumulation

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// A single decision tree node (shallow, for weak learners).
#[derive(Debug, Clone)]
struct GbTreeNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<GbTreeNode>>,
    right: Option<Box<GbTreeNode>>,
    value: f64,
    is_leaf: bool,
}

/// Result of Gradient Boosting estimation.
#[derive(Debug)]
pub struct GradientBoostingResult {
    /// In-sample fitted values
    pub fitted: Array1<f64>,
    /// Initial prediction (mean of y)
    pub init_value: f64,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Number of trees (iterations)
    pub n_trees: usize,
    /// Max depth per tree
    pub max_depth: usize,
    /// Feature importance (cumulative impurity decrease)
    pub feature_importance: Array1<f64>,
    /// In-sample R-squared
    pub r_squared: f64,
    /// MSE
    pub mse: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for GradientBoostingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Gradient Boosting Regression ")?;
        writeln!(f, "Friedman (2001) — sequential additive trees")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Trees:", self.n_trees)?;
        writeln!(f, "{:<20} {:>12}", "Max depth:", self.max_depth)?;
        writeln!(f, "{:<20} {:>12.6}", "Learning rate:", self.learning_rate)?;
        writeln!(f, "{:<20} {:>12.6}", "Init (mean y):", self.init_value)?;
        writeln!(f, "{:<20} {:>12.6}", "In-sample R²:", self.r_squared)?;
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

pub struct GradientBoosting;

impl GradientBoosting {
    /// Estimate Gradient Boosting regression.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Features (n x k)
    /// * `n_trees` - Number of boosting iterations
    /// * `learning_rate` - Shrinkage parameter (default 0.1)
    /// * `max_depth` - Max depth per tree (default 3)
    /// * `subsample` - Fraction of observations per tree (default 1.0)
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        n_trees: usize,
        learning_rate: Option<f64>,
        max_depth: Option<usize>,
        subsample: Option<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<GradientBoostingResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n < 5 || k == 0 {
            return Err(GreenersError::InvalidOperation(
                "GradientBoosting: too few observations or features".into(),
            ));
        }
        if n_trees == 0 {
            return Err(GreenersError::InvalidOperation(
                "GradientBoosting: n_trees must be >= 1".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let lr = learning_rate.unwrap_or(0.1);
        let depth = max_depth.unwrap_or(3);
        let sub = subsample.unwrap_or(1.0).clamp(0.1, 1.0);

        // Initialize with mean of y
        let init_value = y.mean().unwrap_or(0.0);
        let mut fitted = Array1::from_elem(n, init_value);
        let mut feature_importance = Array1::zeros(k);

        let mut trees: Vec<GbTreeNode> = Vec::with_capacity(n_trees);

        for _ in 0..n_trees {
            // Pseudo-residuals (for squared loss, this is the gradient = y - F)
            let residuals = y - &fitted;

            // Subsample
            let n_sub = (n as f64 * sub).round() as usize;
            let n_sub = n_sub.max(5).min(n);
            let indices: Vec<usize> = if sub < 1.0 {
                (0..n_sub).map(|_| Self::rand_int(n)).collect()
            } else {
                (0..n).collect()
            };

            // Fit tree to residuals
            let tree = Self::build_tree(
                &residuals,
                x,
                &indices,
                depth,
                k,
                0,
                &mut feature_importance,
            );

            // Update fitted values
            for i in 0..n {
                let pred = Self::predict_single(&tree, &x.row(i).to_owned());
                fitted[i] += lr * pred;
            }

            trees.push(tree);
        }

        // R-squared
        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let sse = y
            .iter()
            .zip(fitted.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };
        let mse = sse / n as f64;

        Ok(GradientBoostingResult {
            fitted,
            init_value,
            learning_rate: lr,
            n_trees,
            max_depth: depth,
            feature_importance,
            r_squared,
            mse,
            n_obs: n,
            n_features: k,
            variable_names: names,
        })
    }

    /// Build a shallow regression tree.
    fn build_tree(
        y: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        max_depth: usize,
        k: usize,
        depth: usize,
        importance: &mut Array1<f64>,
    ) -> GbTreeNode {
        let n = indices.len();
        let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n.max(1) as f64;

        if n < 5 || depth >= max_depth {
            return GbTreeNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                value: mean,
                is_leaf: true,
            };
        }

        let (best_feature, best_threshold, best_gain) = Self::find_best_split(y, x, indices, k);

        if best_gain < 1e-10 || best_feature >= k {
            return GbTreeNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                value: mean,
                is_leaf: true,
            };
        }

        importance[best_feature] += best_gain;

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
            return GbTreeNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                value: mean,
                is_leaf: true,
            };
        }

        GbTreeNode {
            feature: best_feature,
            threshold: best_threshold,
            left: Some(Box::new(Self::build_tree(
                y,
                x,
                &left_idx,
                max_depth,
                k,
                depth + 1,
                importance,
            ))),
            right: Some(Box::new(Self::build_tree(
                y,
                x,
                &right_idx,
                max_depth,
                k,
                depth + 1,
                importance,
            ))),
            value: mean,
            is_leaf: false,
        }
    }

    /// Find best split using MSE criterion.
    fn find_best_split(
        y: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        k: usize,
    ) -> (usize, f64, f64) {
        let n = indices.len();
        let parent_mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
        let parent_mse: f64 = indices
            .iter()
            .map(|&i| (y[i] - parent_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        let mut best_feature = k;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        for feat in 0..k {
            let mut values: Vec<f64> = indices.iter().map(|&i| x[(i, feat)]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if values.len() < 2 {
                continue;
            }

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

    fn predict_single(tree: &GbTreeNode, x: &Array1<f64>) -> f64 {
        if tree.is_leaf {
            return tree.value;
        }
        if x[tree.feature] <= tree.threshold {
            if let Some(ref left) = tree.left {
                return Self::predict_single(left, x);
            }
        } else if let Some(ref right) = tree.right {
            return Self::predict_single(right, x);
        }
        tree.value
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
            static STATE: Cell<u64> = const { Cell::new(1112223334) };
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
