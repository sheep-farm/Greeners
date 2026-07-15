//! Quantile Regression Forest (Meinshausen 2006).
//!
//! Extends Random Forest to estimate conditional quantiles.
//! Instead of storing only the mean in each leaf, QRF stores
//! all training responses. At prediction time, the weighted
//! average of leaf responses across all trees yields a
//! conditional distribution estimate, from which any quantile
//! can be extracted.
//!
//! Algorithm:
//! 1. Grow trees exactly as in Random Forest (bootstrap + mtry)
//! 2. For each leaf node, store all y values (not just the mean)
//! 3. For prediction at x: traverse all trees, collect the leaf
//!    weights, and compute the weighted empirical CDF of y
//! 4. Return the desired quantile(s) from the CDF

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// A single QRF tree node.
#[derive(Debug, Clone)]
struct QrfNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<QrfNode>>,
    right: Option<Box<QrfNode>>,
    /// All y values in this leaf
    leaf_values: Vec<f64>,
    is_leaf: bool,
}

/// Result of Quantile Regression Forest estimation.
#[derive(Debug)]
pub struct QrfResult {
    /// Predicted quantiles (n x n_quantiles)
    pub quantile_predictions: Array2<f64>,
    /// The quantile levels requested
    pub quantiles: Vec<f64>,
    /// Feature importance (cumulative impurity decrease)
    pub feature_importance: Array1<f64>,
    /// OOB R-squared (using median prediction)
    pub oob_r_squared: f64,
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

impl fmt::Display for QrfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Quantile Regression Forest ")?;
        writeln!(f, "Meinshausen (2006)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Trees:", self.n_trees)?;
        writeln!(f, "{:<20} {:>12}", "Max depth:", self.max_depth)?;
        writeln!(f, "{:<20} {:>12.6}", "OOB R² (median):", self.oob_r_squared)?;

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

        // Quantile predictions at selected observations
        writeln!(f, "\n  Quantile predictions (first 5 obs):")?;
        let n_show = 5.min(self.n_obs);
        let header: String = self
            .quantiles
            .iter()
            .map(|q| format!("q{:.2}", q))
            .collect::<Vec<_>>()
            .join("  ");
        writeln!(f, "  {:<8} {}", "Obs", header)?;
        for i in 0..n_show {
            let vals: Vec<String> = (0..self.quantiles.len())
                .map(|j| format!("{:>10.4}", self.quantile_predictions[(i, j)]))
                .collect();
            writeln!(f, "  {:<8} {}", i + 1, vals.join("  "))?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct QRF;

impl QRF {
    /// Estimate Quantile Regression Forest.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Features (n x k)
    /// * `quantiles` - Quantile levels to estimate (e.g., [0.1, 0.5, 0.9])
    /// * `n_trees` - Number of trees
    /// * `max_depth` - Maximum tree depth
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        quantiles: Vec<f64>,
        n_trees: usize,
        max_depth: usize,
        variable_names: Option<Vec<String>>,
    ) -> Result<QrfResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n < 5 || k == 0 {
            return Err(GreenersError::InvalidOperation(
                "QRF: too few observations or features".into(),
            ));
        }
        if n_trees == 0 {
            return Err(GreenersError::InvalidOperation(
                "QRF: n_trees must be >= 1".into(),
            ));
        }
        for &q in &quantiles {
            if q <= 0.0 || q >= 1.0 {
                return Err(GreenersError::InvalidOperation(
                    "QRF: quantiles must be in (0, 1)".into(),
                ));
            }
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let mtry = (k as f64).sqrt().ceil() as usize;
        let mtry = mtry.max(1).min(k);

        let mut trees: Vec<QrfNode> = Vec::with_capacity(n_trees);
        let mut feature_importance = Array1::zeros(k);

        // OOB tracking
        let mut oob_preds: Vec<Vec<f64>> = vec![Vec::new(); n];

        for _ in 0..n_trees {
            // Bootstrap sample
            let mut boot_indices = Vec::with_capacity(n);
            let mut oob_indices = Vec::with_capacity(n);
            let mut in_boot = vec![false; n];
            for _ in 0..n {
                let idx = Self::rand_int(n);
                boot_indices.push(idx);
                in_boot[idx] = true;
            }
            for (i, &in_boot_val) in in_boot.iter().enumerate().take(n) {
                if !in_boot_val {
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

            // OOB: collect leaf values for each OOB observation
            for &i in &oob_indices {
                let leaf_vals = Self::get_leaf_values(&tree, &x.row(i).to_owned());
                oob_preds[i].extend(leaf_vals);
            }

            trees.push(tree);
        }

        // Predict quantiles for all observations
        let n_q = quantiles.len();
        let mut quantile_predictions = Array2::zeros((n, n_q));

        for i in 0..n {
            // Collect leaf values from all trees
            let mut all_vals: Vec<f64> = Vec::new();
            for tree in &trees {
                let leaf_vals = Self::get_leaf_values(tree, &x.row(i).to_owned());
                all_vals.extend(leaf_vals);
            }

            // Sort and extract quantiles
            all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for (j, &q) in quantiles.iter().enumerate() {
                quantile_predictions[(i, j)] = Self::weighted_quantile(&all_vals, q);
            }
        }

        // OOB R-squared using median
        let y_median = Self::weighted_quantile(&y.to_vec(), 0.5);
        let mut oob_median_preds = Array1::zeros(n);
        for i in 0..n {
            if oob_preds[i].is_empty() {
                oob_median_preds[i] = y_median;
            } else {
                let mut sorted = oob_preds[i].clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                oob_median_preds[i] = Self::weighted_quantile(&sorted, 0.5);
            }
        }
        let tss = y.mapv(|v| (v - y.mean().unwrap_or(0.0)).powi(2)).sum();
        let sse = y
            .iter()
            .zip(oob_median_preds.iter())
            .map(|(a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let oob_r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        Ok(QrfResult {
            quantile_predictions,
            quantiles,
            feature_importance,
            oob_r_squared,
            n_trees,
            max_depth,
            n_obs: n,
            n_features: k,
            variable_names: names,
        })
    }

    /// Build a QRF tree (stores all leaf values).
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
    ) -> QrfNode {
        let n = indices.len();
        let leaf_values: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

        if n < 5 || depth >= max_depth {
            return QrfNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_values,
                is_leaf: true,
            };
        }

        let (best_feature, best_threshold, best_gain) =
            Self::find_best_split(y, x, indices, mtry, k);

        if best_gain < 1e-10 || best_feature >= k {
            return QrfNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_values,
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
            return QrfNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_values,
                is_leaf: true,
            };
        }

        QrfNode {
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
            leaf_values: Vec::new(),
            is_leaf: false,
        }
    }

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

        let mut features: Vec<usize> = (0..k).collect();
        for i in 0..features.len() {
            let j = i + Self::rand_int(features.len() - i);
            features.swap(i, j);
        }
        let features = &features[..mtry.min(features.len())];

        let mut best_feature = k;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        for &feat in features {
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

    /// Get all leaf values for a given observation.
    fn get_leaf_values(tree: &QrfNode, x: &Array1<f64>) -> Vec<f64> {
        if tree.is_leaf {
            return tree.leaf_values.clone();
        }
        if x[tree.feature] <= tree.threshold {
            if let Some(ref left) = tree.left {
                return Self::get_leaf_values(left, x);
            }
        } else if let Some(ref right) = tree.right {
            return Self::get_leaf_values(right, x);
        }
        Vec::new()
    }

    /// Compute weighted quantile from sorted values.
    fn weighted_quantile(sorted: &[f64], q: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let n = sorted.len();
        let pos = q * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        if lower == upper {
            return sorted[lower];
        }
        let frac = pos - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
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
            static STATE: Cell<u64> = const { Cell::new(7778889990) };
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
