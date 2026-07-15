//! XGBoost-style gradient boosting with L1/L2 regularization.
//!
//! Chen & Guestrin (2016). Extends standard gradient boosting
//! with:
//!   - L1 (Lasso) and L2 (Ridge) regularization on leaf weights
//!   - Second-order approximation (Newton's method) using
//!     gradient and Hessian
//!   - Tree-level regularization: gamma (penalty per leaf),
//!     lambda (L2), alpha (L1)
//!
//! Objective:
//!   L = sum l(y_i, y_hat_i) + sum_t [gamma * |T_t| + 0.5 * lambda * ||w_t||^2 + alpha * ||w_t||_1]
//!
//! For squared loss: gradient = y_hat - y, hessian = 1.
//! Leaf weight: w* = -G / (H + lambda), with L1 proximal clipping.
//!
//! Split gain: 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// A single XGBoost tree node.
#[derive(Debug, Clone)]
struct XgbNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<XgbNode>>,
    right: Option<Box<XgbNode>>,
    leaf_weight: f64,
    is_leaf: bool,
}

/// Result of XGBoost estimation.
#[derive(Debug)]
pub struct XgboostResult {
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
    /// L2 regularization (lambda)
    pub lambda: f64,
    /// L1 regularization (alpha)
    pub alpha: f64,
    /// Gamma (leaf penalty)
    pub gamma: f64,
    /// Feature importance (cumulative gain)
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

impl fmt::Display for XgboostResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " XGBoost Regression ")?;
        writeln!(f, "Chen & Guestrin (2016)")?;
        writeln!(f, "L1/L2 regularized gradient boosting (Newton)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Trees:", self.n_trees)?;
        writeln!(f, "{:<20} {:>12}", "Max depth:", self.max_depth)?;
        writeln!(f, "{:<20} {:>12.6}", "Learning rate:", self.learning_rate)?;
        writeln!(f, "{:<20} {:>12.6}", "Lambda (L2):", self.lambda)?;
        writeln!(f, "{:<20} {:>12.6}", "Alpha (L1):", self.alpha)?;
        writeln!(f, "{:<20} {:>12.6}", "Gamma (leaf penalty):", self.gamma)?;
        writeln!(f, "{:<20} {:>12.6}", "In-sample R²:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "MSE:", self.mse)?;

        // Feature importance
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Feature importance (gain):")?;
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

pub struct XGBoost;

impl XGBoost {
    /// Estimate XGBoost regression.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Features (n x k)
    /// * `n_trees` - Number of boosting rounds
    /// * `learning_rate` - Shrinkage (default 0.3)
    /// * `max_depth` - Max tree depth (default 6)
    /// * `lambda` - L2 regularization (default 1.0)
    /// * `alpha` - L1 regularization (default 0.0)
    /// * `gamma` - Leaf penalty (default 0.0)
    /// * `subsample` - Fraction of obs per tree (default 1.0)
    /// * `colsample` - Fraction of features per tree (default 1.0)
    /// * `variable_names` - Optional feature names
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        n_trees: usize,
        learning_rate: Option<f64>,
        max_depth: Option<usize>,
        lambda: Option<f64>,
        alpha: Option<f64>,
        gamma: Option<f64>,
        subsample: Option<f64>,
        colsample: Option<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<XgboostResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n < 5 || k == 0 {
            return Err(GreenersError::InvalidOperation(
                "XGBoost: too few observations or features".into(),
            ));
        }
        if n_trees == 0 {
            return Err(GreenersError::InvalidOperation(
                "XGBoost: n_trees must be >= 1".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let lr = learning_rate.unwrap_or(0.3);
        let depth = max_depth.unwrap_or(6);
        let lam = lambda.unwrap_or(1.0);
        let al = alpha.unwrap_or(0.0);
        let gam = gamma.unwrap_or(0.0);
        let sub = subsample.unwrap_or(1.0).clamp(0.1, 1.0);
        let col = colsample.unwrap_or(1.0).clamp(0.1, 1.0);

        // Initialize with mean of y
        let init_value = y.mean().unwrap_or(0.0);
        let mut fitted = Array1::from_elem(n, init_value);
        let mut feature_importance = Array1::zeros(k);

        for _ in 0..n_trees {
            // Gradients and Hessians for squared loss
            // l(y, y_hat) = 0.5 * (y - y_hat)^2
            // gradient = y_hat - y
            // hessian = 1
            let gradients: Vec<f64> = (0..n).map(|i| fitted[i] - y[i]).collect();
            let hessians: Vec<f64> = vec![1.0; n];

            // Subsample observations
            let n_sub = (n as f64 * sub).round() as usize;
            let n_sub = n_sub.max(5).min(n);
            let obs_indices: Vec<usize> = if sub < 1.0 {
                (0..n_sub).map(|_| Self::rand_int(n)).collect()
            } else {
                (0..n).collect()
            };

            // Subsample features
            let n_feat_sub = (k as f64 * col).round() as usize;
            let n_feat_sub = n_feat_sub.max(1).min(k);
            let mut all_features: Vec<usize> = (0..k).collect();
            for i in 0..all_features.len() {
                let j = i + Self::rand_int(all_features.len() - i);
                all_features.swap(i, j);
            }
            let feat_indices = &all_features[..n_feat_sub];

            // Build tree
            let tree = Self::build_tree(
                x,
                &gradients,
                &hessians,
                &obs_indices,
                feat_indices,
                depth,
                lam,
                al,
                gam,
                0,
                &mut feature_importance,
            );

            // Update fitted values
            for i in 0..n {
                let leaf_weight = Self::predict_single(&tree, &x.row(i).to_owned());
                fitted[i] += lr * leaf_weight;
            }
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

        Ok(XgboostResult {
            fitted,
            init_value,
            learning_rate: lr,
            n_trees,
            max_depth: depth,
            lambda: lam,
            alpha: al,
            gamma: gam,
            feature_importance,
            r_squared,
            mse,
            n_obs: n,
            n_features: k,
            variable_names: names,
        })
    }

    /// Build XGBoost tree using Newton splits.
    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        x: &Array2<f64>,
        gradients: &[f64],
        hessians: &[f64],
        indices: &[usize],
        features: &[usize],
        max_depth: usize,
        lambda: f64,
        alpha: f64,
        gamma: f64,
        depth: usize,
        importance: &mut Array1<f64>,
    ) -> XgbNode {
        let n = indices.len();
        let g_sum: f64 = indices.iter().map(|&i| gradients[i]).sum();
        let h_sum: f64 = indices.iter().map(|&i| hessians[i]).sum();

        // Leaf weight: w* = -(G - alpha*sign(w)) / (H + lambda)
        // Simplified: w* = -G / (H + lambda), with L1 proximal
        let leaf_weight = Self::compute_leaf_weight(g_sum, h_sum, lambda, alpha);

        if n < 5 || depth >= max_depth {
            return XgbNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_weight,
                is_leaf: true,
            };
        }

        let (best_feature, best_threshold, best_gain) =
            Self::find_best_split_xgb(x, gradients, hessians, indices, features, lambda, gamma);

        if best_gain <= 0.0 || best_feature >= x.ncols() {
            return XgbNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_weight,
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
            return XgbNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_weight,
                is_leaf: true,
            };
        }

        XgbNode {
            feature: best_feature,
            threshold: best_threshold,
            left: Some(Box::new(Self::build_tree(
                x,
                gradients,
                hessians,
                &left_idx,
                features,
                max_depth,
                lambda,
                alpha,
                gamma,
                depth + 1,
                importance,
            ))),
            right: Some(Box::new(Self::build_tree(
                x,
                gradients,
                hessians,
                &right_idx,
                features,
                max_depth,
                lambda,
                alpha,
                gamma,
                depth + 1,
                importance,
            ))),
            leaf_weight,
            is_leaf: false,
        }
    }

    /// Compute leaf weight with L1/L2 regularization.
    fn compute_leaf_weight(g_sum: f64, h_sum: f64, lambda: f64, alpha: f64) -> f64 {
        // w* = -(G - alpha * sign(w)) / (H + lambda)
        // Proximal: if |G| < alpha, w = 0 (L1 thresholding)
        if alpha > 0.0 {
            if g_sum.abs() < alpha {
                return 0.0;
            }
            return -(g_sum - alpha * g_sum.signum()) / (h_sum + lambda);
        }
        -g_sum / (h_sum + lambda)
    }

    /// Find best split using XGBoost gain criterion.
    fn find_best_split_xgb(
        x: &Array2<f64>,
        gradients: &[f64],
        hessians: &[f64],
        indices: &[usize],
        features: &[usize],
        lambda: f64,
        gamma: f64,
    ) -> (usize, f64, f64) {
        let g_sum: f64 = indices.iter().map(|&i| gradients[i]).sum();
        let h_sum: f64 = indices.iter().map(|&i| hessians[i]).sum();

        let parent_score = g_sum * g_sum / (h_sum + lambda);

        let mut best_feature = x.ncols();
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

                let mut gl = 0.0_f64;
                let mut hl = 0.0_f64;
                let mut gr = 0.0_f64;
                let mut hr = 0.0_f64;
                for &i in indices {
                    if x[(i, feat)] <= threshold {
                        gl += gradients[i];
                        hl += hessians[i];
                    } else {
                        gr += gradients[i];
                        hr += hessians[i];
                    }
                }

                if hl < 1e-10 || hr < 1e-10 {
                    continue;
                }

                let left_score = gl * gl / (hl + lambda);
                let right_score = gr * gr / (hr + lambda);
                let gain = 0.5 * (left_score + right_score - parent_score) - gamma;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn predict_single(tree: &XgbNode, x: &Array1<f64>) -> f64 {
        if tree.is_leaf {
            return tree.leaf_weight;
        }
        if x[tree.feature] <= tree.threshold {
            if let Some(ref left) = tree.left {
                return Self::predict_single(left, x);
            }
        } else if let Some(ref right) = tree.right {
            return Self::predict_single(right, x);
        }
        tree.leaf_weight
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
            static STATE: Cell<u64> = const { Cell::new(3141592653) };
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
