//! Causal Forest (Wager & Athey 2018).
//!
//! Estimates heterogeneous treatment effects (HTE) via random
//! forests of causal trees. Each tree is a "honest" causal tree:
//!   1. Split sample: one half for splitting, one half for
//!      estimating leaf treatment effects
//!   2. Splitting criterion maximizes variance of treatment
//!      effect estimates across leaves
//!   3. Leaf effect: tau_hat = mean(Y|treated, leaf) - mean(Y|control, leaf)
//!
//! The forest averages predictions across all trees. Treatment
//! effect heterogeneity is captured via leaf assignments.
//!
//! Requires: treatment indicator T, outcome Y, features X.
//! Assumes: unconfoundedness, overlap, SUTVA.

use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// A single causal tree node.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CausalNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<CausalNode>>,
    right: Option<Box<CausalNode>>,
    /// Treatment effect estimate in this leaf
    leaf_effect: f64,
    /// Number of treated in leaf
    leaf_n_treated: usize,
    /// Number of control in leaf
    leaf_n_control: usize,
    /// Leaf variance of effect (for SE)
    leaf_var: f64,
    is_leaf: bool,
}

/// Result of Causal Forest estimation.
#[derive(Debug)]
pub struct CausalForestResult {
    /// Predicted treatment effect for each observation (n)
    pub treatment_effects: Array1<f64>,
    /// Average treatment effect (ATE)
    pub ate: f64,
    /// Standard error of ATE
    pub ate_se: f64,
    /// 95% CI for ATE
    pub ate_ci: [f64; 2],
    /// Feature importance (heterogeneity-based)
    pub feature_importance: Array1<f64>,
    /// Heterogeneity: SD of treatment effects
    pub heterogeneity: f64,
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

impl fmt::Display for CausalForestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Causal Forest ")?;
        writeln!(f, "Wager & Athey (2018)")?;
        writeln!(f, "Honest causal trees for heterogeneous treatment effects")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Trees:", self.n_trees)?;
        writeln!(f, "{:<20} {:>12}", "Max depth:", self.max_depth)?;
        writeln!(f, "{:<20} {:>12.6}", "ATE:", self.ate)?;
        writeln!(f, "{:<20} {:>12.6}", "ATE SE:", self.ate_se)?;
        writeln!(
            f,
            "{:<20} [{:.4}, {:.4}]",
            "95% CI:", self.ate_ci[0], self.ate_ci[1]
        )?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Heterogeneity (SD):", self.heterogeneity
        )?;

        // Feature importance
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Feature importance (heterogeneity):")?;
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

        // Distribution of treatment effects
        writeln!(f, "\n  Treatment effect distribution:")?;
        let mut sorted_te = self.treatment_effects.to_vec();
        sorted_te.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_te.len();
        writeln!(
            f,
            "  Min: {:>10.4}  Q1: {:>10.4}  Median: {:>10.4}  Q3: {:>10.4}  Max: {:>10.4}",
            sorted_te[0],
            sorted_te[n / 4],
            sorted_te[n / 2],
            sorted_te[3 * n / 4],
            sorted_te[n - 1]
        )?;

        write!(f, "{:=^78}", "")
    }
}

pub struct CausalForest;

impl CausalForest {
    /// Estimate Causal Forest.
    ///
    /// # Arguments
    /// * `y` - Outcome (n)
    /// * `t` - Treatment indicator (n), true if treated
    /// * `x` - Features (n x k)
    /// * `n_trees` - Number of trees (default 100)
    /// * `max_depth` - Max tree depth (default 5)
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        t: &[bool],
        x: &Array2<f64>,
        n_trees: Option<usize>,
        max_depth: Option<usize>,
        variable_names: Option<Vec<String>>,
    ) -> Result<CausalForestResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if t.len() != n || x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "CausalForest: dimension mismatch".into(),
            ));
        }
        if n < 20 {
            return Err(GreenersError::InvalidOperation(
                "CausalForest: need at least 20 observations".into(),
            ));
        }

        let n_treated = t.iter().filter(|&&t| t).count();
        let n_control = n - n_treated;
        if n_treated < 5 || n_control < 5 {
            return Err(GreenersError::InvalidOperation(
                "CausalForest: need at least 5 treated and 5 control".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let trees = n_trees.unwrap_or(100);
        let depth = max_depth.unwrap_or(5);
        let mtry = (k as f64).sqrt().ceil() as usize;
        let mtry = mtry.max(1).min(k);

        let mut forest: Vec<CausalNode> = Vec::with_capacity(trees);
        let mut feature_importance = Array1::zeros(k);
        let mut te_preds: Vec<Vec<f64>> = vec![Vec::new(); n]; // per-obs predictions

        for _ in 0..trees {
            // Bootstrap sample
            let mut boot_indices = Vec::with_capacity(n);
            for _ in 0..n {
                boot_indices.push(Self::rand_int(n));
            }

            // Honest splitting: split boot sample into two halves
            let mut split_idx = boot_indices.clone();
            for i in 0..split_idx.len() {
                let j = i + Self::rand_int(split_idx.len() - i);
                split_idx.swap(i, j);
            }
            let mid = split_idx.len() / 2;
            let split_half = &split_idx[..mid];
            let est_half = &split_idx[mid..];

            // Build tree on split half
            let tree = Self::build_tree(
                y,
                t,
                x,
                split_half,
                est_half,
                depth,
                mtry,
                k,
                0,
                &mut feature_importance,
            );

            // Predict for all observations
            for (i, te_pred) in te_preds.iter_mut().enumerate().take(n) {
                let te = Self::predict_te(&tree, &x.row(i).to_owned());
                te_pred.push(te);
            }

            forest.push(tree);
        }

        // Average predictions across trees
        let mut treatment_effects = Array1::zeros(n);
        for i in 0..n {
            if te_preds[i].is_empty() {
                treatment_effects[i] = 0.0;
            } else {
                treatment_effects[i] = te_preds[i].iter().sum::<f64>() / te_preds[i].len() as f64;
            }
        }

        // ATE = mean of treatment effects
        let ate = treatment_effects.mean().unwrap_or(0.0);

        // SE of ATE: SD of individual TE predictions / sqrt(n)
        let te_var = treatment_effects.mapv(|v| (v - ate).powi(2)).sum() / n as f64;
        let ate_se = (te_var / n as f64).sqrt();

        // Heterogeneity: SD of treatment effects
        let heterogeneity = te_var.sqrt();

        // CI
        let z = 1.959964;
        let ate_ci = [ate - z * ate_se, ate + z * ate_se];

        Ok(CausalForestResult {
            treatment_effects,
            ate,
            ate_se,
            ate_ci,
            feature_importance,
            heterogeneity,
            n_trees: trees,
            max_depth: depth,
            n_obs: n,
            n_features: k,
            variable_names: names,
        })
    }

    /// Build a honest causal tree.
    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        y: &Array1<f64>,
        t: &[bool],
        x: &Array2<f64>,
        split_idx: &[usize],
        est_idx: &[usize],
        max_depth: usize,
        mtry: usize,
        k: usize,
        depth: usize,
        importance: &mut Array1<f64>,
    ) -> CausalNode {
        // Compute leaf effect from estimation half
        let (effect, var, n_t, n_c) = Self::compute_leaf_effect(y, t, est_idx);

        if split_idx.len() < 10 || est_idx.len() < 6 || depth >= max_depth {
            return CausalNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_effect: effect,
                leaf_n_treated: n_t,
                leaf_n_control: n_c,
                leaf_var: var,
                is_leaf: true,
            };
        }

        let (best_feature, best_threshold, best_gain) =
            Self::find_best_causal_split(y, t, x, split_idx, mtry, k);

        if best_gain < 1e-10 || best_feature >= k {
            return CausalNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_effect: effect,
                leaf_n_treated: n_t,
                leaf_n_control: n_c,
                leaf_var: var,
                is_leaf: true,
            };
        }

        importance[best_feature] += best_gain;

        // Split both halves
        let mut split_left = Vec::new();
        let mut split_right = Vec::new();
        let mut est_left = Vec::new();
        let mut est_right = Vec::new();

        for &i in split_idx {
            if x[(i, best_feature)] <= best_threshold {
                split_left.push(i);
            } else {
                split_right.push(i);
            }
        }
        for &i in est_idx {
            if x[(i, best_feature)] <= best_threshold {
                est_left.push(i);
            } else {
                est_right.push(i);
            }
        }

        if split_left.is_empty()
            || split_right.is_empty()
            || est_left.is_empty()
            || est_right.is_empty()
        {
            return CausalNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_effect: effect,
                leaf_n_treated: n_t,
                leaf_n_control: n_c,
                leaf_var: var,
                is_leaf: true,
            };
        }

        CausalNode {
            feature: best_feature,
            threshold: best_threshold,
            left: Some(Box::new(Self::build_tree(
                y,
                t,
                x,
                &split_left,
                &est_left,
                max_depth,
                mtry,
                k,
                depth + 1,
                importance,
            ))),
            right: Some(Box::new(Self::build_tree(
                y,
                t,
                x,
                &split_right,
                &est_right,
                max_depth,
                mtry,
                k,
                depth + 1,
                importance,
            ))),
            leaf_effect: effect,
            leaf_n_treated: n_t,
            leaf_n_control: n_c,
            leaf_var: var,
            is_leaf: false,
        }
    }

    /// Compute leaf treatment effect: tau = mean(Y|T=1) - mean(Y|T=0).
    fn compute_leaf_effect(
        y: &Array1<f64>,
        t: &[bool],
        indices: &[usize],
    ) -> (f64, f64, usize, usize) {
        let mut y_t_sum = 0.0_f64;
        let mut y_c_sum = 0.0_f64;
        let mut n_t = 0_usize;
        let mut n_c = 0_usize;

        for &i in indices {
            if t[i] {
                y_t_sum += y[i];
                n_t += 1;
            } else {
                y_c_sum += y[i];
                n_c += 1;
            }
        }

        if n_t == 0 || n_c == 0 {
            return (0.0, 0.0, n_t, n_c);
        }

        let y_t_mean = y_t_sum / n_t as f64;
        let y_c_mean = y_c_sum / n_c as f64;
        let effect = y_t_mean - y_c_mean;

        // Variance: Var(tau) = Var(Y|T=1)/n_t + Var(Y|T=0)/n_c
        let mut y_t_var = 0.0_f64;
        let mut y_c_var = 0.0_f64;
        for &i in indices {
            if t[i] {
                y_t_var += (y[i] - y_t_mean).powi(2);
            } else {
                y_c_var += (y[i] - y_c_mean).powi(2);
            }
        }
        y_t_var /= n_t as f64;
        y_c_var /= n_c as f64;
        let var = y_t_var / n_t as f64 + y_c_var / n_c as f64;

        (effect, var, n_t, n_c)
    }

    /// Find best split maximizing treatment effect heterogeneity.
    fn find_best_causal_split(
        y: &Array1<f64>,
        t: &[bool],
        x: &Array2<f64>,
        indices: &[usize],
        mtry: usize,
        k: usize,
    ) -> (usize, f64, f64) {
        let (parent_effect, parent_var, _, _) = Self::compute_leaf_effect(y, t, indices);

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
            if values.len() < 4 {
                continue;
            }

            let n_thresh = 3.min(values.len() - 1);
            for thresh_i in 0..n_thresh {
                let idx = (thresh_i + 1) * values.len() / (n_thresh + 1);
                let threshold = values[idx];

                let mut left_idx = Vec::new();
                let mut right_idx = Vec::new();
                for &i in indices {
                    if x[(i, feat)] <= threshold {
                        left_idx.push(i);
                    } else {
                        right_idx.push(i);
                    }
                }

                if left_idx.len() < 5 || right_idx.len() < 5 {
                    continue;
                }

                let (left_effect, left_var, lt, lc) = Self::compute_leaf_effect(y, t, &left_idx);
                let (right_effect, right_var, rt, rc) = Self::compute_leaf_effect(y, t, &right_idx);

                if lt == 0 || lc == 0 || rt == 0 || rc == 0 {
                    continue;
                }

                // Gain: variance of effects across leaves (maximize heterogeneity)
                // Also penalize by leaf variance (uncertainty)
                let gain = (left_effect - right_effect).powi(2)
                    - 0.5 * (left_var + right_var)
                    - parent_var;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        let _ = parent_effect;
        (best_feature, best_threshold, best_gain)
    }

    fn predict_te(tree: &CausalNode, x: &Array1<f64>) -> f64 {
        if tree.is_leaf {
            return tree.leaf_effect;
        }
        if x[tree.feature] <= tree.threshold {
            if let Some(ref left) = tree.left {
                return Self::predict_te(left, x);
            }
        } else if let Some(ref right) = tree.right {
            return Self::predict_te(right, x);
        }
        tree.leaf_effect
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
            static STATE: Cell<u64> = const { Cell::new(1618033988) };
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
