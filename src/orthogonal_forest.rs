//! Orthogonal Random Forest (ORF) for CATE estimation
//! (Oprescu, Syrgkanis & Wu 2019).
//!
//! A doubly-robust forest estimator that combines:
//!   1. Orthogonalization of treatment and outcome with respect
//!      to confounders W (residualization)
//!   2. Random forest on the orthogonalized signals
//!   3. Local moment estimation for CATE
//!
//! Procedure:
//!   - Split sample into two halves: I (estimation) and J (nuisance)
//!   - On J: estimate T(W) = E[T|W] and Y(W) = E[Y|W] via OLS
//!   - On I: compute residualized T_tilde = T - T_hat(W) and
//!     Y_tilde = Y - Y_hat(W)
//!   - Build a random forest on I using T_tilde as treatment and
//!     features X for splitting
//!   - For each leaf, CATE = sum(T_tilde * Y_tilde) / sum(T_tilde^2)
//!
//! This is a simplified implementation suitable for small datasets.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// ORF tree node.
#[derive(Debug, Clone)]
struct OrfNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<OrfNode>>,
    right: Option<Box<OrfNode>>,
    /// CATE estimate in leaf
    leaf_cate: f64,
    /// Number of obs in leaf
    _leaf_n: usize,
    is_leaf: bool,
}

/// Result of ORF estimation.
#[derive(Debug)]
pub struct OrfResult {
    /// Predicted CATE for each observation (n)
    pub cate: Array1<f64>,
    /// ATE (mean CATE)
    pub ate: f64,
    /// Standard error of ATE
    pub ate_se: f64,
    /// 95% CI for ATE
    pub ate_ci: [f64; 2],
    /// Number of trees
    pub n_trees: usize,
    /// Max depth
    pub max_depth: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Variable names for features X
    pub feature_names: Vec<String>,
    /// Feature importance (split counts normalized)
    pub feature_importance: Array1<f64>,
}

impl fmt::Display for OrfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Orthogonal Random Forest ")?;
        writeln!(f, "Oprescu, Syrgkanis & Wu (2019)")?;
        writeln!(f, "Doubly-robust CATE via orthogonalization + forest")?;
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

        // Feature importance
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Feature importance:")?;
        let mut imp_vec: Vec<(String, f64)> = self
            .feature_names
            .iter()
            .zip(self.feature_importance.iter())
            .map(|(name, &imp)| (name.clone(), imp))
            .collect();
        imp_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        writeln!(f, "  {:<14} {:>12}", "Feature", "Importance")?;
        writeln!(f, "{:-^78}", "")?;
        for (name, imp) in imp_vec {
            writeln!(f, "  {:<14} {:>12.4}", name, imp)?;
        }

        // CATE distribution
        let mut sorted = self.cate.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        writeln!(f, "\n  CATE distribution:")?;
        writeln!(
            f,
            "  Min: {:>10.4}  Q1: {:>10.4}  Median: {:>10.4}  Q3: {:>10.4}  Max: {:>10.4}",
            sorted[0],
            sorted[n / 4],
            sorted[n / 2],
            sorted[3 * n / 4],
            sorted[n - 1]
        )?;

        write!(f, "{:=^78}", "")
    }
}

pub struct OrthogonalForest;

impl OrthogonalForest {
    /// Estimate ORF for CATE.
    ///
    /// # Arguments
    /// * `y` - Outcome (n)
    /// * `t` - Treatment (n), true if treated
    /// * `x` - Features for CATE heterogeneity (n x k)
    /// * `w` - Confounders for orthogonalization (n x p)
    /// * `n_trees` - Number of trees (default 50)
    /// * `max_depth` - Max tree depth (default 5)
    /// * `feature_names` - Optional names for X features
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        y: &Array1<f64>,
        t: &[bool],
        x: &Array2<f64>,
        w: &Array2<f64>,
        n_trees: Option<usize>,
        max_depth: Option<usize>,
        feature_names: Option<Vec<String>>,
    ) -> Result<OrfResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        let p = w.ncols();
        if t.len() != n || x.nrows() != n || w.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "ORF: dimension mismatch".into(),
            ));
        }
        if n < 30 {
            return Err(GreenersError::InvalidOperation(
                "ORF: need at least 30 observations".into(),
            ));
        }

        let n_treated = t.iter().filter(|&&t| t).count();
        let n_control = n - n_treated;
        if n_treated < 5 || n_control < 5 {
            return Err(GreenersError::InvalidOperation(
                "ORF: need at least 5 treated and 5 control".into(),
            ));
        }

        let names = feature_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let m = n_trees.unwrap_or(50);
        let depth = max_depth.unwrap_or(5);

        let t_vec: Array1<f64> = t.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect();

        // Split sample: half for nuisance, half for forest
        let split = n / 2;
        let nuisance_idx: Vec<usize> = (0..split).collect();
        let forest_idx: Vec<usize> = (split..n).collect();

        // Estimate nuisance on first half
        // T(W) = E[T|W] via OLS
        let t_hat_beta = Self::ols_subset(&t_vec, w, &nuisance_idx, p)?;
        // Y(W) = E[Y|W] via OLS
        let y_hat_beta = Self::ols_subset(y, w, &nuisance_idx, p)?;

        // Residualize on forest half
        let n_forest = forest_idx.len();
        let mut t_tilde = Array1::zeros(n_forest);
        let mut y_tilde = Array1::zeros(n_forest);
        let mut x_forest = Array2::zeros((n_forest, k));
        for (i, &idx) in forest_idx.iter().enumerate() {
            let t_pred = Self::predict_ols(&t_hat_beta, &w.row(idx).to_owned(), p);
            let y_pred = Self::predict_ols(&y_hat_beta, &w.row(idx).to_owned(), p);
            t_tilde[i] = t_vec[idx] - t_pred;
            y_tilde[i] = y[idx] - y_pred;
            for j in 0..k {
                x_forest[(i, j)] = x[(idx, j)];
            }
        }

        // Build forest
        let mut trees: Vec<OrfNode> = Vec::with_capacity(m);
        let mut split_counts = vec![0_usize; k];

        for _ in 0..m {
            // Bootstrap sample
            let boot_idx: Vec<usize> = (0..n_forest).map(|_| Self::rand_int(n_forest)).collect();
            let tree = Self::build_tree(
                &y_tilde,
                &t_tilde,
                &x_forest,
                &boot_idx,
                depth,
                k,
                &mut split_counts,
            );
            trees.push(tree);
        }

        // Predict CATE for all observations
        // For obs in nuisance half, we need to predict using X
        let mut cate = Array1::zeros(n);
        for i in 0..n {
            let x_i = x.row(i).to_owned();
            let mut cate_sum = 0.0;
            for tree in &trees {
                cate_sum += Self::predict_tree(tree, &x_i);
            }
            cate[i] = cate_sum / m as f64;
        }

        // ATE
        let ate = cate.mean().unwrap_or(0.0);

        // SE: variance of CATE / n
        let cate_var = cate.mapv(|v| (v - ate).powi(2)).sum() / n as f64;
        let ate_se = (cate_var / n as f64).sqrt();

        let z = 1.959964;
        let ate_ci = [ate - z * ate_se, ate + z * ate_se];

        // Feature importance
        let total_splits: usize = split_counts.iter().sum();
        let feature_importance = Array1::from_vec(
            split_counts
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

        Ok(OrfResult {
            cate,
            ate,
            ate_se,
            ate_ci,
            n_trees: m,
            max_depth: depth,
            n_obs: n,
            n_features: k,
            feature_names: names,
            feature_importance,
        })
    }

    fn ols_subset(
        y: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        k: usize,
    ) -> Result<Array1<f64>, GreenersError> {
        let n = indices.len();
        let mut x_full = Array2::zeros((n, k + 1));
        let mut y_sub = Array1::zeros(n);
        for (i, &idx) in indices.iter().enumerate() {
            x_full[(i, 0)] = 1.0;
            for j in 0..k {
                x_full[(i, j + 1)] = x[(idx, j)];
            }
            y_sub[i] = y[idx];
        }
        let xt = x_full.t();
        let xtx = xt.dot(&x_full);
        let xtx_inv = (&xtx + Array2::<f64>::eye(k + 1) * 1e-8).inv()?;
        let xty = xt.dot(&y_sub);
        Ok(xtx_inv.dot(&xty))
    }

    fn predict_ols(beta: &Array1<f64>, x: &Array1<f64>, k: usize) -> f64 {
        let mut pred = beta[0];
        for j in 0..k {
            pred += beta[j + 1] * x[j];
        }
        pred
    }

    fn build_tree(
        y_tilde: &Array1<f64>,
        t_tilde: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        max_depth: usize,
        k: usize,
        split_counts: &mut [usize],
    ) -> OrfNode {
        Self::build_node(y_tilde, t_tilde, x, indices, max_depth, k, 0, split_counts)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_node(
        y_tilde: &Array1<f64>,
        t_tilde: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        max_depth: usize,
        k: usize,
        depth: usize,
        split_counts: &mut [usize],
    ) -> OrfNode {
        let n = indices.len();
        // CATE = sum(T_tilde * Y_tilde) / sum(T_tilde^2)
        let sum_ty: f64 = indices.iter().map(|&i| t_tilde[i] * y_tilde[i]).sum();
        let sum_t2: f64 = indices.iter().map(|&i| t_tilde[i] * t_tilde[i]).sum();
        let cate = if sum_t2.abs() > 1e-10 {
            sum_ty / sum_t2
        } else {
            0.0
        };

        if n < 5 || depth >= max_depth {
            return OrfNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_cate: cate,
                _leaf_n: n,
                is_leaf: true,
            };
        }

        // Find best split: maximize heterogeneity in CATE
        let (best_feature, best_threshold, best_gain) =
            Self::find_split(y_tilde, t_tilde, x, indices, k);

        if best_gain < 0.001 || best_feature >= k {
            return OrfNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_cate: cate,
                _leaf_n: n,
                is_leaf: true,
            };
        }

        split_counts[best_feature] += 1;

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
            return OrfNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_cate: cate,
                _leaf_n: n,
                is_leaf: true,
            };
        }

        OrfNode {
            feature: best_feature,
            threshold: best_threshold,
            left: Some(Box::new(Self::build_node(
                y_tilde,
                t_tilde,
                x,
                &left_idx,
                max_depth,
                k,
                depth + 1,
                split_counts,
            ))),
            right: Some(Box::new(Self::build_node(
                y_tilde,
                t_tilde,
                x,
                &right_idx,
                max_depth,
                k,
                depth + 1,
                split_counts,
            ))),
            leaf_cate: cate,
            _leaf_n: n,
            is_leaf: false,
        }
    }

    fn find_split(
        y_tilde: &Array1<f64>,
        t_tilde: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        k: usize,
    ) -> (usize, f64, f64) {
        let n = indices.len();
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

                let mut left_idx = Vec::new();
                let mut right_idx = Vec::new();
                for &i in indices {
                    if x[(i, feat)] <= threshold {
                        left_idx.push(i);
                    } else {
                        right_idx.push(i);
                    }
                }

                if left_idx.len() < 3 || right_idx.len() < 3 {
                    continue;
                }

                // CATE for each side
                let left_ty: f64 = left_idx.iter().map(|&i| t_tilde[i] * y_tilde[i]).sum();
                let left_t2: f64 = left_idx.iter().map(|&i| t_tilde[i] * t_tilde[i]).sum();
                let left_cate = if left_t2.abs() > 1e-10 {
                    left_ty / left_t2
                } else {
                    0.0
                };

                let right_ty: f64 = right_idx.iter().map(|&i| t_tilde[i] * y_tilde[i]).sum();
                let right_t2: f64 = right_idx.iter().map(|&i| t_tilde[i] * t_tilde[i]).sum();
                let right_cate = if right_t2.abs() > 1e-10 {
                    right_ty / right_t2
                } else {
                    0.0
                };

                // Gain = weighted variance of CATE
                let nl = left_idx.len() as f64;
                let nr = right_idx.len() as f64;
                let parent_cate = (left_cate * nl + right_cate * nr) / n as f64;
                let gain = nl * (left_cate - parent_cate).powi(2)
                    + nr * (right_cate - parent_cate).powi(2);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn predict_tree(tree: &OrfNode, x: &Array1<f64>) -> f64 {
        if tree.is_leaf {
            return tree.leaf_cate;
        }
        if x[tree.feature] <= tree.threshold {
            if let Some(ref left) = tree.left {
                return Self::predict_tree(left, x);
            }
        } else if let Some(ref right) = tree.right {
            return Self::predict_tree(right, x);
        }
        tree.leaf_cate
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
            static STATE: Cell<u64> = const { Cell::new(9876543210) };
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
