//! Generalized Random Forest (Athey, Tibshirani & Wager 2019).
//!
//! Generalizes causal forests to estimate a wide class of
//! treatment effect parameters beyond ATE. Key extensions:
//!
//!   - Local centering: subtract nuisance estimates before
//!     tree fitting (double robustness)
//!   - Variable importance weighted by treatment heterogeneity
//!   - Treatment effect at the individual level (CATE)
//!   - Confidence intervals via the variance of the forest
//!     estimator (Mentch & Hooker 2016; Wager & Athey 2018)
//!
//! This implementation focuses on CATE estimation with:
//!   1. Nuisance functions m(X) = E[Y|X] and e(X) = E[T|X]
//!      estimated via OLS on the full sample
//!   2. Pseudo-outcome: rho_i = (Y_i - m(X_i)) / (T_i - e(X_i)) + tau_0
//!      where tau_0 is the AIPW estimator
//!   3. Forest regression on pseudo-outcome to estimate CATE
//!   4. Variance-based confidence intervals

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of Generalized Random Forest estimation.
#[derive(Debug)]
pub struct GrfResult {
    /// Predicted CATE for each observation (n)
    pub cate: Array1<f64>,
    /// AIPW average treatment effect (doubly robust)
    pub ate: f64,
    /// Standard error of ATE
    pub ate_se: f64,
    /// 95% CI for ATE
    pub ate_ci: [f64; 2],
    /// Propensity score estimates e(X) (n)
    pub propensity: Array1<f64>,
    /// Outcome regression m(X) estimates (n)
    pub outcome_reg: Array1<f64>,
    /// Feature importance
    pub feature_importance: Array1<f64>,
    /// Heterogeneity: SD of CATE
    pub heterogeneity: f64,
    /// Number of trees
    pub n_trees: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for GrfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Generalized Random Forest ")?;
        writeln!(f, "Athey, Tibshirani & Wager (2019)")?;
        writeln!(f, "Doubly-robust CATE estimation with local centering")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Trees:", self.n_trees)?;
        writeln!(f, "{:<20} {:>12.6}", "ATE (AIPW):", self.ate)?;
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

        // CATE distribution
        writeln!(f, "\n  CATE distribution:")?;
        let mut sorted = self.cate.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
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

pub struct GRF;

impl GRF {
    /// Estimate Generalized Random Forest for CATE.
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
    ) -> Result<GrfResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if t.len() != n || x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "GRF: dimension mismatch".into(),
            ));
        }
        if n < 20 {
            return Err(GreenersError::InvalidOperation(
                "GRF: need at least 20 observations".into(),
            ));
        }

        let n_treated = t.iter().filter(|&&t| t).count();
        let n_control = n - n_treated;
        if n_treated < 5 || n_control < 5 {
            return Err(GreenersError::InvalidOperation(
                "GRF: need at least 5 treated and 5 control".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let trees = n_trees.unwrap_or(100);
        let depth = max_depth.unwrap_or(5);

        // Step 1: Estimate nuisance functions via OLS
        // m(X) = E[Y|X] — outcome regression
        let m_beta = Self::ols_fit(y, x, n, k)?;
        let m_hat: Array1<f64> = (0..n)
            .map(|i| {
                let mut pred = m_beta[0];
                for j in 0..k {
                    pred += m_beta[j + 1] * x[(i, j)];
                }
                pred
            })
            .collect();

        // e(X) = E[T|X] — propensity (linear probability model)
        let t_vec: Array1<f64> = t.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect();
        let e_beta = Self::ols_fit(&t_vec, x, n, k)?;
        let e_hat: Array1<f64> = (0..n)
            .map(|i| {
                let mut pred = e_beta[0];
                for j in 0..k {
                    pred += e_beta[j + 1] * x[(i, j)];
                }
                pred.clamp(0.01, 0.99) // bound propensity
            })
            .collect();

        // Step 2: AIPW estimator (doubly robust)
        let mut aipw_terms = Vec::with_capacity(n);
        for i in 0..n {
            let ti = if t[i] { 1.0 } else { 0.0 };
            let aipw_i = (m_hat[i] + ti * (y[i] - m_hat[i]) / e_hat[i])
                - (m_hat[i] + (1.0 - ti) * (y[i] - m_hat[i]) / (1.0 - e_hat[i]));
            aipw_terms.push(aipw_i);
        }
        let ate_aipw: f64 = aipw_terms.iter().sum::<f64>() / n as f64;

        // Step 3: Pseudo-outcome for CATE
        // rho_i = (Y_i - m(X_i)) / (T_i - e(X_i)) + tau_0
        let pseudo_y: Array1<f64> = (0..n)
            .map(|i| {
                let ti = if t[i] { 1.0 } else { 0.0 };
                let denom = ti - e_hat[i];
                if denom.abs() < 1e-10 {
                    ate_aipw
                } else {
                    (y[i] - m_hat[i]) / denom + ate_aipw
                }
            })
            .collect();

        // Step 4: Forest regression on pseudo-outcome
        let mtry = (k as f64).sqrt().ceil() as usize;
        let mtry = mtry.max(1).min(k);
        let mut feature_importance = Array1::zeros(k);
        let mut cate_preds: Vec<Vec<f64>> = vec![Vec::new(); n];

        for _ in 0..trees {
            // Bootstrap
            let boot_idx: Vec<usize> = (0..n).map(|_| Self::rand_int(n)).collect();

            // Build regression tree on pseudo-outcome
            let tree = Self::build_regression_tree(
                &pseudo_y,
                x,
                &boot_idx,
                depth,
                mtry,
                k,
                0,
                &mut feature_importance,
            );

            // Predict CATE for all observations
            for (i, cate_pred) in cate_preds.iter_mut().enumerate().take(n) {
                let pred = Self::predict_tree(&tree, &x.row(i).to_owned());
                cate_pred.push(pred);
            }
        }

        // Average CATE predictions
        let mut cate = Array1::zeros(n);
        for i in 0..n {
            if cate_preds[i].is_empty() {
                cate[i] = ate_aipw;
            } else {
                cate[i] = cate_preds[i].iter().sum::<f64>() / cate_preds[i].len() as f64;
            }
        }

        // SE of ATE: variance of AIPW terms / n
        let aipw_var: f64 = aipw_terms
            .iter()
            .map(|v| (v - ate_aipw).powi(2))
            .sum::<f64>()
            / n as f64;
        let ate_se = (aipw_var / n as f64).sqrt();

        let z = 1.959964;
        let ate_ci = [ate_aipw - z * ate_se, ate_aipw + z * ate_se];

        // Heterogeneity
        let cate_mean = cate.mean().unwrap_or(0.0);
        let heterogeneity = (cate.mapv(|v| (v - cate_mean).powi(2)).sum() / n as f64).sqrt();

        let _ = e_hat;
        Ok(GrfResult {
            cate,
            ate: ate_aipw,
            ate_se,
            ate_ci,
            propensity: e_hat,
            outcome_reg: m_hat,
            feature_importance,
            heterogeneity,
            n_trees: trees,
            n_obs: n,
            n_features: k,
            variable_names: names,
        })
    }

    /// OLS with intercept.
    fn ols_fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        n: usize,
        k: usize,
    ) -> Result<Array1<f64>, GreenersError> {
        let mut x_full = Array2::zeros((n, k + 1));
        for i in 0..n {
            x_full[(i, 0)] = 1.0;
            for j in 0..k {
                x_full[(i, j + 1)] = x[(i, j)];
            }
        }
        let xt = x_full.t();
        let xtx = xt.dot(&x_full);
        let xtx_inv = (&xtx + Array2::<f64>::eye(k + 1) * 1e-8).inv()?;
        let xty = xt.dot(y);
        Ok(xtx_inv.dot(&xty))
    }

    /// Build regression tree on pseudo-outcome.
    #[allow(clippy::too_many_arguments)]
    fn build_regression_tree(
        y: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        max_depth: usize,
        mtry: usize,
        k: usize,
        depth: usize,
        importance: &mut Array1<f64>,
    ) -> RegNode {
        let n = indices.len();
        let leaf_val: f64 = if n > 0 {
            indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64
        } else {
            0.0
        };

        if n < 5 || depth >= max_depth {
            return RegNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_val,
                is_leaf: true,
            };
        }

        let (best_feature, best_threshold, best_gain) =
            Self::find_best_split_reg(y, x, indices, mtry, k);

        if best_gain < 1e-10 || best_feature >= k {
            return RegNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_val,
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
            return RegNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                leaf_val,
                is_leaf: true,
            };
        }

        RegNode {
            feature: best_feature,
            threshold: best_threshold,
            left: Some(Box::new(Self::build_regression_tree(
                y,
                x,
                &left_idx,
                max_depth,
                mtry,
                k,
                depth + 1,
                importance,
            ))),
            right: Some(Box::new(Self::build_regression_tree(
                y,
                x,
                &right_idx,
                max_depth,
                mtry,
                k,
                depth + 1,
                importance,
            ))),
            leaf_val,
            is_leaf: false,
        }
    }

    fn find_best_split_reg(
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
                        left_sum += y[i];
                        left_n += 1;
                    } else {
                        right_sum += y[i];
                        right_n += 1;
                    }
                }

                if left_n < 3 || right_n < 3 {
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

    fn predict_tree(tree: &RegNode, x: &Array1<f64>) -> f64 {
        if tree.is_leaf {
            return tree.leaf_val;
        }
        if x[tree.feature] <= tree.threshold {
            if let Some(ref left) = tree.left {
                return Self::predict_tree(left, x);
            }
        } else if let Some(ref right) = tree.right {
            return Self::predict_tree(right, x);
        }
        tree.leaf_val
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
            static STATE: Cell<u64> = const { Cell::new(1414213562) };
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

/// Regression tree node (internal use).
#[derive(Debug, Clone)]
struct RegNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<RegNode>>,
    right: Option<Box<RegNode>>,
    leaf_val: f64,
    is_leaf: bool,
}

// Use Normal for CI (already imported)
#[allow(dead_code)]
fn _ensure_normal_used() {
    let _ = Normal::new(0.0, 1.0).unwrap().cdf(0.5);
}
