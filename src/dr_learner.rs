//! Doubly-Robust Learner (DR-learner, Kennedy 2023).
//!
//! A meta-learning approach for CATE estimation that combines
//! doubly-robust pseudo-outcomes with regression. Unlike GRF
//! (which uses forests), DR-learner uses any base learner for
//! the final CATE regression.
//!
//! Procedure:
//! 1. Split sample into 3 folds: A (nuisance), B (CATE), C (evaluation)
//! 2. On fold A: estimate m(X) = E[Y|X] and e(X) = E[T|X]
//! 3. On fold B: compute DR pseudo-outcome:
//!    psi_i = (m_A(X_i) + T_i*(Y_i - m_A(X_i))/e_A(X_i))
//!    minus (m_A(X_i) + (1-T_i)*(Y_i - m_A(X_i))/(1-e_A(X_i)))
//!    Then regress psi_i on X_i to get CATE model
//! 4. On fold C: evaluate CATE model, compute ATE
//! 5. Rotate folds and average
//!
//! Doubly robust: consistent if either m(X) or e(X) is correct.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::Normal;
use std::fmt;

/// Result of DR-learner estimation.
#[derive(Debug)]
pub struct DrLearnerResult {
    /// Predicted CATE for each observation (n)
    pub cate: Array1<f64>,
    /// ATE (averaged DR pseudo-outcomes)
    pub ate: f64,
    /// Standard error of ATE
    pub ate_se: f64,
    /// 95% CI for ATE
    pub ate_ci: [f64; 2],
    /// Propensity score e(X) (n)
    pub propensity: Array1<f64>,
    /// Outcome regression m(X) (n)
    pub outcome_reg: Array1<f64>,
    /// CATE regression coefficients
    pub cate_coefficients: Array1<f64>,
    /// Number of folds
    pub n_folds: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for DrLearnerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " DR-Learner ")?;
        writeln!(f, "Kennedy (2023)")?;
        writeln!(f, "Doubly-robust CATE via pseudo-outcome regression")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Folds:", self.n_folds)?;
        writeln!(f, "{:<20} {:>12.6}", "ATE:", self.ate)?;
        writeln!(f, "{:<20} {:>12.6}", "ATE SE:", self.ate_se)?;
        writeln!(
            f,
            "{:<20} [{:.4}, {:.4}]",
            "95% CI:", self.ate_ci[0], self.ate_ci[1]
        )?;

        // CATE coefficients
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  CATE regression coefficients:")?;
        writeln!(f, "  {:<14} {:>12}", "Variable", "Coef")?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "  {:<14} {:>12.6}",
            "Intercept", self.cate_coefficients[0]
        )?;
        for (j, name) in self.variable_names.iter().enumerate() {
            if j + 1 < self.cate_coefficients.len() {
                writeln!(f, "  {:<14} {:>12.6}", name, self.cate_coefficients[j + 1])?;
            }
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

pub struct DRLearner;

impl DRLearner {
    /// Estimate DR-learner for CATE.
    ///
    /// # Arguments
    /// * `y` - Outcome (n)
    /// * `t` - Treatment indicator (n), true if treated
    /// * `x` - Features (n x k)
    /// * `n_folds` - Number of cross-fitting folds (default 3)
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        t: &[bool],
        x: &Array2<f64>,
        n_folds: Option<usize>,
        variable_names: Option<Vec<String>>,
    ) -> Result<DrLearnerResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if t.len() != n || x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "DRLearner: dimension mismatch".into(),
            ));
        }
        if n < 30 {
            return Err(GreenersError::InvalidOperation(
                "DRLearner: need at least 30 observations".into(),
            ));
        }

        let n_treated = t.iter().filter(|&&t| t).count();
        let n_control = n - n_treated;
        if n_treated < 5 || n_control < 5 {
            return Err(GreenersError::InvalidOperation(
                "DRLearner: need at least 5 treated and 5 control".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let folds = n_folds.unwrap_or(3).min(n / 10).max(2);

        // Create fold assignments (shuffle then split)
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..n {
            let j = i + Self::rand_int(n - i);
            indices.swap(i, j);
        }
        let fold_size = n / folds;
        let fold_of: Vec<usize> = (0..n)
            .map(|i| (i / fold_size.max(1)).min(folds - 1))
            .collect();
        let mut fold_assignment = vec![0_usize; n];
        for (pos, &orig_idx) in indices.iter().enumerate() {
            fold_assignment[orig_idx] = fold_of[pos];
        }

        // Cross-fitting: for each fold, use other folds for nuisance,
        // this fold for pseudo-outcome and CATE regression
        let mut pseudo_outcomes = Array1::zeros(n);
        let mut m_hat_all = Array1::zeros(n);
        let mut e_hat_all = Array1::zeros(n);

        for fold in 0..folds {
            // Nuisance fold = all other folds
            let nuisance_idx: Vec<usize> = (0..n).filter(|&i| fold_assignment[i] != fold).collect();
            let cate_idx: Vec<usize> = (0..n).filter(|&i| fold_assignment[i] == fold).collect();

            if nuisance_idx.is_empty() || cate_idx.is_empty() {
                continue;
            }

            // Estimate m(X) and e(X) on nuisance fold
            let m_beta = Self::ols_subset(y, x, &nuisance_idx, k)?;
            let t_vec: Array1<f64> = t.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect();
            let e_beta = Self::ols_subset(&t_vec, x, &nuisance_idx, k)?;

            // Predict on CATE fold
            for &i in &cate_idx {
                let m_pred = Self::predict_ols(&m_beta, &x.row(i).to_owned(), k);
                let e_pred = Self::predict_ols(&e_beta, &x.row(i).to_owned(), k).clamp(0.01, 0.99);
                m_hat_all[i] = m_pred;
                e_hat_all[i] = e_pred;

                // DR pseudo-outcome
                let ti = if t[i] { 1.0 } else { 0.0 };
                let psi = (m_pred + ti * (y[i] - m_pred) / e_pred)
                    - (m_pred + (1.0 - ti) * (y[i] - m_pred) / (1.0 - e_pred));
                pseudo_outcomes[i] = psi;
            }
        }

        // Regress pseudo-outcomes on X for CATE model
        let cate_beta = Self::ols_full(&pseudo_outcomes, x, n, k)?;

        // Predict CATE for all observations
        let mut cate = Array1::zeros(n);
        for i in 0..n {
            cate[i] = Self::predict_ols(&cate_beta, &x.row(i).to_owned(), k);
        }

        // ATE = mean of pseudo-outcomes
        let ate = pseudo_outcomes.mean().unwrap_or(0.0);

        // SE: variance of pseudo-outcomes / n
        let po_var = pseudo_outcomes.mapv(|v| (v - ate).powi(2)).sum() / n as f64;
        let ate_se = (po_var / n as f64).sqrt();

        let z = 1.959964;
        let ate_ci = [ate - z * ate_se, ate + z * ate_se];

        let _ =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        Ok(DrLearnerResult {
            cate,
            ate,
            ate_se,
            ate_ci,
            propensity: e_hat_all,
            outcome_reg: m_hat_all,
            cate_coefficients: cate_beta,
            n_folds: folds,
            n_obs: n,
            n_features: k,
            variable_names: names,
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

    fn ols_full(
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

    fn predict_ols(beta: &Array1<f64>, x: &Array1<f64>, k: usize) -> f64 {
        let mut pred = beta[0];
        for j in 0..k {
            pred += beta[j + 1] * x[j];
        }
        pred
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
            static STATE: Cell<u64> = const { Cell::new(2236067977) };
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
