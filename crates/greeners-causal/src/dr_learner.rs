//! Doubly robust conditional average treatment effect estimation (Kennedy, 2023).
//!
//! Cross-fitted linear models estimate the treatment propensity and the two
//! treatment-specific outcome means. The augmented inverse-probability score is
//! regressed on covariates for CATE, and averaged for ATE. Identification requires
//! independent observations, consistency, conditional exchangeability and overlap.
//!
//! Propensities are clipped to [0.01, 0.99]. Clipping is a numerical safeguard,
//! not an overlap diagnostic; it can invalidate propensity-model consistency.
//! Point estimates remain doubly robust when either the outcome models or the
//! bounded propensity model is correct. The reported influence-score SE and Wald
//! interval require BOTH nuisance functions to be consistently estimated at
//! adequate product rates; double robustness of the point estimate does not
//! qualify those intervals under misspecification.
//!
//! Reference: [Kennedy (2023), Algorithm 1](https://arxiv.org/abs/2004.14497).

use faer::{linalg::solvers::Svd, Mat};
use greeners_core::GreenersError;
use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::fmt;

/// Result of DR-learner estimation.
#[derive(Debug)]
pub struct DrLearnerResult {
    /// Predicted CATE for each observation (n)
    pub cate: Array1<f64>,
    /// ATE (averaged DR pseudo-outcomes)
    pub ate: f64,
    /// Conventional influence-score SE; requires consistent outcome and propensity models.
    pub ate_se: f64,
    /// 95% CI for ATE
    pub ate_ci: [f64; 2],
    /// Propensity score e(X) (n)
    pub propensity: Array1<f64>,
    /// Cross-fitted factual conditional mean: D*mu1(X) + (1-D)*mu0(X).
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
        writeln!(
            f,
            "Wald inference requires consistent outcome and propensity models."
        )?;
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
        sorted.sort_by(|a, b| a.total_cmp(b));
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
    /// * `n_folds` - Number of cross-fitting folds (default 3, at least 2).
    ///   Fits use a local seed 2236067977, independent of previous estimator calls.
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

        if y.iter().chain(x.iter()).any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "DRLearner: outcomes and features must be finite".into(),
            ));
        }
        if n_folds.is_some_and(|folds| folds < 2) {
            return Err(GreenersError::InvalidOperation(
                "DRLearner: cross-fitting requires at least two folds".into(),
            ));
        }
        if variable_names
            .as_ref()
            .is_some_and(|names| names.len() != k)
        {
            return Err(GreenersError::ShapeMismatch(
                "DRLearner: need one name per feature".into(),
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
        indices.shuffle(&mut StdRng::seed_from_u64(2236067977));
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

        let t_vec: Array1<f64> = t.iter().map(|&treated| f64::from(treated)).collect();
        for fold in 0..folds {
            // Nuisance fold = all other folds
            let nuisance_idx: Vec<usize> = (0..n).filter(|&i| fold_assignment[i] != fold).collect();
            let cate_idx: Vec<usize> = (0..n).filter(|&i| fold_assignment[i] == fold).collect();

            if nuisance_idx.is_empty() || cate_idx.is_empty() {
                return Err(GreenersError::InvalidOperation(
                    "DRLearner: empty training or evaluation fold".into(),
                ));
            }

            let treated: Vec<usize> = nuisance_idx.iter().copied().filter(|&i| t[i]).collect();
            let control: Vec<usize> = nuisance_idx.iter().copied().filter(|&i| !t[i]).collect();
            let mu1_beta = Self::ols_subset(y, x, &treated, k)?;
            let mu0_beta = Self::ols_subset(y, x, &control, k)?;
            let e_beta = Self::ols_subset(&t_vec, x, &nuisance_idx, k)?;

            // Predict on CATE fold
            for &i in &cate_idx {
                let mu1 = Self::predict_ols(&mu1_beta, &x.row(i).to_owned(), k);
                let mu0 = Self::predict_ols(&mu0_beta, &x.row(i).to_owned(), k);
                let propensity = Self::predict_ols(&e_beta, &x.row(i).to_owned(), k);
                if !propensity.is_finite() {
                    return Err(GreenersError::InvalidOperation(
                        "DRLearner: non-finite propensity prediction".into(),
                    ));
                }
                let e_pred = propensity.clamp(0.01, 0.99);
                m_hat_all[i] = if t[i] { mu1 } else { mu0 };
                e_hat_all[i] = e_pred;

                // DR pseudo-outcome
                let ti = if t[i] { 1.0 } else { 0.0 };
                let psi = mu1 - mu0 + ti * (y[i] - mu1) / e_pred
                    - (1.0 - ti) * (y[i] - mu0) / (1.0 - e_pred);
                if !psi.is_finite() || !mu1.is_finite() || !mu0.is_finite() || !e_pred.is_finite() {
                    return Err(GreenersError::InvalidOperation(
                        "DRLearner: nuisance predictions or score exceed numerical precision"
                            .into(),
                    ));
                }
                pseudo_outcomes[i] = psi;
            }
        }

        // Regress pseudo-outcomes on X for CATE model
        let cate_beta = Self::ols_subset(&pseudo_outcomes, x, &(0..n).collect::<Vec<_>>(), k)?;

        // Predict CATE for all observations
        let mut cate = Array1::zeros(n);
        for i in 0..n {
            cate[i] = Self::predict_ols(&cate_beta, &x.row(i).to_owned(), k);
        }

        // ATE = mean of pseudo-outcomes
        let ate = pseudo_outcomes.mean().unwrap_or(0.0);

        // The stable norm preserves SE = sqrt(sum((score - ATE)^2)) / n
        // without squaring deviations that can underflow or overflow.
        let deviation_norm = pseudo_outcomes
            .iter()
            .fold(0.0_f64, |norm, &v| norm.hypot(v - ate));
        let ate_se = deviation_norm / n as f64;

        let z = 1.959964;
        let ate_ci = [ate - z * ate_se, ate + z * ate_se];

        if !ate.is_finite()
            || !ate_se.is_finite()
            || cate.iter().chain(ate_ci.iter()).any(|v| !v.is_finite())
        {
            return Err(GreenersError::InvalidOperation(
                "DRLearner: effect summaries exceed numerical precision".into(),
            ));
        }

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
        if n < k + 1 {
            return Err(GreenersError::InvalidOperation(
                "DRLearner: insufficient training-arm observations for the nuisance design".into(),
            ));
        }
        // Column scaling makes the rank decision insensitive to predictor units.
        // Reject unidentified models instead of adding an undeclared ridge prior.
        let scales: Vec<f64> = x_full
            .columns()
            .into_iter()
            .map(|col| col.iter().fold(0.0_f64, |norm, &v| norm.hypot(v)))
            .collect();
        for (j, &scale) in scales.iter().enumerate() {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(GreenersError::InvalidOperation(
                    "DRLearner: nuisance design is rank deficient or non-finite".into(),
                ));
            }
            x_full.column_mut(j).mapv_inplace(|v| v / scale);
        }
        let matrix = Mat::from_fn(n, k + 1, |i, j| x_full[(i, j)]);
        let svd = Svd::new_thin(matrix.as_ref()).map_err(|_| GreenersError::OptimizationFailed)?;
        let singular = svd.S().column_vector();
        let tolerance = f64::EPSILON
            * n.max(k + 1) as f64
            * (0..k + 1).map(|j| singular[j]).fold(0.0, f64::max);
        if (0..k + 1).any(|j| !singular[j].is_finite() || singular[j] <= tolerance) {
            return Err(GreenersError::InvalidOperation(
                "DRLearner: nuisance design is rank deficient".into(),
            ));
        }
        let projected: Vec<f64> = (0..k + 1)
            .map(|j| (0..n).map(|i| svd.U()[(i, j)] * y_sub[i]).sum::<f64>() / singular[j])
            .collect();
        let beta = Array1::from_shape_fn(k + 1, |i| {
            (0..k + 1)
                .map(|j| svd.V()[(i, j)] * projected[j])
                .sum::<f64>()
                / scales[i]
        });
        if beta.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "DRLearner: non-finite nuisance coefficients".into(),
            ));
        }
        Ok(beta)
    }

    fn predict_ols(beta: &Array1<f64>, x: &Array1<f64>, k: usize) -> f64 {
        let mut pred = beta[0];
        for j in 0..k {
            pred += beta[j + 1] * x[j];
        }
        pred
    }
}
