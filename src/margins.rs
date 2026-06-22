use ndarray::{Array1, Array2};

/// Average Marginal Effects result.
#[derive(Debug, Clone)]
pub struct MarginalEffectsResult {
    pub variable_names: Vec<String>,
    pub effects: Vec<f64>,
    pub n_obs: usize,
}

/// Marginal effects computation for nonlinear models.
pub struct Margins;

impl Margins {
    /// Average Marginal Effects for Logit.
    ///
    /// AME_k = (1/n) Σ_i Λ'(X_iβ) · β_k
    /// where Λ'(z) = Λ(z)(1 - Λ(z)) and Λ(z) = 1/(1+exp(-z))
    pub fn ame_logit(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
    ) -> MarginalEffectsResult {
        let n = x.nrows();
        let k = x.ncols();
        let deriv: Vec<f64> = (0..n)
            .map(|i| {
                let eta = x.row(i).dot(params);
                let p = crate::logistic(eta);
                p * (1.0 - p)
            })
            .collect();
        let effects: Vec<f64> = (0..k)
            .map(|j| deriv.iter().map(|&d| d * params[j]).sum::<f64>() / n as f64)
            .collect();
        MarginalEffectsResult {
            variable_names: variable_names.to_vec(),
            effects,
            n_obs: n,
        }
    }

    /// Average Marginal Effects for Probit.
    ///
    /// AME_k = (1/n) Σ_i φ(X_iβ) · β_k
    /// where φ is the standard normal PDF.
    pub fn ame_probit(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
    ) -> MarginalEffectsResult {
        let n = x.nrows();
        let k = x.ncols();
        let deriv: Vec<f64> = (0..n)
            .map(|i| {
                let eta = x.row(i).dot(params);
                crate::norm_pdf(eta)
            })
            .collect();
        let effects: Vec<f64> = (0..k)
            .map(|j| deriv.iter().map(|&d| d * params[j]).sum::<f64>() / n as f64)
            .collect();
        MarginalEffectsResult {
            variable_names: variable_names.to_vec(),
            effects,
            n_obs: n,
        }
    }

    /// Average Marginal Effects for Poisson/NegBin (exponential mean).
    ///
    /// AME_k = β_k · (1/n) Σ_i exp(X_iβ)
    pub fn ame_exponential(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
    ) -> MarginalEffectsResult {
        let n = x.nrows();
        let k = x.ncols();
        let mu_bar: f64 =
            (0..n).map(|i| x.row(i).dot(params).exp()).sum::<f64>() / n as f64;
        let effects: Vec<f64> = (0..k).map(|j| params[j] * mu_bar).collect();
        MarginalEffectsResult {
            variable_names: variable_names.to_vec(),
            effects,
            n_obs: n,
        }
    }

    /// Compute AME on a modified X matrix (for `at=` functionality).
    ///
    /// Replaces column `col_idx` with `value` before computing AME.
    pub fn with_at(
        x: &Array2<f64>,
        col_idx: usize,
        value: f64,
    ) -> Array2<f64> {
        let mut x_mod = x.clone();
        x_mod.column_mut(col_idx).fill(value);
        x_mod
    }
}
