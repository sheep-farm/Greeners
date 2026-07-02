use ndarray::{Array1, Array2};

/// Average Marginal Effects result with standard errors.
#[derive(Debug, Clone)]
pub struct MarginalEffectsResult {
    pub variable_names: Vec<String>,
    pub effects: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub n_obs: usize,
}

/// Marginal effects computation for nonlinear models.
pub struct Margins;

impl Margins {
    /// Average Marginal Effects for Logit with delta method SEs.
    ///
    /// AME_k = (1/n) Σ_i Λ'(X_iβ) · β_k
    /// SE via numerical gradient of AME w.r.t. β, sandwiched with V.
    pub fn ame_logit(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
    ) -> MarginalEffectsResult {
        Self::ame_generic(params, x, variable_names, LinkFn::Logit, None)
    }

    /// AME for Logit with covariance matrix (for SEs).
    pub fn ame_logit_with_vcov(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
        vcov: &Array2<f64>,
    ) -> MarginalEffectsResult {
        Self::ame_generic(params, x, variable_names, LinkFn::Logit, Some(vcov))
    }

    /// Average Marginal Effects for Probit with delta method SEs.
    pub fn ame_probit(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
    ) -> MarginalEffectsResult {
        Self::ame_generic(params, x, variable_names, LinkFn::Probit, None)
    }

    pub fn ame_probit_with_vcov(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
        vcov: &Array2<f64>,
    ) -> MarginalEffectsResult {
        Self::ame_generic(params, x, variable_names, LinkFn::Probit, Some(vcov))
    }

    /// Average Marginal Effects for Poisson/NegBin (exponential mean).
    pub fn ame_exponential(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
    ) -> MarginalEffectsResult {
        Self::ame_generic(params, x, variable_names, LinkFn::Exponential, None)
    }

    pub fn ame_exponential_with_vcov(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
        vcov: &Array2<f64>,
    ) -> MarginalEffectsResult {
        Self::ame_generic(params, x, variable_names, LinkFn::Exponential, Some(vcov))
    }

    /// Compute AME on a modified X matrix (for `at=` functionality).
    pub fn with_at(x: &Array2<f64>, col_idx: usize, value: f64) -> Array2<f64> {
        let mut x_mod = x.clone();
        x_mod.column_mut(col_idx).fill(value);
        x_mod
    }

    fn ame_generic(
        params: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: &[String],
        link: LinkFn,
        vcov: Option<&Array2<f64>>,
    ) -> MarginalEffectsResult {
        let n = x.nrows();
        let k = x.ncols();

        let compute_effects = |beta: &[f64]| -> Vec<f64> {
            let beta_arr = Array1::from(beta.to_vec());
            (0..k)
                .map(|j| {
                    let sum: f64 = (0..n)
                        .map(|i| {
                            let eta = x.row(i).dot(&beta_arr);
                            let deriv = match link {
                                LinkFn::Logit => {
                                    let p = crate::logistic(eta);
                                    p * (1.0 - p)
                                }
                                LinkFn::Probit => crate::norm_pdf(eta),
                                LinkFn::Exponential => eta.exp(),
                            };
                            deriv * beta_arr[j]
                        })
                        .sum();
                    sum / n as f64
                })
                .collect()
        };

        let effects = compute_effects(params.as_slice().unwrap());

        // Delta method SEs via numerical gradient
        let std_errors = if let Some(v) = vcov {
            let h = 1e-7;
            let mut se = vec![0.0; k];
            for j in 0..k {
                let mut grad = Array1::<f64>::zeros(k);
                let params_slice = params.as_slice().unwrap();
                for p in 0..k {
                    let mut beta_plus = params_slice.to_vec();
                    let mut beta_minus = params_slice.to_vec();
                    beta_plus[p] += h;
                    beta_minus[p] -= h;
                    let ame_plus = compute_effects(&beta_plus);
                    let ame_minus = compute_effects(&beta_minus);
                    grad[p] = (ame_plus[j] - ame_minus[j]) / (2.0 * h);
                }
                se[j] = grad.dot(&v.dot(&grad)).max(0.0).sqrt();
            }
            se
        } else {
            vec![f64::NAN; k]
        };

        let z_values: Vec<f64> = effects
            .iter()
            .zip(&std_errors)
            .map(|(&e, &s)| if s > 1e-15 { e / s } else { f64::NAN })
            .collect();

        let p_values: Vec<f64> = z_values
            .iter()
            .map(|&z| {
                if z.is_finite() {
                    crate::t_pvalue_two(z, 1e12)
                } else {
                    f64::NAN
                }
            })
            .collect();

        MarginalEffectsResult {
            variable_names: variable_names.to_vec(),
            effects,
            std_errors,
            z_values,
            p_values,
            n_obs: n,
        }
    }
}

enum LinkFn {
    Logit,
    Probit,
    Exponential,
}
