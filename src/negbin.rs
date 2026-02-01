use crate::error::GreenersError;
use crate::glm::{Family, GLM};
use crate::ols::PredictionResult;
use crate::{CovarianceType, DataFrame, Formula, InferenceType};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result from Negative Binomial regression.
#[derive(Debug, Clone)]
pub struct NegBinResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub aic: f64,
    pub bic: f64,
    pub pseudo_r2: f64,
    pub pearson_chi2: f64,
    pub alpha: f64,
    pub n_obs: usize,
    pub df_resid: usize,
    pub df_model: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub inference_type: InferenceType,
    pub variable_names: Option<Vec<String>>,
    pub omitted_vars: Vec<String>,
    _x_data: Array2<f64>,
    _y_data: Array1<f64>,
}

impl fmt::Display for NegBinResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" Negative Binomial Regression (alpha={:.4}) ", self.alpha)
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Dep. Variable:", "y", "Log-Likelihood:", self.log_likelihood
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Model:", "NegBin", "Pseudo R-sq:", self.pseudo_r2
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "No. Observations:", self.n_obs, "Deviance:", self.deviance
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Df Residuals:", self.df_resid, "AIC:", self.aic
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}",
            "", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} {:>10.4} {:>10.4} {:>8.3} {:>8.3} {:>10.3} {:>10.3}",
                name,
                self.params[i],
                self.std_errors[i],
                self.z_values[i],
                self.p_values[i],
                self.conf_lower[i],
                self.conf_upper[i]
            )?;
        }

        if !self.omitted_vars.is_empty() {
            writeln!(f, "{:-^78}", "")?;
            writeln!(f, "Omitted due to collinearity:")?;
            for var in &self.omitted_vars {
                writeln!(f, "  o.{}", var)?;
            }
        }

        writeln!(f, "{:=^78}", "")
    }
}

impl NegBinResult {
    /// Predict expected counts.
    pub fn predict_count(&self, x_new: &Array2<f64>) -> Array1<f64> {
        x_new.dot(&self.params).mapv(f64::exp)
    }

    /// Fitted values.
    pub fn fitted_values(&self) -> Array1<f64> {
        self.predict_count(&self._x_data)
    }

    /// Pearson residuals.
    pub fn pearson_residuals(&self) -> Array1<f64> {
        let mu = self.fitted_values();
        let n = self._y_data.len();
        let mut resid = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i].max(1e-10);
            let v = m + self.alpha * m * m;
            resid[i] = (self._y_data[i] - m) / v.sqrt();
        }
        resid
    }

    /// Deviance residuals.
    pub fn residuals(&self) -> Array1<f64> {
        let mu = self.fitted_values();
        let n = self._y_data.len();
        let alpha = self.alpha;
        let mut resid = Array1::<f64>::zeros(n);
        for i in 0..n {
            let y = self._y_data[i];
            let m = mu[i].max(1e-10);
            let d = if y > 1e-10 {
                2.0 * (y * (y / m).ln()
                    - (y + 1.0 / alpha) * ((1.0 + alpha * y) / (1.0 + alpha * m)).ln())
            } else {
                2.0 / alpha * (1.0 + alpha * m).ln()
            };
            resid[i] = d.max(0.0).sqrt() * (y - m).signum();
        }
        resid
    }

    /// Average Marginal Effects.
    pub fn marginal_effects(&self, x: &Array2<f64>) -> Array1<f64> {
        let mu = self.predict_count(x);
        let mean_mu = mu.mean().unwrap_or(1.0);
        &self.params * mean_mu
    }

    /// Confidence intervals.
    pub fn conf_int(&self, alpha: f64) -> Vec<(f64, f64)> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - alpha / 2.0);
        (0..self.params.len())
            .map(|i| {
                (
                    self.params[i] - z * self.std_errors[i],
                    self.params[i] + z * self.std_errors[i],
                )
            })
            .collect()
    }

    /// Prediction with SE and CI.
    pub fn get_prediction(&self, x_new: &Array2<f64>, alpha: f64) -> PredictionResult {
        let eta = x_new.dot(&self.params);
        let mu = eta.mapv(f64::exp);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - alpha / 2.0);

        let n = x_new.nrows();
        let mut se = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x_i = x_new.row(i);
            let var_eta: f64 = x_i
                .iter()
                .zip(self.std_errors.iter())
                .map(|(xi, sei)| xi * xi * sei * sei)
                .sum();
            se[i] = mu[i] * var_eta.sqrt();
        }

        PredictionResult {
            mean: mu.clone(),
            se: se.clone(),
            ci_lower: &mu - z * &se,
            ci_upper: &mu + z * &se,
        }
    }

    /// Model stats: (AIC, BIC, LogLik, PseudoR2).
    pub fn model_stats(&self) -> (f64, f64, f64, f64) {
        (self.aic, self.bic, self.log_likelihood, self.pseudo_r2)
    }

    /// LR test vs Poisson (H0: alpha = 0).
    /// Returns (lr_stat, p_value).
    /// Uses chi2-bar(1) = 0.5*chi2(0) + 0.5*chi2(1) mixture distribution.
    pub fn lr_test_vs_poisson(&self, poisson_ll: f64) -> (f64, f64) {
        let lr_stat = 2.0 * (self.log_likelihood - poisson_ll);
        // Under H0 (boundary), the mixture gives p = 0.5 * P(chi2(1) > LR)
        let p_value = if lr_stat > 0.0 {
            let chi2 = statrs::distribution::ChiSquared::new(1.0).unwrap();
            0.5 * (1.0 - chi2.cdf(lr_stat))
        } else {
            1.0
        };
        (lr_stat, p_value)
    }
}

/// Negative Binomial regression estimator.
pub struct NegBin;

impl NegBin {
    /// Fit via formula.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        cov_type: CovarianceType,
    ) -> Result<NegBinResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        let mut var_names = Vec::new();
        if formula.intercept {
            var_names.push("const".to_string());
        }
        for var in &formula.independents {
            var_names.push(var.clone());
        }
        Self::fit_with_names(&y, &x, cov_type, Some(var_names))
    }

    /// Fit from arrays with automatic alpha estimation.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
    ) -> Result<NegBinResult, GreenersError> {
        Self::fit_with_names(y, x, cov_type, None)
    }

    /// Fit with known alpha.
    pub fn fit_with_alpha(
        y: &Array1<f64>,
        x: &Array2<f64>,
        alpha: f64,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
    ) -> Result<NegBinResult, GreenersError> {
        let glm_result = GLM::fit_with_names(
            y,
            x,
            Family::NegativeBinomial(alpha),
            cov_type,
            variable_names,
        )?;

        Ok(NegBinResult {
            params: glm_result.params,
            std_errors: glm_result.std_errors,
            z_values: glm_result.z_values,
            p_values: glm_result.p_values,
            conf_lower: glm_result.conf_lower,
            conf_upper: glm_result.conf_upper,
            log_likelihood: glm_result.log_likelihood,
            deviance: glm_result.deviance,
            null_deviance: glm_result.null_deviance,
            aic: glm_result.aic,
            bic: glm_result.bic,
            pseudo_r2: glm_result.pseudo_r2,
            pearson_chi2: glm_result.pearson_chi2,
            alpha,
            n_obs: glm_result.n_obs,
            df_resid: glm_result.df_resid,
            df_model: glm_result.df_model,
            n_iter: glm_result.n_iter,
            converged: glm_result.converged,
            inference_type: glm_result.inference_type,
            variable_names: glm_result.variable_names,
            omitted_vars: glm_result.omitted_vars,
            _x_data: glm_result._x_data,
            _y_data: glm_result._y_data,
        })
    }

    /// Fit with automatic alpha estimation via profile likelihood.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
    ) -> Result<NegBinResult, GreenersError> {
        // Step 1: Fit Poisson to get initial mu estimates
        let poisson_glm =
            GLM::fit_with_names(y, x, Family::Poisson, CovarianceType::NonRobust, None)?;
        let mu = poisson_glm.predict_mean(x);

        // Step 2: Method of moments initial alpha
        let n = y.len() as f64;
        let mut sum_num = 0.0;
        for i in 0..y.len() {
            let m = mu[i].max(1e-10);
            sum_num += ((y[i] - m).powi(2) - y[i]) / (m * m);
        }
        let alpha_init = (sum_num / n).max(0.01);

        // Step 3: Profile likelihood — grid search around initial estimate
        let candidates = [
            alpha_init * 0.1,
            alpha_init * 0.25,
            alpha_init * 0.5,
            alpha_init * 0.75,
            alpha_init,
            alpha_init * 1.5,
            alpha_init * 2.0,
            alpha_init * 4.0,
        ];

        let mut best_alpha = alpha_init;
        let mut best_ll = f64::NEG_INFINITY;

        for &a in &candidates {
            if a <= 0.0 {
                continue;
            }
            if let Ok(res) = GLM::fit_with_names(
                y,
                x,
                Family::NegativeBinomial(a),
                CovarianceType::NonRobust,
                None,
            ) {
                if res.log_likelihood > best_ll {
                    best_ll = res.log_likelihood;
                    best_alpha = a;
                }
            }
        }

        // Step 4: Newton refinement on alpha (1D optimization)
        let h = 0.01 * best_alpha.max(0.001);
        for _ in 0..10 {
            let ll_center = GLM::fit_with_names(
                y,
                x,
                Family::NegativeBinomial(best_alpha),
                CovarianceType::NonRobust,
                None,
            )
            .map(|r| r.log_likelihood)
            .unwrap_or(f64::NEG_INFINITY);

            let ll_plus = GLM::fit_with_names(
                y,
                x,
                Family::NegativeBinomial(best_alpha + h),
                CovarianceType::NonRobust,
                None,
            )
            .map(|r| r.log_likelihood)
            .unwrap_or(f64::NEG_INFINITY);

            let ll_minus = GLM::fit_with_names(
                y,
                x,
                Family::NegativeBinomial((best_alpha - h).max(1e-6)),
                CovarianceType::NonRobust,
                None,
            )
            .map(|r| r.log_likelihood)
            .unwrap_or(f64::NEG_INFINITY);

            let grad = (ll_plus - ll_minus) / (2.0 * h);
            let hess = (ll_plus - 2.0 * ll_center + ll_minus) / (h * h);

            if hess.abs() < 1e-12 || !hess.is_finite() || !grad.is_finite() {
                break;
            }

            let step = -grad / hess;
            let new_alpha = (best_alpha + step).max(1e-6);

            if let Ok(res) = GLM::fit_with_names(
                y,
                x,
                Family::NegativeBinomial(new_alpha),
                CovarianceType::NonRobust,
                None,
            ) {
                if res.log_likelihood > ll_center {
                    best_alpha = new_alpha;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Step 5: Final fit with optimal alpha and requested cov_type
        Self::fit_with_alpha(y, x, best_alpha, cov_type, variable_names)
    }
}
