use crate::error::GreenersError;
use crate::glm::{Family, GLM};
use crate::ols::PredictionResult;
use crate::{CovarianceType, DataFrame, Formula, InferenceType};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result from Poisson regression.
#[derive(Debug, Clone)]
pub struct PoissonResult {
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
    pub n_obs: usize,
    pub df_resid: usize,
    pub df_model: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub inference_type: InferenceType,
    pub variable_names: Option<Vec<String>>,
    pub omitted_vars: Vec<(usize, String)>,
    _x_data: Array2<f64>,
    _y_data: Array1<f64>,
}

impl fmt::Display for PoissonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Poisson Regression Results ")?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Dep. Variable:", "y", "Log-Likelihood:", self.log_likelihood
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Model:", "Poisson", "Pseudo R-sq:", self.pseudo_r2
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Method:", "IRLS", "Deviance:", self.deviance
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "No. Observations:", self.n_obs, "Pearson chi2:", self.pearson_chi2
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Df Residuals:", self.df_resid, "AIC:", self.aic
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Df Model:", self.df_model, "BIC:", self.bic
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}",
            "", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"
        )?;
        writeln!(f, "{:-^78}", "")?;

        let total = self.params.len() + self.omitted_vars.len();
        let mut fit_idx = 0usize;
        for pos in 0..total {
            if let Some((_, name)) = self.omitted_vars.iter().find(|(p, _)| *p == pos) {
                writeln!(f, "{:<12} (omitted)", name)?;
            } else {
                let name = self
                    .variable_names
                    .as_ref()
                    .and_then(|n| n.get(fit_idx).cloned())
                    .unwrap_or_else(|| format!("x{}", fit_idx));
                writeln!(
                    f,
                    "{:<12} {:>10.4} {:>10.4} {:>8.3} {:>8.3} {:>10.3} {:>10.3}",
                    name,
                    self.params[fit_idx],
                    self.std_errors[fit_idx],
                    self.z_values[fit_idx],
                    self.p_values[fit_idx],
                    self.conf_lower[fit_idx],
                    self.conf_upper[fit_idx]
                )?;
                fit_idx += 1;
            }
        }

        writeln!(f, "{:=^78}", "")?;
        for (_, name) in &self.omitted_vars {
            writeln!(f, "note: {} omitted because of collinearity", name)?;
        }
        Ok(())
    }
}

impl PoissonResult {
    /// X matrix used in estimation (for marginal effects).
    pub fn x_data(&self) -> &Array2<f64> { &self._x_data }

    /// Predict expected counts for new data.
    pub fn predict_count(&self, x_new: &Array2<f64>) -> Array1<f64> {
        let eta = x_new.dot(&self.params);
        eta.mapv(f64::exp)
    }

    /// Fitted values (in-sample predicted counts).
    pub fn fitted_values(&self) -> Array1<f64> {
        self.predict_count(&self._x_data)
    }

    /// Deviance residuals.
    pub fn residuals(&self) -> Array1<f64> {
        let mu = self.fitted_values();
        let n = self._y_data.len();
        let mut resid = Array1::<f64>::zeros(n);
        for i in 0..n {
            let y = self._y_data[i];
            let m = mu[i].max(1e-10);
            let d = if y > 1e-10 {
                2.0 * (y * (y / m).ln() - (y - m))
            } else {
                2.0 * m
            };
            resid[i] = d.max(0.0).sqrt() * (y - m).signum();
        }
        resid
    }

    /// Pearson residuals: (y - μ) / sqrt(μ).
    pub fn pearson_residuals(&self) -> Array1<f64> {
        let mu = self.fitted_values();
        (&self._y_data - &mu) / mu.mapv(|m| m.max(1e-10).sqrt())
    }

    /// Average Marginal Effects.
    /// For Poisson with log link: AME_j = mean(μ_i) * β_j = mean(exp(x'β)) * β_j.
    pub fn marginal_effects(&self, x: &Array2<f64>) -> Array1<f64> {
        let mu = self.predict_count(x);
        let mean_mu = mu.mean().unwrap_or(1.0);
        &self.params * mean_mu
    }

    /// Confidence intervals at arbitrary alpha.
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

        // SE via delta method: SE(mu) = mu * SE(eta)
        // For now, approximate SE(eta) from the diagonal
        let n = x_new.nrows();
        let mut se = Array1::<f64>::zeros(n);
        for i in 0..n {
            // SE(eta_i) ≈ sqrt(x_i' * Cov * x_i) — use std_errors as proxy
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

    /// Model comparison statistics: (AIC, BIC, LogLik, PseudoR2).
    pub fn model_stats(&self) -> (f64, f64, f64, f64) {
        (self.aic, self.bic, self.log_likelihood, self.pseudo_r2)
    }

    /// Cameron-Trivedi overdispersion test.
    /// Tests H0: Var(y) = μ (equidispersion) vs H1: Var(y) = μ + α*g(μ).
    /// Returns (t_statistic, p_value).
    pub fn overdispersion_test(&self) -> Result<(f64, f64), GreenersError> {
        let mu = self.fitted_values();
        let n = self._y_data.len();

        // Auxiliary regression: ((y - μ)² - y) / μ on μ
        // Under H0 (equidispersion), the coefficient on μ is 0
        let mut dep = Array1::<f64>::zeros(n);
        let mut reg = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i].max(1e-10);
            dep[i] = ((self._y_data[i] - m).powi(2) - self._y_data[i]) / m;
            reg[i] = m;
        }

        // Simple regression: dep = alpha * reg + error
        let reg_mean = reg.mean().unwrap_or(0.0);
        let dep_mean = dep.mean().unwrap_or(0.0);
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            num += (reg[i] - reg_mean) * (dep[i] - dep_mean);
            den += (reg[i] - reg_mean).powi(2);
        }

        if den < 1e-15 {
            return Err(GreenersError::SingularMatrix);
        }

        let alpha_hat = num / den;
        let residuals: Array1<f64> = (0..n).map(|i| dep[i] - alpha_hat * reg[i]).collect();
        let sigma2 = residuals.mapv(|r| r.powi(2)).sum() / (n - 1) as f64;
        let se_alpha = (sigma2 / den).sqrt();

        if se_alpha < 1e-15 {
            return Err(GreenersError::SingularMatrix);
        }

        let t_stat = alpha_hat / se_alpha;
        // Two-sided test using normal approximation
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));

        Ok((t_stat, p_value))
    }

    /// Goodness-of-fit test based on deviance.
    /// Under H0 (correct specification), deviance ~ chi2(df_resid).
    /// Returns (deviance, p_value).
    pub fn goodness_of_fit(&self) -> (f64, f64) {
        let chi2_p = 1.0
            - statrs::distribution::ChiSquared::new(self.df_resid as f64)
                .map(|d| {
                    use statrs::distribution::ContinuousCDF;
                    d.cdf(self.deviance)
                })
                .unwrap_or(0.0);
        (self.deviance, chi2_p)
    }
}

/// Poisson regression estimator.
pub struct Poisson;

impl Poisson {
    /// Fit via formula and DataFrame.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        cov_type: CovarianceType,
    ) -> Result<PoissonResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        let var_names = data.formula_var_names(formula)?;
        Self::fit_with_names(&y, &x, cov_type, Some(var_names))
    }

    /// Fit from arrays.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
    ) -> Result<PoissonResult, GreenersError> {
        Self::fit_with_names(y, x, cov_type, None)
    }

    /// Fit with variable names.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
    ) -> Result<PoissonResult, GreenersError> {
        let glm_result = GLM::fit_with_names(y, x, Family::Poisson, cov_type, variable_names)?;

        Ok(PoissonResult {
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
            n_obs: glm_result.n_obs,
            df_resid: glm_result.df_resid,
            df_model: glm_result.df_model,
            n_iter: glm_result.n_iter,
            converged: glm_result.converged,
            inference_type: glm_result.inference_type,
            variable_names: glm_result.variable_names,
            omitted_vars: glm_result.omitted_vars,
            _x_data: glm_result._x_data.clone(),
            _y_data: glm_result._y_data.clone(),
        })
    }

    /// Fit with exposure (offset = ln(exposure)).
    pub fn fit_with_exposure(
        y: &Array1<f64>,
        x: &Array2<f64>,
        exposure: &Array1<f64>,
        cov_type: CovarianceType,
    ) -> Result<PoissonResult, GreenersError> {
        // Add ln(exposure) as an additional column with coefficient fixed to 1
        // Approximation: include it as a regular column
        let n = x.nrows();
        let k = x.ncols();
        let mut x_with_offset = Array2::<f64>::zeros((n, k + 1));
        x_with_offset.slice_mut(ndarray::s![.., ..k]).assign(x);
        for i in 0..n {
            x_with_offset[[i, k]] = exposure[i].max(1e-10).ln();
        }
        Self::fit(y, &x_with_offset, cov_type)
    }
}
