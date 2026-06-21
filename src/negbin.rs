use crate::error::GreenersError;
use crate::glm::{Family, GLM};
use crate::linalg::LinalgInverse as _;
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
    /// X matrix used in estimation (for marginal effects).
    pub fn x_data(&self) -> &Array2<f64> { &self._x_data }

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

/// Result from Generalized Poisson regression.
#[derive(Debug, Clone)]
pub struct GenPoissonResult {
    pub params: Array1<f64>,
    pub alpha: f64,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for GenPoissonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" Generalized Poisson (alpha={:.4}) ", self.alpha)
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "No. Observations:", self.n_obs, "Log-Likelihood:", self.log_likelihood
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "AIC:", self.aic, "BIC:", self.bic
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
        writeln!(f, "{:=^78}", "")
    }
}

/// Result from NegBinP regression.
#[derive(Debug, Clone)]
pub struct NegBinPResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub aic: f64,
    pub bic: f64,
    pub alpha: f64,
    pub p_param: f64,
    pub n_obs: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for NegBinPResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" NegBinP (p={:.1}, alpha={:.4}) ", self.p_param, self.alpha)
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "No. Observations:", self.n_obs, "Log-Likelihood:", self.log_likelihood
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "AIC:", self.aic, "BIC:", self.bic
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
        writeln!(f, "{:=^78}", "")
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
        let var_names = data.formula_var_names(formula)?;
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

/// NegBinP: Negative Binomial with flexible P parameter.
///
/// Variance function: Var(Y) = mu + alpha * mu^p
/// - p=1 gives NB1 (linear variance)
/// - p=2 gives NB2 (quadratic variance, standard NegBin)
pub struct NegBinP;

impl NegBinP {
    /// Fit NegBinP model via IRLS with given p parameter.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        p_param: f64,
    ) -> Result<NegBinPResult, GreenersError> {
        Self::fit_with_names(y, x, p_param, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        p_param: f64,
        variable_names: Option<Vec<String>>,
    ) -> Result<NegBinPResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        // Initialize with Poisson (via OLS on log(y+0.5))
        let log_y: Array1<f64> = y.mapv(|v| (v + 0.5).ln());
        let ols = crate::OLS::fit(&log_y, x, CovarianceType::NonRobust)?;
        let mut beta = ols.params.clone();

        // Estimate initial alpha from method of moments
        let mu_init = x.dot(&beta).mapv(f64::exp);
        let mut alpha = {
            let mut s = 0.0;
            for i in 0..n {
                let m = mu_init[i].max(1e-10);
                s += ((y[i] - m).powi(2) - m) / m.powf(p_param);
            }
            (s / n as f64).max(0.01)
        };

        let max_iter = 100;
        let tol = 1e-6;
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            n_iter = iter + 1;
            let eta = x.dot(&beta);
            let mu: Array1<f64> = eta.mapv(|e| e.exp().max(1e-10));

            // IRLS weights: w_i = mu_i^2 / V(mu_i), where V = mu + alpha*mu^p
            let mut w = Array1::<f64>::zeros(n);
            let mut z = Array1::<f64>::zeros(n);
            for i in 0..n {
                let m = mu[i];
                let v = (m + alpha * m.powf(p_param)).max(1e-10);
                w[i] = m * m / v;
                z[i] = eta[i] + (y[i] - m) / m;
            }

            // Weighted least squares: beta = (X'WX)^-1 X'Wz
            let mut xtwx = Array2::<f64>::zeros((k, k));
            let mut xtwz = Array1::<f64>::zeros(k);
            for i in 0..n {
                let xi = x.row(i);
                for a in 0..k {
                    xtwz[a] += w[i] * xi[a] * z[i];
                    for b in 0..k {
                        xtwx[[a, b]] += w[i] * xi[a] * xi[b];
                    }
                }
            }

            let xtwx_inv: Array2<f64> = match xtwx.inv() {
                Ok(inv) => inv,
                Err(_) => break,
            };
            let new_beta = xtwx_inv.dot(&xtwz);

            // Update alpha via moment estimator
            let eta_new = x.dot(&new_beta);
            let mu_new: Array1<f64> = eta_new.mapv(|e: f64| e.exp().max(1e-10));
            let mut alpha_num = 0.0;
            for i in 0..n {
                let m = mu_new[i];
                alpha_num += ((y[i] - m).powi(2) - m) / m.powf(p_param);
            }
            alpha = (alpha_num / n as f64).max(1e-6);

            let change = &new_beta - &beta;
            let diff = change
                .mapv(|d: f64| d.abs())
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            beta = new_beta;

            if diff < tol {
                converged = true;
                break;
            }
        }

        // Log-likelihood for NegBinP
        let mu: Array1<f64> = x.dot(&beta).mapv(|e| e.exp().max(1e-10));
        let mut ll = 0.0;
        for i in 0..n {
            let m = mu[i];
            let yi = y[i];
            let r = 1.0 / (alpha * m.powf(p_param - 1.0)).max(1e-10);
            // NB log-likelihood with r = 1/(alpha * mu^(p-1))
            ll += statrs::function::gamma::ln_gamma(yi + r)
                - statrs::function::gamma::ln_gamma(r)
                - statrs::function::gamma::ln_gamma(yi + 1.0)
                + r * (r / (r + m)).max(1e-15).ln()
                + yi * (m / (r + m)).max(1e-15).ln();
        }

        // Deviance
        let mut deviance = 0.0;
        for i in 0..n {
            let m = mu[i];
            let yi = y[i];
            if yi > 1e-10 {
                deviance += 2.0 * yi * (yi / m).ln();
            }
            deviance -= 2.0 * (yi - m);
        }

        // Standard errors from Fisher information
        let mut fisher = Array2::<f64>::zeros((k, k));
        for i in 0..n {
            let m = mu[i];
            let v = (m + alpha * m.powf(p_param)).max(1e-10);
            let wi = m * m / v;
            let xi = x.row(i);
            for a in 0..k {
                for b in 0..k {
                    fisher[[a, b]] += wi * xi[a] * xi[b];
                }
            }
        }

        let cov = fisher.inv().unwrap_or(Array2::eye(k) * 1e-4);
        let std_errors: Array1<f64> = (0..k).map(|i| cov[[i, i]].max(0.0).sqrt()).collect();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_values = &beta / std_errors.mapv(|s| if s > 1e-15 { s } else { 1.0 });
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal.cdf(z.abs())));
        let z_crit = normal.inverse_cdf(0.975);
        let conf_lower = &beta - z_crit * &std_errors;
        let conf_upper = &beta + z_crit * &std_errors;

        let k_f = (k + 1) as f64; // +1 for alpha
        let aic = -2.0 * ll + 2.0 * k_f;
        let bic = -2.0 * ll + k_f * (n as f64).ln();

        Ok(NegBinPResult {
            params: beta,
            std_errors,
            z_values,
            p_values,
            conf_lower,
            conf_upper,
            log_likelihood: ll,
            deviance,
            aic,
            bic,
            alpha,
            p_param,
            n_obs: n,
            n_iter,
            converged,
            variable_names,
        })
    }
}

/// Generalized Poisson regression.
///
/// P(Y=y) = mu*(mu + alpha*y)^(y-1) * exp(-(mu + alpha*y)) / y!
pub struct GenPoisson;

impl GenPoisson {
    pub fn fit(y: &Array1<f64>, x: &Array2<f64>) -> Result<GenPoissonResult, GreenersError> {
        Self::fit_with_names(y, x, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<GenPoissonResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        // Initialize beta from Poisson (log-linear)
        let log_y: Array1<f64> = y.mapv(|v| (v + 0.5).ln());
        let ols = crate::OLS::fit(&log_y, x, CovarianceType::NonRobust)?;
        let mut beta = ols.params.clone();
        let mut alpha = 0.1_f64;

        let max_iter = 100;
        let tol = 1e-6;
        let mut converged = false;
        let mut n_iter = 0;

        // Newton-Raphson on full (beta, alpha) log-likelihood
        for iter in 0..max_iter {
            n_iter = iter + 1;
            let mu: Array1<f64> = x.dot(&beta).mapv(|e| e.exp().max(1e-10));

            // Gradient and Hessian
            let mut grad_beta = Array1::<f64>::zeros(k);
            let mut grad_alpha = 0.0;
            let mut hess_bb = Array2::<f64>::zeros((k, k));

            for i in 0..n {
                let m = mu[i];
                let yi = y[i];
                let t = m + alpha * yi;
                if t <= 0.0 {
                    continue;
                }

                // d ll / d mu_i = 1/mu + (y-1)/t - 1, then chain rule * mu * x
                let dll_dmu = 1.0 / m + if yi > 1.0 { (yi - 1.0) / t } else { 0.0 } - 1.0;
                let dm_dbeta = m; // d mu / d eta * d eta / d beta_j = mu * x_j

                for j in 0..k {
                    grad_beta[j] += dll_dmu * dm_dbeta * x[[i, j]];
                }

                // d ll / d alpha = (y-1)*y/t - y
                grad_alpha += if yi > 1.0 { (yi - 1.0) * yi / t } else { 0.0 } - yi;

                // Approximate Hessian for beta (expected information)
                let d2 = -1.0 / (m * m) - if yi > 1.0 { (yi - 1.0) / (t * t) } else { 0.0 };
                let wi = -(d2 * m * m + dll_dmu * m);
                for a in 0..k {
                    for b in 0..k {
                        hess_bb[[a, b]] -= wi * x[[i, a]] * x[[i, b]];
                    }
                }
            }

            // Update beta
            let neg_hess_inv = match (-&hess_bb).inv() {
                Ok(inv) => inv,
                Err(_) => break,
            };
            let delta_beta = neg_hess_inv.dot(&grad_beta);
            let new_beta = &beta + &delta_beta;

            // Update alpha via simple gradient step with line search
            let step_alpha = 0.01 * grad_alpha.signum() * grad_alpha.abs().min(0.1);
            let new_alpha = (alpha + step_alpha).clamp(-0.99, 0.99);

            let diff = delta_beta
                .mapv(|d| d.abs())
                .iter()
                .copied()
                .fold(0.0_f64, f64::max)
                + (new_alpha - alpha).abs();

            beta = new_beta;
            alpha = new_alpha;

            if diff < tol {
                converged = true;
                break;
            }
        }

        // Final log-likelihood
        let mu: Array1<f64> = x.dot(&beta).mapv(|e| e.exp().max(1e-10));
        let mut ll = 0.0;
        for i in 0..n {
            let m = mu[i];
            let yi = y[i];
            let t = m + alpha * yi;
            if t > 0.0 {
                ll += m.ln() + if yi > 1.0 { (yi - 1.0) * t.ln() } else { 0.0 }
                    - t
                    - statrs::function::gamma::ln_gamma(yi + 1.0);
            }
        }

        // Standard errors from numerical Hessian
        let mut fisher = Array2::<f64>::zeros((k, k));
        for i in 0..n {
            let m = mu[i];
            let yi = y[i];
            let t = (m + alpha * yi).max(1e-10);
            let w = 1.0 / m + if yi > 1.0 { (yi - 1.0) / (t * t) } else { 0.0 };
            let wi = w * m * m;
            let xi = x.row(i);
            for a in 0..k {
                for b in 0..k {
                    fisher[[a, b]] += wi * xi[a] * xi[b];
                }
            }
        }

        let cov = fisher.inv().unwrap_or(Array2::eye(k) * 1e-4);
        let std_errors: Array1<f64> = (0..k).map(|i| cov[[i, i]].max(0.0).sqrt()).collect();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_values = &beta / std_errors.mapv(|s| if s > 1e-15 { s } else { 1.0 });
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal.cdf(z.abs())));
        let z_crit = normal.inverse_cdf(0.975);
        let conf_lower = &beta - z_crit * &std_errors;
        let conf_upper = &beta + z_crit * &std_errors;

        let k_f = (k + 1) as f64;
        let aic = -2.0 * ll + 2.0 * k_f;
        let bic = -2.0 * ll + k_f * (n as f64).ln();

        Ok(GenPoissonResult {
            params: beta,
            alpha,
            std_errors,
            z_values,
            p_values,
            conf_lower,
            conf_upper,
            log_likelihood: ll,
            aic,
            bic,
            n_obs: n,
            n_iter,
            converged,
            variable_names,
        })
    }
}
