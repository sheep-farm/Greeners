use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::{CovarianceType, DataFrame, Formula, InferenceType, OLS};
use ndarray::{Array1, Array2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Distribution family for GLM.
#[derive(Debug, Clone)]
pub enum Family {
    /// Gaussian (Normal) — canonical link: Identity
    Gaussian,
    /// Binomial — canonical link: Logit
    Binomial,
    /// Poisson — canonical link: Log
    Poisson,
    /// Gamma — canonical link: InversePower
    Gamma,
    /// Inverse Gaussian — canonical link: InverseSquared
    InverseGaussian,
    /// Tweedie with power parameter p (1 < p < 2 typical)
    Tweedie(f64),
    /// Negative Binomial with dispersion parameter alpha
    NegativeBinomial(f64),
}

/// Link function for GLM.
#[derive(Debug, Clone, PartialEq)]
pub enum Link {
    Identity,
    Log,
    Logit,
    Probit,
    InversePower,
    InverseSquared,
    /// Complementary log-log: g(μ) = log(-log(1-μ))
    CLogLog,
    /// Power link: g(μ) = μ^power (power=0 is Log)
    Power(f64),
    /// Negative binomial link: g(μ) = log(μ / (μ + 1/alpha))
    NegativeBinomial(f64),
    /// Cauchy (quantile) link: g(μ) = tan(π(μ - 0.5))
    Cauchy,
}

impl Family {
    /// Returns the canonical link for this family.
    pub fn canonical_link(&self) -> Link {
        match self {
            Family::Gaussian => Link::Identity,
            Family::Binomial => Link::Logit,
            Family::Poisson => Link::Log,
            Family::Gamma => Link::InversePower,
            Family::InverseGaussian => Link::InverseSquared,
            Family::Tweedie(p) => {
                if (*p).abs() < 1e-10 {
                    Link::Identity
                } else {
                    Link::Log
                }
            }
            Family::NegativeBinomial(_) => Link::Log,
        }
    }

    /// Variance function V(μ).
    fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => (mu * (1.0 - mu)).max(1e-10),
            Family::Poisson => mu.max(1e-10),
            Family::Gamma => (mu * mu).max(1e-10),
            Family::InverseGaussian => (mu * mu * mu).max(1e-10),
            Family::Tweedie(p) => mu.powf(*p).max(1e-10),
            Family::NegativeBinomial(alpha) => (mu + alpha * mu * mu).max(1e-10),
        }
    }

    /// Unit deviance: d(y, μ) for a single observation.
    fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-10);
        let y = y.max(0.0);
        match self {
            Family::Gaussian => (y - mu).powi(2),
            Family::Binomial => {
                let y_c = y.clamp(1e-10, 1.0 - 1e-10);
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                2.0 * (y_c * (y_c / mu_c).ln() + (1.0 - y_c) * ((1.0 - y_c) / (1.0 - mu_c)).ln())
            }
            Family::Poisson => {
                if y > 1e-10 {
                    2.0 * (y * (y / mu).ln() - (y - mu))
                } else {
                    2.0 * mu
                }
            }
            Family::Gamma => 2.0 * (-(y / mu).ln() + (y - mu) / mu),
            Family::InverseGaussian => (y - mu).powi(2) / (mu * mu * y).max(1e-10),
            Family::Tweedie(p) => {
                let p = *p;
                if (p - 1.0).abs() < 1e-10 {
                    // Poisson-like
                    if y > 1e-10 {
                        2.0 * (y * (y / mu).ln() - (y - mu))
                    } else {
                        2.0 * mu
                    }
                } else if (p - 2.0).abs() < 1e-10 {
                    // Gamma-like
                    2.0 * (-(y / mu).ln() + (y - mu) / mu)
                } else {
                    // General Tweedie
                    let a = y.max(1e-10).powf(2.0 - p) / ((1.0 - p) * (2.0 - p));
                    let b = y * mu.powf(1.0 - p) / (1.0 - p);
                    let c = mu.powf(2.0 - p) / (2.0 - p);
                    2.0 * (a - b + c)
                }
            }
            Family::NegativeBinomial(alpha) => {
                let inv_alpha = 1.0 / alpha;
                let term1 = if y > 1e-10 { y * (y / mu).ln() } else { 0.0 };
                let term2 = (y + inv_alpha) * ((mu + inv_alpha) / (y + inv_alpha)).ln();
                2.0 * (term1 - term2)
            }
        }
    }

    /// Log-likelihood contribution for a single observation.
    fn log_likelihood_obs(&self, y: f64, mu: f64, dispersion: f64) -> f64 {
        let mu = mu.max(1e-10);
        match self {
            Family::Gaussian => {
                // Floor prevents ln(0) when dispersion=0 (perfect fit)
                let sigma2 = dispersion.max(1e-300);
                -0.5 * ((y - mu).powi(2) / sigma2 + sigma2.ln() + std::f64::consts::TAU.ln())
            }
            Family::Binomial => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                y * mu_c.ln() + (1.0 - y) * (1.0 - mu_c).ln()
            }
            Family::Poisson => {
                // y * ln(mu) - mu - ln(y!)
                // Drop the ln(y!) constant
                y * mu.ln() - mu
            }
            Family::Gamma => {
                let nu = 1.0 / dispersion;
                nu * (nu * y / mu).ln() - nu * y / mu - (nu).ln()
                // simplified; drop gamma function constant
            }
            _ => {
                // Fallback: use deviance-based approximation
                -0.5 * self.unit_deviance(y, mu) / dispersion
            }
        }
    }

    /// Starting μ values.
    fn starting_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        match self {
            Family::Binomial => y.mapv(|v| (v + 0.5) / 2.0),
            Family::Poisson | Family::NegativeBinomial(_) => {
                let mean = y.mean().unwrap_or(1.0).max(0.1);
                y.mapv(|v| (v + mean) / 2.0)
            }
            Family::Gamma | Family::InverseGaussian | Family::Tweedie(_) => {
                let mean = y.mean().unwrap_or(1.0).max(0.01);
                y.mapv(|v| (v.max(0.01) + mean) / 2.0)
            }
            Family::Gaussian => y.clone(),
        }
    }

    /// Whether dispersion is fixed (1.0) or estimated.
    fn fixed_dispersion(&self) -> bool {
        matches!(self, Family::Binomial | Family::Poisson)
    }

    fn name(&self) -> String {
        match self {
            Family::Gaussian => "Gaussian".into(),
            Family::Binomial => "Binomial".into(),
            Family::Poisson => "Poisson".into(),
            Family::Gamma => "Gamma".into(),
            Family::InverseGaussian => "InverseGaussian".into(),
            Family::Tweedie(p) => format!("Tweedie(p={:.2})", p),
            Family::NegativeBinomial(a) => format!("NegBin(alpha={:.2})", a),
        }
    }
}

impl Link {
    /// g(μ) → η
    fn link(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => mu,
            Link::Log => mu.max(1e-10).ln(),
            Link::Logit => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                (mu_c / (1.0 - mu_c)).ln()
            }
            Link::Probit => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.inverse_cdf(mu_c)
            }
            Link::InversePower => 1.0 / mu.max(1e-10),
            Link::InverseSquared => 1.0 / (mu * mu).max(1e-10),
            Link::CLogLog => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                (-(1.0 - mu_c).ln()).max(1e-10).ln()
            }
            Link::Power(p) => {
                if p.abs() < 1e-10 {
                    mu.max(1e-10).ln() // Power(0) = Log
                } else {
                    mu.max(1e-10).powf(*p)
                }
            }
            Link::NegativeBinomial(alpha) => {
                let inv_alpha = 1.0 / alpha;
                (mu.max(1e-10) / (mu.max(1e-10) + inv_alpha)).ln()
            }
            Link::Cauchy => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                (std::f64::consts::PI * (mu_c - 0.5)).tan()
            }
        }
    }

    /// g⁻¹(η) → μ
    fn linkinv(&self, eta: f64) -> f64 {
        match self {
            Link::Identity => eta,
            Link::Log => eta.clamp(-30.0, 30.0).exp(),
            Link::Logit => {
                let e = eta.clamp(-30.0, 30.0);
                1.0 / (1.0 + (-e).exp())
            }
            Link::Probit => {
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.cdf(eta)
            }
            Link::InversePower => 1.0 / eta.max(1e-10),
            // Tighter clamp: eta=1e-4 → μ=100, keeping IRLS weights (μ³/4) bounded
            Link::InverseSquared => 1.0 / eta.max(1e-4).sqrt(),
            Link::CLogLog => {
                // g^-1(eta) = 1 - exp(-exp(eta))
                1.0 - (-eta.clamp(-30.0, 30.0).exp()).exp()
            }
            Link::Power(p) => {
                if p.abs() < 1e-10 {
                    eta.clamp(-30.0, 30.0).exp()
                } else {
                    eta.max(1e-10).powf(1.0 / p)
                }
            }
            Link::NegativeBinomial(alpha) => {
                // mu = (1/alpha) * exp(eta) / (1 - exp(eta))
                let inv_alpha = 1.0 / alpha;
                let e = eta.clamp(-30.0, 30.0).exp();
                inv_alpha * e / (1.0 - e).max(1e-10)
            }
            Link::Cauchy => {
                // mu = 0.5 + arctan(eta) / pi
                0.5 + eta.atan() / std::f64::consts::PI
            }
        }
    }

    /// g'(μ) — derivative of the link function
    fn deriv(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => 1.0,
            Link::Log => 1.0 / mu.max(1e-10),
            Link::Logit => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                1.0 / (mu_c * (1.0 - mu_c))
            }
            Link::Probit => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                let normal = Normal::new(0.0, 1.0).unwrap();
                let eta = normal.inverse_cdf(mu_c);
                use statrs::distribution::Continuous;
                1.0 / normal.pdf(eta).max(1e-10)
            }
            Link::InversePower => -1.0 / (mu * mu).max(1e-10),
            Link::InverseSquared => -2.0 / (mu * mu * mu).max(1e-10),
            Link::CLogLog => {
                // g'(mu) = 1 / ((1-mu) * log(1-mu))  [with sign]
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                -1.0 / ((1.0 - mu_c) * (1.0 - mu_c).ln()).abs().max(1e-10)
            }
            Link::Power(p) => {
                if p.abs() < 1e-10 {
                    1.0 / mu.max(1e-10)
                } else {
                    p * mu.max(1e-10).powf(p - 1.0)
                }
            }
            Link::NegativeBinomial(alpha) => {
                let inv_alpha = 1.0 / alpha;
                // d/dmu log(mu/(mu + 1/alpha)) = 1/mu - 1/(mu + 1/alpha) = (1/alpha) / (mu * (mu + 1/alpha))
                inv_alpha / (mu.max(1e-10) * (mu.max(1e-10) + inv_alpha))
            }
            Link::Cauchy => {
                // g'(mu) = pi / cos^2(pi*(mu-0.5))
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                let cos_val = (std::f64::consts::PI * (mu_c - 0.5)).cos();
                std::f64::consts::PI / (cos_val * cos_val).max(1e-10)
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            Link::Identity => "Identity",
            Link::Log => "Log",
            Link::Logit => "Logit",
            Link::Probit => "Probit",
            Link::InversePower => "InversePower",
            Link::InverseSquared => "InverseSquared",
            Link::CLogLog => "CLogLog",
            Link::Power(_) => "Power",
            Link::NegativeBinomial(_) => "NegativeBinomial",
            Link::Cauchy => "Cauchy",
        }
    }
}

/// Result from a GLM estimation.
#[derive(Debug, Clone)]
pub struct GlmResult {
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
    pub dispersion: f64,
    pub n_obs: usize,
    pub df_resid: usize,
    pub df_model: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub family: Family,
    pub link: Link,
    pub inference_type: InferenceType,
    pub variable_names: Option<Vec<String>>,
    pub omitted_vars: Vec<String>,
    // Store design matrix and y for predict/residuals
    pub(crate) _x_data: Array2<f64>,
    pub(crate) _y_data: Array1<f64>,
}

impl GlmResult {
    /// Predict linear predictor (η = Xβ) for new data.
    pub fn predict(&self, x_new: &Array2<f64>) -> Array1<f64> {
        x_new.dot(&self.params)
    }

    /// Predict mean response (μ = g⁻¹(Xβ)) for new data.
    pub fn predict_mean(&self, x_new: &Array2<f64>) -> Array1<f64> {
        let eta = x_new.dot(&self.params);
        eta.mapv(|e| self.link.linkinv(e))
    }

    /// Fitted values (μ̂) from the training data.
    pub fn fitted_values(&self) -> Array1<f64> {
        self.predict_mean(&self._x_data)
    }

    /// Deviance residuals: sign(y-μ) * sqrt(d_i).
    pub fn residuals(&self) -> Array1<f64> {
        let mu = self.fitted_values();
        let n = self._y_data.len();
        let mut resid = Array1::<f64>::zeros(n);
        for i in 0..n {
            let d = self.family.unit_deviance(self._y_data[i], mu[i]).max(0.0);
            let sign = if self._y_data[i] > mu[i] { 1.0 } else { -1.0 };
            resid[i] = sign * d.sqrt();
        }
        resid
    }

    /// Pearson residuals: (y - μ) / sqrt(V(μ)).
    pub fn pearson_residuals(&self) -> Array1<f64> {
        let mu = self.fitted_values();
        let n = self._y_data.len();
        let mut resid = Array1::<f64>::zeros(n);
        for i in 0..n {
            resid[i] = (self._y_data[i] - mu[i]) / self.family.variance(mu[i]).sqrt();
        }
        resid
    }

    /// Working residuals: (y - μ) * g'(μ).
    pub fn working_residuals(&self) -> Array1<f64> {
        let mu = self.fitted_values();
        let n = self._y_data.len();
        let mut resid = Array1::<f64>::zeros(n);
        for i in 0..n {
            resid[i] = (self._y_data[i] - mu[i]) * self.link.deriv(mu[i]);
        }
        resid
    }

    /// Compute confidence intervals at a custom significance level.
    pub fn conf_int(&self, alpha: f64) -> Vec<(f64, f64)> {
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal_dist.inverse_cdf(1.0 - alpha / 2.0);

        (0..self.params.len())
            .map(|i| {
                let margin = self.std_errors[i] * z_crit;
                (self.params[i] - margin, self.params[i] + margin)
            })
            .collect()
    }

    /// Prediction with standard errors and confidence intervals (on the mean response scale).
    ///
    /// Uses the delta method: SE(mu) = |dmu/deta| * SE(eta)
    pub fn get_prediction(&self, x_new: &Array2<f64>, alpha: f64) -> crate::ols::PredictionResult {
        let eta = x_new.dot(&self.params);
        let mu = eta.mapv(|e| self.link.linkinv(e));

        // SE on the linear predictor scale
        // Requires (X'WX)^-1 which we approximate from stored data
        let xw = {
            let n = self._x_data.nrows();
            let mut xw = self._x_data.clone();
            for i in 0..n {
                let mu_i = self.link.linkinv(self._x_data.row(i).dot(&self.params));
                let v = self.family.variance(mu_i);
                let g_prime = self.link.deriv(mu_i);
                let w = 1.0 / (v * g_prime * g_prime).max(1e-10);
                xw.row_mut(i).mapv_inplace(|x| x * w);
            }
            xw
        };
        let xtwx = self._x_data.t().dot(&xw);
        let inv_xtwx = xtwx
            .inv()
            .unwrap_or_else(|_| ndarray::Array2::eye(self.params.len()));

        let n_pred = x_new.nrows();
        let mut se_eta = ndarray::Array1::<f64>::zeros(n_pred);
        for i in 0..n_pred {
            let xi = x_new.row(i);
            let var_i = xi.dot(&inv_xtwx.dot(&xi)) * self.dispersion;
            se_eta[i] = var_i.max(0.0).sqrt();
        }

        // Transform to mean scale via delta method
        let se_mu: ndarray::Array1<f64> = (0..n_pred)
            .map(|i| {
                let dmu_deta = 1.0 / self.link.deriv(self.link.linkinv(eta[i])).abs().max(1e-10);
                se_eta[i] * dmu_deta
            })
            .collect();

        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal_dist.inverse_cdf(1.0 - alpha / 2.0);
        let margin = &se_mu * z_crit;
        let ci_lower = &mu - &margin;
        let ci_upper = &mu + &margin;

        crate::ols::PredictionResult {
            mean: mu,
            se: se_mu,
            ci_lower,
            ci_upper,
        }
    }

    /// Change inference type and recompute p-values and confidence intervals.
    pub fn with_inference(mut self, inference_type: InferenceType) -> Result<Self, GreenersError> {
        let (p_values, conf_lower, conf_upper) = crate::ols::OlsResult::compute_inference(
            &self.z_values,
            &self.std_errors,
            &self.params,
            self.df_resid,
            &inference_type,
        )?;
        self.p_values = p_values;
        self.conf_lower = conf_lower;
        self.conf_upper = conf_upper;
        self.inference_type = inference_type;
        Ok(self)
    }

    /// Model comparison statistics: (AIC, BIC, Log-Likelihood, Pseudo R²).
    pub fn model_stats(&self) -> (f64, f64, f64, f64) {
        (self.aic, self.bic, self.log_likelihood, self.pseudo_r2)
    }
}

impl fmt::Display for GlmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stat_label = match self.inference_type {
            InferenceType::StudentT => "t",
            InferenceType::Normal => "z",
        };

        writeln!(f, "\n{:=^78}", " Generalized Linear Model Results ")?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Family:",
            self.family.name(),
            "No. Observations:",
            self.n_obs
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Link:",
            self.link.name(),
            "Df Residuals:",
            self.df_resid
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Method:", "IRLS", "Df Model:", self.df_model
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "Deviance:", self.deviance, "Pearson chi2:", self.pearson_chi2
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "Log-Likelihood:", self.log_likelihood, "Pseudo R-sq:", self.pseudo_r2
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15}",
            "AIC:", self.aic, "Iterations:", self.n_iter
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "BIC:", self.bic, "Dispersion:", self.dispersion
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8}",
            "Variable", "coef", "std err", stat_label, "P>|z|", "[0.025", "0.975]"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let var_name = if let Some(ref names) = self.variable_names {
                if i < names.len() {
                    names[i].clone()
                } else {
                    format!("x{}", i)
                }
            } else {
                format!("x{}", i)
            };

            writeln!(
                f,
                "{:<12} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3}",
                var_name,
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

/// Generalized Linear Model estimator.
pub struct GLM;

impl GLM {
    /// Fit GLM from a formula and DataFrame using canonical link.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        family: Family,
        cov_type: CovarianceType,
    ) -> Result<GlmResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;

        let mut var_names = Vec::new();
        if formula.intercept {
            var_names.push("const".to_string());
        }
        for var in &formula.independents {
            var_names.push(var.clone());
        }

        let link = family.canonical_link();
        Self::fit_internal(&y, &x, family, link, cov_type, Some(var_names))
    }

    /// Fit GLM from arrays using canonical link.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        family: Family,
        cov_type: CovarianceType,
    ) -> Result<GlmResult, GreenersError> {
        let link = family.canonical_link();
        Self::fit_internal(y, x, family, link, cov_type, None)
    }

    /// Fit GLM from arrays with variable names using canonical link.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        family: Family,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
    ) -> Result<GlmResult, GreenersError> {
        let link = family.canonical_link();
        Self::fit_internal(y, x, family, link, cov_type, variable_names)
    }

    /// Fit GLM with an explicit (possibly non-canonical) link.
    pub fn fit_with_link(
        y: &Array1<f64>,
        x: &Array2<f64>,
        family: Family,
        link: Link,
        cov_type: CovarianceType,
    ) -> Result<GlmResult, GreenersError> {
        Self::fit_internal(y, x, family, link, cov_type, None)
    }

    fn fit_internal(
        y: &Array1<f64>,
        x: &Array2<f64>,
        family: Family,
        link: Link,
        cov_type: CovarianceType,
        variable_names: Option<Vec<String>>,
    ) -> Result<GlmResult, GreenersError> {
        let n = x.nrows();
        let _k = x.ncols();

        // Validate inputs
        if y.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "y and X have different number of observations".into(),
            ));
        }
        if y.iter().any(|v| !v.is_finite()) || x.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "Input data contains NaN or Inf values".into(),
            ));
        }

        // Detect collinearity
        let tolerance = 1e-10;
        let (x_clean, keep_indices, omit_indices) = OLS::detect_collinearity(x, tolerance);

        let mut omitted_var_names = Vec::new();
        let mut clean_var_names = Vec::new();
        if let Some(ref names) = variable_names {
            for &idx in &omit_indices {
                if idx < names.len() {
                    omitted_var_names.push(names[idx].clone());
                }
            }
            for &idx in &keep_indices {
                if idx < names.len() {
                    clean_var_names.push(names[idx].clone());
                }
            }
        }

        let x_use = &x_clean;
        let k = x_use.ncols();

        if n <= k {
            return Err(GreenersError::ShapeMismatch(
                "Degrees of freedom <= 0 after removing collinear variables".into(),
            ));
        }

        // IRLS
        let max_iter = 100;
        let tol = 1e-8;

        // Initialize μ
        let mut mu = family.starting_mu(y);
        let mut eta = mu.mapv(|m| link.link(m));
        let mut beta = Array1::<f64>::zeros(k);

        // Warm-start: solve initial WLS
        {
            let w_vec: Array1<f64> = (0..n)
                .map(|i| {
                    let v = family.variance(mu[i]);
                    let g_prime = link.deriv(mu[i]);
                    1.0 / (v * g_prime * g_prime).max(1e-10)
                })
                .collect();
            let z: Array1<f64> = (0..n)
                .map(|i| eta[i] + (y[i] - mu[i]) * link.deriv(mu[i]))
                .collect();

            // β = (X'WX)⁻¹ X'Wz
            let mut xw = x_use.clone();
            for (i, mut row) in xw.axis_iter_mut(Axis(0)).enumerate() {
                row *= w_vec[i];
            }
            let xtwx = x_use.t().dot(&xw);
            let xtwz = x_use.t().dot(&(&w_vec * &z));
            if let Ok(inv) = xtwx.inv() {
                beta = inv.dot(&xtwz);
            }
        }

        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            // η = Xβ
            eta = x_use.dot(&beta);
            // μ = g⁻¹(η)
            mu = eta.mapv(|e| link.linkinv(e));

            // IRLS weights: W_i = 1 / (V(μ_i) * (g'(μ_i))²)
            let w_vec: Array1<f64> = (0..n)
                .map(|i| {
                    let v = family.variance(mu[i]);
                    let g_prime = link.deriv(mu[i]);
                    1.0 / (v * g_prime * g_prime).max(1e-10)
                })
                .collect();

            // Working variable: z_i = η_i + (y_i - μ_i) * g'(μ_i)
            let z: Array1<f64> = (0..n)
                .map(|i| eta[i] + (y[i] - mu[i]) * link.deriv(mu[i]))
                .collect();

            // Weighted least squares: β_new = (X'WX)⁻¹ X'Wz
            let mut xw = x_use.clone();
            for (i, mut row) in xw.axis_iter_mut(Axis(0)).enumerate() {
                row *= w_vec[i];
            }
            let xtwx = x_use.t().dot(&xw);
            let xtwz = x_use.t().dot(&(&w_vec * &z));

            let inv_xtwx = match xtwx.inv() {
                Ok(m) => m,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };
            let beta_new = inv_xtwx.dot(&xtwz);

            let change = (&beta_new - &beta).mapv(|v| v.powi(2)).sum().sqrt();
            beta = beta_new;
            n_iter = iter + 1;

            if change < tol {
                converged = true;
                break;
            }
        }

        // Final mu/eta
        eta = x_use.dot(&beta);
        mu = eta.mapv(|e| link.linkinv(e));

        // Deviance
        let deviance: f64 = (0..n).map(|i| family.unit_deviance(y[i], mu[i])).sum();

        // Null deviance (intercept-only model)
        let y_mean = y.mean().unwrap_or(0.5);
        let mu_null = match family {
            Family::Binomial => y_mean.clamp(1e-10, 1.0 - 1e-10),
            _ => y_mean.max(1e-10),
        };
        let null_deviance: f64 = (0..n).map(|i| family.unit_deviance(y[i], mu_null)).sum();

        // Pearson chi-squared
        let pearson_chi2: f64 = (0..n)
            .map(|i| (y[i] - mu[i]).powi(2) / family.variance(mu[i]))
            .sum();

        // Dispersion
        let df_resid = n - k;
        let dispersion = if family.fixed_dispersion() {
            1.0
        } else {
            pearson_chi2 / df_resid as f64
        };

        // Log-likelihood
        let log_likelihood: f64 = (0..n)
            .map(|i| family.log_likelihood_obs(y[i], mu[i], dispersion))
            .sum();

        // Covariance matrix
        let w_vec: Array1<f64> = (0..n)
            .map(|i| {
                let v = family.variance(mu[i]);
                let g_prime = link.deriv(mu[i]);
                1.0 / (v * g_prime * g_prime).max(1e-10)
            })
            .collect();

        let mut xw = x_use.clone();
        for (i, mut row) in xw.axis_iter_mut(Axis(0)).enumerate() {
            row *= w_vec[i];
        }
        let xtwx = x_use.t().dot(&xw);
        let inv_xtwx = xtwx.inv()?;

        let cov_matrix = match &cov_type {
            CovarianceType::NonRobust => &inv_xtwx * dispersion,
            CovarianceType::HC1
            | CovarianceType::HC2
            | CovarianceType::HC3
            | CovarianceType::HC4 => {
                // Sandwich estimator: (X'WX)⁻¹ M (X'WX)⁻¹
                // where M = X' diag(pearson_resid²) X with HC adjustments
                let pearson_resid: Array1<f64> = (0..n)
                    .map(|i| (y[i] - mu[i]) / family.variance(mu[i]).sqrt())
                    .collect();

                let hat_values = if matches!(
                    cov_type,
                    CovarianceType::HC2 | CovarianceType::HC3 | CovarianceType::HC4
                ) {
                    // h_i = x_i' (X'WX)⁻¹ x_i * w_i
                    let mut h = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        let xi = x_use.row(i).to_owned();
                        h[i] = xi.dot(&inv_xtwx.dot(&xi)) * w_vec[i];
                    }
                    h
                } else {
                    Array1::<f64>::zeros(n)
                };

                let adj_resid2: Array1<f64> = (0..n)
                    .map(|i| {
                        let r2 = pearson_resid[i].powi(2) * family.variance(mu[i]);
                        match &cov_type {
                            CovarianceType::HC1 => r2 * n as f64 / df_resid as f64,
                            CovarianceType::HC2 => r2 / (1.0 - hat_values[i]).max(1e-10),
                            CovarianceType::HC3 => r2 / (1.0 - hat_values[i]).max(1e-10).powi(2),
                            CovarianceType::HC4 => {
                                let delta = (4.0_f64).min(n as f64 * hat_values[i] / k as f64);
                                r2 / (1.0 - hat_values[i]).max(1e-10).powf(delta)
                            }
                            _ => r2,
                        }
                    })
                    .collect();

                let mut meat = Array2::<f64>::zeros((k, k));
                for i in 0..n {
                    let xi = x_use.row(i).to_owned();
                    let g_prime = link.deriv(mu[i]);
                    let w_i = 1.0 / (family.variance(mu[i]) * g_prime * g_prime).max(1e-10);
                    let s = &xi * (adj_resid2[i] * w_i * w_i);
                    for j1 in 0..k {
                        for j2 in 0..k {
                            meat[[j1, j2]] += s[j1] * xi[j2];
                        }
                    }
                }

                inv_xtwx.dot(&meat).dot(&inv_xtwx)
            }
            _ => {
                // For NeweyWest, Clustered, etc. — fall back to non-robust for now
                &inv_xtwx * dispersion
            }
        };

        let std_errors = cov_matrix.diag().mapv(|v| v.max(0.0).sqrt());
        let z_values = &beta / &std_errors;

        let df_model = k.saturating_sub(1); // excluding intercept

        // Inference
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));
        let z_crit = normal_dist.inverse_cdf(0.975);
        let margin = &std_errors * z_crit;
        let conf_lower = &beta - &margin;
        let conf_upper = &beta + &margin;

        // AIC/BIC
        let k_f = k as f64;
        let n_f = n as f64;
        let aic = -2.0 * log_likelihood + 2.0 * k_f;
        let bic = -2.0 * log_likelihood + k_f * n_f.ln();

        // Pseudo R²
        let pseudo_r2 = if null_deviance.abs() > 1e-10 {
            1.0 - deviance / null_deviance
        } else {
            0.0
        };

        Ok(GlmResult {
            params: beta,
            std_errors,
            z_values,
            p_values,
            conf_lower,
            conf_upper,
            log_likelihood,
            deviance,
            null_deviance,
            aic,
            bic,
            pseudo_r2,
            pearson_chi2,
            dispersion,
            n_obs: n,
            df_resid,
            df_model,
            n_iter,
            converged,
            family,
            link,
            inference_type: InferenceType::Normal,
            variable_names: if !clean_var_names.is_empty() {
                Some(clean_var_names)
            } else {
                variable_names
            },
            omitted_vars: omitted_var_names,
            _x_data: x_use.clone(),
            _y_data: y.clone(),
        })
    }
}
