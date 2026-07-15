//! MIDAS (Mixed Data Sampling) regression.
//!
//! Regresses a low-frequency variable (e.g., quarterly GDP) on
//! high-frequency variables (e.g., monthly indicators) without
//! aggregating the high-frequency data. Uses Almon polynomial
//! weighting scheme to avoid parameter proliferation.
//!
//! Model: y_t = α + β₀ Σ_{k=0}^{p-1} w(k, γ) x_{t-k/h} + ε_t
//!
//! where w(k, γ) are normalized Almon weights:
//!   w(k, γ) = exp(γ₁k + γ₂k² + ...) / Σ exp(γ₁j + γ₂j² + ...)
//!
//! Estimation: NLS via grid search + Newton refinement on γ.

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{array, Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of MIDAS regression.
#[derive(Debug)]
pub struct MidasResult {
    /// Intercept
    pub alpha: f64,
    /// Beta coefficient (overall scale of high-freq effect)
    pub beta: f64,
    /// Almon polynomial parameters (gamma)
    pub gamma: Array1<f64>,
    /// Standard error of alpha
    pub alpha_se: f64,
    /// Standard error of beta
    pub beta_se: f64,
    /// t-statistic of beta
    pub beta_t: f64,
    /// p-value of beta
    pub beta_p: f64,
    /// R-squared
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Number of low-frequency observations
    pub n_obs: usize,
    /// Number of high-frequency lags
    pub n_lags: usize,
    /// Frequency ratio (e.g., 3 for monthly→quarterly)
    pub freq_ratio: usize,
    /// Degree of Almon polynomial
    pub poly_degree: usize,
    /// Implied weights
    pub weights: Array1<f64>,
}

impl fmt::Display for MidasResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " MIDAS Regression ")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "High-freq lags:", self.n_lags)?;
        writeln!(f, "{:<20} {:>12}", "Frequency ratio:", self.freq_ratio)?;
        writeln!(f, "{:<20} {:>12}", "Almon poly degree:", self.poly_degree)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "Adj. R-squared:", self.adj_r_squared)?;
        writeln!(f, "{:<20} {:>12.6}", "Log-likelihood:", self.log_likelihood)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Parameter", "Coef.", "Std.Err.", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            "alpha",
            self.alpha,
            self.alpha_se,
            self.alpha / self.alpha_se.max(1e-10),
            0.0
        )?;
        writeln!(
            f,
            "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            "beta", self.beta, self.beta_se, self.beta_t, self.beta_p
        )?;
        for i in 0..self.gamma.len() {
            writeln!(f, "{:<12} {:>12.6}", format!("gamma_{i}"), self.gamma[i])?;
        }

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Implied Almon weights:")?;
        for i in 0..self.weights.len() {
            let _ = writeln!(f, "  w({i:>2}) = {:.6}", self.weights[i]);
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct Midas;

impl Midas {
    /// Estimate MIDAS regression with Almon polynomial weighting.
    ///
    /// # Arguments
    /// * `y` - Low-frequency dependent variable (n_low)
    /// * `x` - High-frequency regressor (n_high = n_low * freq_ratio)
    /// * `freq_ratio` - Frequency ratio (e.g., 3 for monthly→quarterly)
    /// * `n_lags` - Number of high-frequency lags to include
    /// * `poly_degree` - Degree of Almon polynomial (1=linear, 2=quadratic)
    pub fn fit(
        y: &Array1<f64>,
        x: &Array1<f64>,
        freq_ratio: usize,
        n_lags: usize,
        poly_degree: usize,
    ) -> Result<MidasResult, GreenersError> {
        let n_low = y.len();
        let n_high = x.len();
        if n_high < n_low * freq_ratio {
            return Err(GreenersError::ShapeMismatch(format!(
                "MIDAS: x has {n_high} obs, need at least {} for {n_low} low-freq obs with ratio {freq_ratio}",
                n_low * freq_ratio
            )));
        }
        if freq_ratio < 1 || n_lags < 1 || poly_degree < 1 {
            return Err(GreenersError::InvalidOperation(
                "MIDAS: freq_ratio, n_lags, poly_degree must be >= 1".into(),
            ));
        }

        // Build the MIDAS regressor: for each low-freq period t,
        // x_midas_t = Σ_{k=0}^{n_lags-1} w(k, γ) * x_{t*freq_ratio - k}
        // We optimize over γ via grid search + Newton.

        let mut best_gamma = Array1::zeros(poly_degree);
        let mut best_sse = f64::INFINITY;
        let mut best_alpha = 0.0_f64;
        let mut best_beta = 0.0_f64;

        // Grid search over gamma
        let n_grid = if poly_degree == 1 { 21 } else { 11 };
        let grid_range = 2.0; // search in [-2, 2]

        if poly_degree == 1 {
            for i in 0..n_grid {
                let g0 = -grid_range + 2.0 * grid_range * i as f64 / (n_grid - 1) as f64;
                let gamma = array![g0];
                let (alpha, beta, sse) = Self::estimate_alm(y, x, freq_ratio, n_lags, &gamma)?;
                if sse < best_sse {
                    best_sse = sse;
                    best_gamma = gamma;
                    best_alpha = alpha;
                    best_beta = beta;
                }
            }
        } else {
            for i in 0..n_grid {
                let g0 = -grid_range + 2.0 * grid_range * i as f64 / (n_grid - 1) as f64;
                for j in 0..n_grid {
                    let g1 = -grid_range + 2.0 * grid_range * j as f64 / (n_grid - 1) as f64;
                    let gamma = array![g0, g1];
                    let (alpha, beta, sse) = Self::estimate_alm(y, x, freq_ratio, n_lags, &gamma)?;
                    if sse < best_sse {
                        best_sse = sse;
                        best_gamma = gamma;
                        best_alpha = alpha;
                        best_beta = beta;
                    }
                }
            }
        }

        // Newton refinement
        let h = 0.01;
        for _ in 0..20 {
            let (_a0, _b0, sse0) = Self::estimate_alm(y, x, freq_ratio, n_lags, &best_gamma)?;
            let mut grad: Array1<f64> = Array1::zeros(poly_degree);
            let mut hess: Array2<f64> = Array2::zeros((poly_degree, poly_degree));
            for p in 0..poly_degree {
                let mut g_p = best_gamma.clone();
                g_p[p] += h;
                let (_, _, sse_p) = Self::estimate_alm(y, x, freq_ratio, n_lags, &g_p)?;
                let mut g_m = best_gamma.clone();
                g_m[p] -= h;
                let (_, _, sse_m) = Self::estimate_alm(y, x, freq_ratio, n_lags, &g_m)?;
                grad[p] = (sse_p - sse_m) / (2.0 * h);
                hess[(p, p)] = (sse_p - 2.0 * sse0 + sse_m) / (h * h);
            }
            if poly_degree > 1 {
                for p in 0..poly_degree {
                    for q in (p + 1)..poly_degree {
                        let mut g_pp = best_gamma.clone();
                        g_pp[p] += h;
                        g_pp[q] += h;
                        let (_, _, s1) = Self::estimate_alm(y, x, freq_ratio, n_lags, &g_pp)?;
                        let mut g_pm = best_gamma.clone();
                        g_pm[p] += h;
                        g_pm[q] -= h;
                        let (_, _, s2) = Self::estimate_alm(y, x, freq_ratio, n_lags, &g_pm)?;
                        let mut g_mp = best_gamma.clone();
                        g_mp[p] -= h;
                        g_mp[q] += h;
                        let (_, _, s3) = Self::estimate_alm(y, x, freq_ratio, n_lags, &g_mp)?;
                        let mut g_mm = best_gamma.clone();
                        g_mm[p] -= h;
                        g_mm[q] -= h;
                        let (_, _, s4) = Self::estimate_alm(y, x, freq_ratio, n_lags, &g_mm)?;
                        hess[(p, q)] = (s1 - s2 - s3 + s4) / (4.0 * h * h);
                        hess[(q, p)] = hess[(p, q)];
                    }
                }
            }
            let hess_reg = &hess + Array2::eye(poly_degree) * 1e-8;
            let hess_inv = hess_reg.inv()?;
            let delta = hess_inv.dot(&grad);
            let new_gamma = &best_gamma - &delta;
            let (_, _, new_sse) = Self::estimate_alm(y, x, freq_ratio, n_lags, &new_gamma)?;
            if new_sse < sse0 {
                best_gamma = new_gamma;
                best_sse = new_sse;
                let (a, b, _) = Self::estimate_alm(y, x, freq_ratio, n_lags, &best_gamma)?;
                best_alpha = a;
                best_beta = b;
            }
            if delta.mapv(|v| v.abs()).sum() < 1e-8 {
                break;
            }
        }

        // Compute weights
        let weights = Self::alm_weights(n_lags, &best_gamma);

        // R-squared
        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let r_squared = if tss > 1e-15 {
            1.0 - best_sse / tss
        } else {
            0.0
        };
        let n_params = 2 + poly_degree;
        let adj_r2 = 1.0 - (1.0 - r_squared) * (n_low - 1) as f64 / (n_low - n_params) as f64;

        // SE of beta (from SSE)
        let sigma2 = best_sse / (n_low - n_params) as f64;
        let beta_se = (sigma2 / (best_beta * best_beta).max(1e-10)).sqrt() * best_beta.abs();
        let alpha_se = (sigma2 / n_low as f64).sqrt();
        let beta_t = best_beta / beta_se.max(1e-10);
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let beta_p = 2.0 * (1.0 - normal.cdf(beta_t.abs()));

        let log_likelihood = -(n_low as f64) / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln()
            - best_sse / (2.0 * sigma2);

        Ok(MidasResult {
            alpha: best_alpha,
            beta: best_beta,
            gamma: best_gamma,
            alpha_se,
            beta_se,
            beta_t,
            beta_p,
            r_squared,
            adj_r_squared: adj_r2,
            log_likelihood,
            n_obs: n_low,
            n_lags,
            freq_ratio,
            poly_degree,
            weights,
        })
    }

    /// Compute Almon weights for given gamma.
    fn alm_weights(n_lags: usize, gamma: &Array1<f64>) -> Array1<f64> {
        let mut raw = Array1::zeros(n_lags);
        for k in 0..n_lags {
            let mut val = 0.0;
            for (p, g) in gamma.iter().enumerate() {
                val += g * (k as f64).powi((p + 1) as i32);
            }
            raw[k] = val.exp();
        }
        let sum = raw.sum().max(1e-10);
        raw / sum
    }

    /// Estimate alpha, beta given gamma (concentrated least squares).
    fn estimate_alm(
        y: &Array1<f64>,
        x: &Array1<f64>,
        freq_ratio: usize,
        n_lags: usize,
        gamma: &Array1<f64>,
    ) -> Result<(f64, f64, f64), GreenersError> {
        let n_low = y.len();
        let weights = Self::alm_weights(n_lags, gamma);

        // Build weighted high-freq regressor
        let mut x_midas = Array1::zeros(n_low);
        for t in 0..n_low {
            let mut val = 0.0;
            for k in 0..n_lags {
                let base = t * freq_ratio + (freq_ratio - 1);
                if base >= k && base - k < x.len() {
                    val += weights[k] * x[base - k];
                }
            }
            x_midas[t] = val;
        }

        // OLS: y = alpha + beta * x_midas
        let _n = n_low as f64;
        let x_mean = x_midas.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);
        let sxx = x_midas.mapv(|v| (v - x_mean).powi(2)).sum();
        let sxy = (&x_midas - x_mean).dot(&(y - y_mean));
        if sxx.abs() < 1e-15 {
            return Ok((y_mean, 0.0, y.mapv(|v| (v - y_mean).powi(2)).sum()));
        }
        let beta = sxy / sxx;
        let alpha = y_mean - beta * x_mean;
        let residuals = y - alpha - beta * &x_midas;
        let sse = residuals.dot(&residuals);
        Ok((alpha, beta, sse))
    }
}
