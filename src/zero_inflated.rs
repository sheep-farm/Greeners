use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::InferenceType;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result from Zero-Inflated models (ZIP/ZINB).
#[derive(Debug)]
pub struct ZeroInflatedResult {
    pub model_name: String,
    /// Count model coefficients (Poisson or NegBin part).
    pub count_params: Array1<f64>,
    pub count_std_errors: Array1<f64>,
    pub count_z_values: Array1<f64>,
    pub count_p_values: Array1<f64>,
    /// Inflate model coefficients (Logit part for P(excess zero)).
    pub inflate_params: Array1<f64>,
    pub inflate_std_errors: Array1<f64>,
    pub inflate_z_values: Array1<f64>,
    pub inflate_p_values: Array1<f64>,
    /// NegBin dispersion (None for ZIP).
    pub alpha: Option<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub iterations: usize,
    pub converged: bool,
    pub inference_type: InferenceType,
    pub count_var_names: Option<Vec<String>>,
    pub inflate_var_names: Option<Vec<String>>,
    _x_count: Array2<f64>,
    _x_inflate: Array2<f64>,
    _y_data: Array1<f64>,
}

impl fmt::Display for ZeroInflatedResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", format!(" {} Results ", self.model_name))?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "No. Observations:", self.n_obs, "Log-Likelihood:", self.log_likelihood
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Method:", "MLE", "AIC:", self.aic
        )?;

        if let Some(a) = self.alpha {
            writeln!(f, "{:<20} {:>15.4}", "Alpha (NB):", a)?;
        }

        // Count model
        writeln!(f, "\n{:-^78}", " Count Model ")?;
        writeln!(
            f,
            "{:<12} {:>10} {:>10} {:>8} {:>8}",
            "", "coef", "std err", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.count_params.len() {
            let name = self
                .count_var_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} {:>10.4} {:>10.4} {:>8.3} {:>8.3}",
                name,
                self.count_params[i],
                self.count_std_errors[i],
                self.count_z_values[i],
                self.count_p_values[i]
            )?;
        }

        // Inflate model
        writeln!(f, "\n{:-^78}", " Inflate Model (Logit) ")?;
        writeln!(
            f,
            "{:<12} {:>10} {:>10} {:>8} {:>8}",
            "", "coef", "std err", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.inflate_params.len() {
            let name = self
                .inflate_var_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("z{}", i));
            writeln!(
                f,
                "{:<12} {:>10.4} {:>10.4} {:>8.3} {:>8.3}",
                name,
                self.inflate_params[i],
                self.inflate_std_errors[i],
                self.inflate_z_values[i],
                self.inflate_p_values[i]
            )?;
        }

        writeln!(f, "{:=^78}", "")
    }
}

impl ZeroInflatedResult {
    /// Predicted expected counts: E[y] = (1 - π) * μ.
    pub fn predict_count(&self, x_count: &Array2<f64>, x_inflate: &Array2<f64>) -> Array1<f64> {
        let mu = x_count.dot(&self.count_params).mapv(f64::exp);
        let pi = x_inflate
            .dot(&self.inflate_params)
            .mapv(|v| 1.0 / (1.0 + (-v).exp()));
        (1.0 - &pi) * &mu
    }

    /// Predicted probability of observing zero.
    pub fn predict_proba_zero(
        &self,
        x_count: &Array2<f64>,
        x_inflate: &Array2<f64>,
    ) -> Array1<f64> {
        let mu = x_count.dot(&self.count_params).mapv(f64::exp);
        let pi = x_inflate
            .dot(&self.inflate_params)
            .mapv(|v| 1.0 / (1.0 + (-v).exp()));

        let n = mu.len();
        let mut p_zero = Array1::<f64>::zeros(n);
        for i in 0..n {
            let f0 = match self.alpha {
                Some(alpha) => {
                    let r = 1.0 / alpha;
                    (r / (r + mu[i])).powf(r)
                }
                None => (-mu[i]).exp(),
            };
            p_zero[i] = pi[i] + (1.0 - pi[i]) * f0;
        }
        p_zero
    }

    /// Model stats: (AIC, BIC, LogLik).
    pub fn model_stats(&self) -> (f64, f64, f64) {
        (self.aic, self.bic, self.log_likelihood)
    }
}

/// Zero-Inflated Poisson estimator.
pub struct ZIP;

impl ZIP {
    /// Fit ZIP model.
    /// `x_count`: design matrix for count component (with intercept).
    /// `x_inflate`: design matrix for inflate component (with intercept).
    /// If `x_inflate` is None, uses `x_count`.
    pub fn fit(
        y: &Array1<f64>,
        x_count: &Array2<f64>,
        x_inflate: Option<&Array2<f64>>,
    ) -> Result<ZeroInflatedResult, GreenersError> {
        let x_infl = x_inflate.unwrap_or(x_count);
        fit_zero_inflated(y, x_count, x_infl, false, None, None)
    }

    /// Fit with variable names.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x_count: &Array2<f64>,
        x_inflate: Option<&Array2<f64>>,
        count_names: Option<Vec<String>>,
        inflate_names: Option<Vec<String>>,
    ) -> Result<ZeroInflatedResult, GreenersError> {
        let x_infl = x_inflate.unwrap_or(x_count);
        fit_zero_inflated(y, x_count, x_infl, false, count_names, inflate_names)
    }
}

/// Zero-Inflated Negative Binomial estimator.
pub struct ZINB;

impl ZINB {
    /// Fit ZINB model.
    pub fn fit(
        y: &Array1<f64>,
        x_count: &Array2<f64>,
        x_inflate: Option<&Array2<f64>>,
    ) -> Result<ZeroInflatedResult, GreenersError> {
        let x_infl = x_inflate.unwrap_or(x_count);
        fit_zero_inflated(y, x_count, x_infl, true, None, None)
    }

    /// Fit with variable names.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x_count: &Array2<f64>,
        x_inflate: Option<&Array2<f64>>,
        count_names: Option<Vec<String>>,
        inflate_names: Option<Vec<String>>,
    ) -> Result<ZeroInflatedResult, GreenersError> {
        let x_infl = x_inflate.unwrap_or(x_count);
        fit_zero_inflated(y, x_count, x_infl, true, count_names, inflate_names)
    }
}

/// Internal fitting for ZIP/ZINB via EM algorithm.
fn fit_zero_inflated(
    y: &Array1<f64>,
    x_count: &Array2<f64>,
    x_inflate: &Array2<f64>,
    use_negbin: bool,
    count_names: Option<Vec<String>>,
    inflate_names: Option<Vec<String>>,
) -> Result<ZeroInflatedResult, GreenersError> {
    let n = y.len();
    let k_count = x_count.ncols();
    let k_inflate = x_inflate.ncols();

    if y.iter().any(|v| !v.is_finite()) || x_count.iter().any(|v| !v.is_finite()) {
        return Err(GreenersError::InvalidOperation(
            "Input data contains NaN or Inf values".into(),
        ));
    }

    // Initialize count params from Poisson MLE (ignoring zeros)
    let mut beta = Array1::<f64>::zeros(k_count);
    let y_mean = y.mean().unwrap_or(1.0).max(0.1);
    beta[0] = y_mean.ln(); // intercept ~ log(mean)

    // Initialize inflate params
    let mut gamma = Array1::<f64>::zeros(k_inflate);
    let zero_frac = y.iter().filter(|&&v| v < 0.5).count() as f64 / n as f64;
    let logit_zero = (zero_frac / (1.0 - zero_frac).max(1e-10)).ln();
    gamma[0] = logit_zero.clamp(-5.0, 5.0);

    let mut alpha = if use_negbin { 1.0 } else { 0.0 };

    let max_iter = 200;
    let tol = 1e-6;
    let mut converged = false;
    let mut iter = 0;
    let mut log_likelihood = f64::NEG_INFINITY;

    for iteration in 0..max_iter {
        iter = iteration + 1;

        // E-step: compute posterior probability of being from inflate component
        let eta_count = x_count.dot(&beta);
        let mu: Array1<f64> = eta_count.mapv(f64::exp);
        let eta_inflate = x_inflate.dot(&gamma);
        let pi: Array1<f64> = eta_inflate.mapv(|v| 1.0 / (1.0 + (-v).exp()));

        let mut w = Array1::<f64>::zeros(n); // posterior P(inflate | y_i=0)
        log_likelihood = 0.0;

        for i in 0..n {
            let p_i = pi[i].clamp(1e-10, 1.0 - 1e-10);
            let mu_i = mu[i].max(1e-10);

            if y[i] < 0.5 {
                // y_i = 0
                let f0 = if use_negbin {
                    let r = (1.0_f64 / alpha).max(1e-6);
                    (r / (r + mu_i)).powf(r)
                } else {
                    (-mu_i).exp()
                };
                let lik = p_i + (1.0 - p_i) * f0;
                w[i] = p_i / lik.max(1e-15);
                log_likelihood += lik.max(1e-15).ln();
            } else {
                // y_i > 0
                w[i] = 0.0;
                let f_y = if use_negbin {
                    let r = (1.0_f64 / alpha).max(1e-6);
                    let yi = y[i];
                    // NB PMF: Γ(r+y)/(Γ(r)*y!) * (r/(r+μ))^r * (μ/(r+μ))^y
                    let log_f = lgamma(r + yi) - lgamma(r) - lgamma(yi + 1.0)
                        + r * (r / (r + mu_i)).ln()
                        + yi * (mu_i / (r + mu_i)).ln();
                    log_f.exp()
                } else {
                    // Poisson PMF
                    let yi = y[i];
                    (yi * mu_i.ln() - mu_i - lgamma(yi + 1.0)).exp()
                };
                log_likelihood += ((1.0 - p_i) * f_y).max(1e-300).ln();
            }
        }

        // M-step: update gamma (inflate) via Newton-Raphson on logistic regression
        // weighted by appropriate posterior
        for _ in 0..5 {
            let eta_inf = x_inflate.dot(&gamma);
            let pi_new: Array1<f64> = eta_inf.mapv(|v| 1.0 / (1.0 + (-v).exp()));

            let mut grad_gamma = Array1::<f64>::zeros(k_inflate);
            let mut hess_gamma = Array2::<f64>::zeros((k_inflate, k_inflate));

            for i in 0..n {
                let p_i = pi_new[i].clamp(1e-10, 1.0 - 1e-10);
                let target = w[i]; // posterior prob of inflate
                let diff = target - p_i;

                for kk in 0..k_inflate {
                    grad_gamma[kk] += diff * x_inflate[[i, kk]];
                }

                let w_ii = p_i * (1.0 - p_i);
                for kk in 0..k_inflate {
                    for ll in 0..k_inflate {
                        hess_gamma[[kk, ll]] -= w_ii * x_inflate[[i, kk]] * x_inflate[[i, ll]];
                    }
                }
            }

            if let Ok(inv_h) = (-&hess_gamma).inv() {
                let step = inv_h.dot(&grad_gamma);
                gamma = &gamma + &step;
                // Clamp to prevent overflow
                gamma.mapv_inplace(|v| v.clamp(-10.0, 10.0));
            }
        }

        // M-step: update beta (count) via weighted Poisson Newton-Raphson
        // Weight: (1 - w_i)
        for _ in 0..5 {
            let eta_c = x_count.dot(&beta);
            let mu_new: Array1<f64> = eta_c.mapv(f64::exp);

            let mut grad_beta = Array1::<f64>::zeros(k_count);
            let mut hess_beta = Array2::<f64>::zeros((k_count, k_count));

            for i in 0..n {
                let wt = 1.0 - w[i];
                if wt < 1e-10 {
                    continue;
                }
                let mu_i = mu_new[i].max(1e-10);
                let resid = y[i] - mu_i;

                for kk in 0..k_count {
                    grad_beta[kk] += wt * resid * x_count[[i, kk]];
                }

                for kk in 0..k_count {
                    for ll in 0..k_count {
                        hess_beta[[kk, ll]] -= wt * mu_i * x_count[[i, kk]] * x_count[[i, ll]];
                    }
                }
            }

            if let Ok(inv_h) = (-&hess_beta).inv() {
                let step = inv_h.dot(&grad_beta);
                beta = &beta + &step;
                beta.mapv_inplace(|v| v.clamp(-20.0, 20.0));
            }
        }

        // M-step: update alpha if NegBin (method of moments)
        if use_negbin {
            let mu_new: Array1<f64> = x_count.dot(&beta).mapv(f64::exp);
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n {
                let wt = 1.0 - w[i];
                if wt < 1e-10 {
                    continue;
                }
                let m = mu_new[i].max(1e-10);
                num += wt * ((y[i] - m).powi(2) - y[i]) / (m * m);
                den += wt;
            }
            alpha = (num / den.max(1.0)).max(0.01);
        }

        // Check convergence via log-likelihood change
        if iteration > 0 {
            let ll_old = log_likelihood;
            // Recompute LL
            let mu_final = x_count.dot(&beta).mapv(f64::exp);
            let pi_final = x_inflate.dot(&gamma).mapv(|v| 1.0 / (1.0 + (-v).exp()));
            let mut ll_new = 0.0;
            for i in 0..n {
                let p_i = pi_final[i].clamp(1e-10, 1.0 - 1e-10);
                let mu_i = mu_final[i].max(1e-10);
                if y[i] < 0.5 {
                    let f0 = if use_negbin {
                        let r = (1.0_f64 / alpha).max(1e-6);
                        (r / (r + mu_i)).powf(r)
                    } else {
                        (-mu_i).exp()
                    };
                    ll_new += (p_i + (1.0 - p_i) * f0).max(1e-15).ln();
                } else {
                    let f_y = if use_negbin {
                        let r = (1.0_f64 / alpha).max(1e-6);
                        let yi = y[i];
                        let log_f = lgamma(r + yi) - lgamma(r) - lgamma(yi + 1.0)
                            + r * (r / (r + mu_i)).ln()
                            + yi * (mu_i / (r + mu_i)).ln();
                        log_f.exp()
                    } else {
                        let yi = y[i];
                        (yi * mu_i.ln() - mu_i - lgamma(yi + 1.0)).exp()
                    };
                    ll_new += ((1.0 - p_i) * f_y).max(1e-300).ln();
                }
            }

            if (ll_new - ll_old).abs() < tol {
                log_likelihood = ll_new;
                converged = true;
                break;
            }
            log_likelihood = ll_new;
        }
    }

    // Compute standard errors via numerical Hessian of full log-likelihood
    let total_params = k_count + k_inflate;
    let mut full_theta = Array1::<f64>::zeros(total_params);
    full_theta.slice_mut(ndarray::s![..k_count]).assign(&beta);
    full_theta.slice_mut(ndarray::s![k_count..]).assign(&gamma);

    let h = 1e-4;
    let mut hessian = Array2::<f64>::zeros((total_params, total_params));

    let _ll_center = compute_zi_ll(
        y,
        x_count,
        x_inflate,
        &full_theta,
        k_count,
        use_negbin,
        alpha,
    );

    for a in 0..total_params {
        for b in a..total_params {
            let mut t_pp = full_theta.clone();
            let mut t_pm = full_theta.clone();
            let mut t_mp = full_theta.clone();
            let mut t_mm = full_theta.clone();

            t_pp[a] += h;
            t_pp[b] += h;
            t_pm[a] += h;
            t_pm[b] -= h;
            t_mp[a] -= h;
            t_mp[b] += h;
            t_mm[a] -= h;
            t_mm[b] -= h;

            let ll_pp = compute_zi_ll(y, x_count, x_inflate, &t_pp, k_count, use_negbin, alpha);
            let ll_pm = compute_zi_ll(y, x_count, x_inflate, &t_pm, k_count, use_negbin, alpha);
            let ll_mp = compute_zi_ll(y, x_count, x_inflate, &t_mp, k_count, use_negbin, alpha);
            let ll_mm = compute_zi_ll(y, x_count, x_inflate, &t_mm, k_count, use_negbin, alpha);

            let d2 = (ll_pp - ll_pm - ll_mp + ll_mm) / (4.0 * h * h);
            hessian[[a, b]] = d2;
            hessian[[b, a]] = d2;
        }
    }

    let cov_matrix = (-&hessian)
        .inv()
        .unwrap_or_else(|_| Array2::eye(total_params) * 1e-4);

    let normal_dist = Normal::new(0.0, 1.0).unwrap();

    let count_se: Array1<f64> = (0..k_count)
        .map(|i| cov_matrix[[i, i]].max(0.0).sqrt())
        .collect();
    let count_z = &beta / count_se.mapv(|s| if s > 1e-15 { s } else { 1.0 });
    let count_p = count_z.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));

    let inflate_se: Array1<f64> = (0..k_inflate)
        .map(|i| cov_matrix[[k_count + i, k_count + i]].max(0.0).sqrt())
        .collect();
    let inflate_z = &gamma / inflate_se.mapv(|s| if s > 1e-15 { s } else { 1.0 });
    let inflate_p = inflate_z.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));

    let k_total = total_params as f64;
    let aic = -2.0 * log_likelihood + 2.0 * k_total;
    let bic = -2.0 * log_likelihood + k_total * (n as f64).ln();

    let model_name = if use_negbin {
        "Zero-Inflated Negative Binomial (ZINB)"
    } else {
        "Zero-Inflated Poisson (ZIP)"
    };

    Ok(ZeroInflatedResult {
        model_name: model_name.to_string(),
        count_params: beta,
        count_std_errors: count_se,
        count_z_values: count_z,
        count_p_values: count_p,
        inflate_params: gamma,
        inflate_std_errors: inflate_se,
        inflate_z_values: inflate_z,
        inflate_p_values: inflate_p,
        alpha: if use_negbin { Some(alpha) } else { None },
        log_likelihood,
        aic,
        bic,
        n_obs: n,
        iterations: iter,
        converged,
        inference_type: InferenceType::Normal,
        count_var_names: count_names,
        inflate_var_names: inflate_names,
        _x_count: x_count.to_owned(),
        _x_inflate: x_inflate.to_owned(),
        _y_data: y.to_owned(),
    })
}

/// Compute ZI log-likelihood for a given parameter vector.
fn compute_zi_ll(
    y: &Array1<f64>,
    x_count: &Array2<f64>,
    x_inflate: &Array2<f64>,
    theta: &Array1<f64>,
    k_count: usize,
    use_negbin: bool,
    alpha: f64,
) -> f64 {
    let n = y.len();
    let beta = theta.slice(ndarray::s![..k_count]);
    let gamma = theta.slice(ndarray::s![k_count..]);

    let mu = x_count.dot(&beta).mapv(f64::exp);
    let pi = x_inflate.dot(&gamma).mapv(|v| 1.0 / (1.0 + (-v).exp()));

    let mut ll = 0.0;
    for i in 0..n {
        let p_i = pi[i].clamp(1e-10, 1.0 - 1e-10);
        let mu_i = mu[i].max(1e-10);

        if y[i] < 0.5 {
            let f0 = if use_negbin {
                let r = (1.0_f64 / alpha).max(1e-6);
                (r / (r + mu_i)).powf(r)
            } else {
                (-mu_i).exp()
            };
            ll += (p_i + (1.0 - p_i) * f0).max(1e-15).ln();
        } else {
            let f_y = if use_negbin {
                let r = (1.0_f64 / alpha).max(1e-6);
                let yi = y[i];
                let log_f = lgamma(r + yi) - lgamma(r) - lgamma(yi + 1.0)
                    + r * (r / (r + mu_i)).ln()
                    + yi * (mu_i / (r + mu_i)).ln();
                log_f.exp()
            } else {
                let yi = y[i];
                (yi * mu_i.ln() - mu_i - lgamma(yi + 1.0)).exp()
            };
            ll += ((1.0 - p_i) * f_y).max(1e-300).ln();
        }
    }
    ll
}

/// Log-gamma function using Lanczos approximation.
#[allow(clippy::excessive_precision)]
fn lgamma(x: f64) -> f64 {
    // Use the Lanczos approximation
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Reflection formula
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().abs().ln()
            - lgamma(1.0 - x);
    }

    let x = x - 1.0;
    let g = 7.0;
    let c = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_08,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let mut sum = c[0];
    for (i, &coeff) in c.iter().enumerate().skip(1) {
        sum += coeff / (x + i as f64);
    }

    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
}
