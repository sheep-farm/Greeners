use crate::error::GreenersError;
use crate::{DataFrame, Formula, InferenceType};
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::fmt;

/// Result from Ordered Logit/Probit regression.
#[derive(Debug)]
pub struct OrderedResult {
    pub model_name: String,
    /// Slope coefficients (k, no intercept — absorbed by cutpoints).
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    /// Cutpoints (thresholds): α_1, ..., α_{J-1}.
    pub thresholds: Vec<f64>,
    pub threshold_std_errors: Vec<f64>,
    pub log_likelihood: f64,
    pub pseudo_r2: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub n_categories: usize,
    pub iterations: usize,
    pub converged: bool,
    pub category_labels: Vec<f64>,
    pub inference_type: InferenceType,
    pub variable_names: Option<Vec<String>>,
    _x_data: Array2<f64>,
}

impl fmt::Display for OrderedResult {
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
            "No. Categories:", self.n_categories, "Pseudo R-sq:", self.pseudo_r2
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Method:", "MLE", "AIC:", self.aic
        )?;

        writeln!(f, "\n{:-^78}", " Coefficients ")?;
        writeln!(
            f,
            "{:<12} {:>10} {:>10} {:>8} {:>8}",
            "", "coef", "std err", "z", "P>|z|"
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
                "{:<12} {:>10.4} {:>10.4} {:>8.3} {:>8.3}",
                name, self.params[i], self.std_errors[i], self.z_values[i], self.p_values[i]
            )?;
        }

        writeln!(f, "\n{:-^78}", " Thresholds ")?;
        for (i, (&t, &se)) in self
            .thresholds
            .iter()
            .zip(self.threshold_std_errors.iter())
            .enumerate()
        {
            writeln!(f, "  cut{:<8} {:>10.4} {:>10.4}", i + 1, t, se)?;
        }

        writeln!(f, "{:=^78}", "")
    }
}

impl OrderedResult {
    /// Predicted probabilities for each category: (n x J).
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let j = self.n_categories;
        let mut probs = Array2::<f64>::zeros((n, j));

        let is_logit = self.model_name.contains("Logit");

        for i in 0..n {
            let xb = x.row(i).dot(&self.params);

            for c in 0..j {
                let p_le = if c < j - 1 {
                    let z = self.thresholds[c] - xb;
                    if is_logit {
                        1.0 / (1.0 + (-z).exp())
                    } else {
                        let normal = Normal::new(0.0, 1.0).unwrap();
                        normal.cdf(z)
                    }
                } else {
                    1.0
                };

                let p_le_prev = if c > 0 {
                    let z = self.thresholds[c - 1] - xb;
                    if is_logit {
                        1.0 / (1.0 + (-z).exp())
                    } else {
                        let normal = Normal::new(0.0, 1.0).unwrap();
                        normal.cdf(z)
                    }
                } else {
                    0.0
                };

                probs[[i, c]] = (p_le - p_le_prev).max(1e-15);
            }
        }

        probs
    }

    /// Predicted category (most likely).
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let probs = self.predict_proba(x);
        let n = probs.nrows();
        let mut preds = Array1::<f64>::zeros(n);
        for i in 0..n {
            let row = probs.row(i);
            let mut max_idx = 0;
            let mut max_val = row[0];
            for (c, &v) in row.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = c;
                }
            }
            preds[i] = self.category_labels[max_idx];
        }
        preds
    }

    /// Model stats: (AIC, BIC, LogLik, PseudoR2).
    pub fn model_stats(&self) -> (f64, f64, f64, f64) {
        (self.aic, self.bic, self.log_likelihood, self.pseudo_r2)
    }
}

/// CDF and PDF helpers
fn logistic_cdf(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn logistic_pdf(z: f64) -> f64 {
    let e = (-z).exp();
    e / (1.0 + e).powi(2)
}

fn normal_cdf(z: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.cdf(z)
}

fn normal_pdf(z: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.pdf(z)
}

/// Ordered Logit estimator.
pub struct OrderedLogit;

/// Ordered Probit estimator.
pub struct OrderedProbit;

impl OrderedLogit {
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
    ) -> Result<OrderedResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        // For ordered models, remove intercept column — absorbed by cutpoints
        let x_no_const = if formula.intercept {
            x.slice(ndarray::s![.., 1..]).to_owned()
        } else {
            x
        };
        let var_names: Vec<String> = if formula.intercept {
            formula.independents.clone()
        } else {
            let mut v = vec!["const".to_string()];
            v.extend(formula.independents.clone());
            v
        };
        fit_ordered(&y, &x_no_const, true, Some(var_names))
    }

    pub fn fit(y: &Array1<f64>, x: &Array2<f64>) -> Result<OrderedResult, GreenersError> {
        fit_ordered(y, x, true, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<OrderedResult, GreenersError> {
        fit_ordered(y, x, true, variable_names)
    }
}

impl OrderedProbit {
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
    ) -> Result<OrderedResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        let x_no_const = if formula.intercept {
            x.slice(ndarray::s![.., 1..]).to_owned()
        } else {
            x
        };
        let var_names: Vec<String> = if formula.intercept {
            formula.independents.clone()
        } else {
            let mut v = vec!["const".to_string()];
            v.extend(formula.independents.clone());
            v
        };
        fit_ordered(&y, &x_no_const, false, Some(var_names))
    }

    pub fn fit(y: &Array1<f64>, x: &Array2<f64>) -> Result<OrderedResult, GreenersError> {
        fit_ordered(y, x, false, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<OrderedResult, GreenersError> {
        fit_ordered(y, x, false, variable_names)
    }
}

/// Internal ordered model fitting.
/// x should NOT contain an intercept column (cutpoints play that role).
fn fit_ordered(
    y: &Array1<f64>,
    x: &Array2<f64>,
    is_logit: bool,
    variable_names: Option<Vec<String>>,
) -> Result<OrderedResult, GreenersError> {
    let n = x.nrows();
    let k = x.ncols();

    if y.iter().any(|v| !v.is_finite()) || x.iter().any(|v| !v.is_finite()) {
        return Err(GreenersError::InvalidOperation(
            "Input data contains NaN or Inf values".into(),
        ));
    }

    // Sort categories
    let mut categories: Vec<f64> = y.iter().copied().collect();
    categories.sort_by(|a, b| a.partial_cmp(b).unwrap());
    categories.dedup();
    let j = categories.len();

    if j < 3 {
        return Err(GreenersError::InvalidOperation(
            "Ordered model requires at least 3 categories".into(),
        ));
    }

    let j_minus_1 = j - 1;
    let y_idx: Vec<usize> = y
        .iter()
        .map(|val| {
            categories
                .iter()
                .position(|c| (c - val).abs() < 1e-10)
                .unwrap_or(0)
        })
        .collect();

    let cdf_fn = if is_logit { logistic_cdf } else { normal_cdf };
    let pdf_fn = if is_logit { logistic_pdf } else { normal_pdf };

    // Total parameter count: k (slopes) + J-1 (cutpoints)
    // Reparametrize: theta = [β_1, ..., β_k, α_1, δ_2, ..., δ_{J-1}]
    // where α_j = α_1 + Σ_{m=2}^{j} exp(δ_m) for j >= 2
    let total_params = k + j_minus_1;
    let mut theta = Array1::<f64>::zeros(total_params);

    // Initialize cutpoints: spread evenly
    theta[k] = 0.0; // α_1
    for m in 1..j_minus_1 {
        theta[k + m] = 0.0; // δ_m ~ log(spacing), start at 0 => spacing = 1
    }

    let max_iter = 200;
    let tol = 1e-6;
    let mut converged = false;
    let mut iter = 0;
    let mut log_likelihood = f64::NEG_INFINITY;

    for iteration in 0..max_iter {
        iter = iteration + 1;

        // Extract beta and cutpoints from theta
        let beta = theta.slice(ndarray::s![..k]).to_owned();
        let mut alphas = vec![0.0; j_minus_1];
        alphas[0] = theta[k];
        for m in 1..j_minus_1 {
            alphas[m] = alphas[m - 1] + theta[k + m].exp();
        }

        // Compute log-likelihood, gradient, and Hessian
        log_likelihood = 0.0;
        let mut gradient = Array1::<f64>::zeros(total_params);
        let mut hessian = Array2::<f64>::zeros((total_params, total_params));

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let x_i = x.row(i);
            let xb: f64 = x_i.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
            let c = y_idx[i];

            // P(y = c) = F(α_c - xβ) - F(α_{c-1} - xβ)
            let z_upper = if c < j_minus_1 {
                alphas[c] - xb
            } else {
                f64::INFINITY
            };
            let z_lower = if c > 0 {
                alphas[c - 1] - xb
            } else {
                f64::NEG_INFINITY
            };

            let f_upper = if z_upper.is_finite() {
                cdf_fn(z_upper)
            } else {
                1.0
            };
            let f_lower = if z_lower.is_finite() {
                cdf_fn(z_lower)
            } else {
                0.0
            };

            let p = (f_upper - f_lower).max(1e-15);
            log_likelihood += p.ln();

            let pdf_upper = if z_upper.is_finite() {
                pdf_fn(z_upper)
            } else {
                0.0
            };
            let pdf_lower = if z_lower.is_finite() {
                pdf_fn(z_lower)
            } else {
                0.0
            };

            let dp_dbeta_factor = -(pdf_upper - pdf_lower) / p;

            // Gradient w.r.t. β
            for kk in 0..k {
                gradient[kk] += dp_dbeta_factor * x_i[kk];
            }

            // Gradient w.r.t. cutpoints (via chain rule with reparametrization)
            // d/d(alpha_m) of P: +pdf(alpha_m - xb) if c==m, -pdf(alpha_m - xb) if c==m+1
            for m in 0..j_minus_1 {
                let d_p_d_alpha_m = if c == m {
                    pdf_upper / p
                } else if m + 1 == c {
                    -pdf_lower / p
                } else {
                    0.0
                };

                if d_p_d_alpha_m.abs() < 1e-20 {
                    continue;
                }

                // Chain rule: d(alpha_m)/d(theta) depends on reparametrization
                // alpha_0 = theta[k], alpha_m = alpha_{m-1} + exp(theta[k+m])
                // d(alpha_m)/d(theta[k]) = 1 for all m
                gradient[k] += d_p_d_alpha_m;
                // d(alpha_m)/d(theta[k+s]) = exp(theta[k+s]) for s <= m
                for s in 1..=m {
                    gradient[k + s] += d_p_d_alpha_m * theta[k + s].exp();
                }
            }

            // Approximate Hessian via outer product of score (BHHH)
            // For stability, use this instead of exact Hessian
            let mut score_i = Array1::<f64>::zeros(total_params);
            for kk in 0..k {
                score_i[kk] = dp_dbeta_factor * x_i[kk];
            }
            for m in 0..j_minus_1 {
                let d_p_d_alpha_m = if c == m {
                    pdf_upper / p
                } else if m + 1 == c {
                    -pdf_lower / p
                } else {
                    0.0
                };
                if d_p_d_alpha_m.abs() < 1e-20 {
                    continue;
                }
                score_i[k] += d_p_d_alpha_m;
                for s in 1..=m {
                    score_i[k + s] += d_p_d_alpha_m * theta[k + s].exp();
                }
            }

            // BHHH: H ≈ -Σ score_i * score_i'
            for a in 0..total_params {
                for b in 0..total_params {
                    hessian[[a, b]] -= score_i[a] * score_i[b];
                }
            }
        }

        // Newton step
        let neg_hessian = -&hessian;
        let inv_neg_hessian = match neg_hessian.inv() {
            Ok(m) => m,
            Err(_) => {
                // Fallback: add ridge
                let mut ridge = neg_hessian;
                for i in 0..total_params {
                    ridge[[i, i]] += 1e-4;
                }
                ridge.inv().map_err(|_| GreenersError::OptimizationFailed)?
            }
        };

        let change = inv_neg_hessian.dot(&gradient);

        // Line search: halve step if log-likelihood decreases
        let mut step_size = 1.0;
        for _ in 0..10 {
            let theta_new = &theta + step_size * &change;
            // Evaluate log-likelihood at new theta
            let beta_new = theta_new.slice(ndarray::s![..k]).to_owned();
            let mut alphas_new = vec![0.0; j_minus_1];
            alphas_new[0] = theta_new[k];
            for m in 1..j_minus_1 {
                alphas_new[m] = alphas_new[m - 1] + theta_new[k + m].exp();
            }

            let mut ll_new = 0.0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let xb_new: f64 = x
                    .row(i)
                    .iter()
                    .zip(beta_new.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let c = y_idx[i];
                let f_u = if c < j_minus_1 {
                    cdf_fn(alphas_new[c] - xb_new)
                } else {
                    1.0
                };
                let f_l = if c > 0 {
                    cdf_fn(alphas_new[c - 1] - xb_new)
                } else {
                    0.0
                };
                ll_new += (f_u - f_l).max(1e-15).ln();
            }

            if ll_new >= log_likelihood - 1e-8 {
                theta = theta_new;
                break;
            }
            step_size *= 0.5;
        }

        let diff = (step_size * &change).mapv(|v| v.powi(2)).sum().sqrt();
        if diff < tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(GreenersError::OptimizationFailed);
    }

    // Final values
    let beta = theta.slice(ndarray::s![..k]).to_owned();
    let mut alphas = vec![0.0; j_minus_1];
    alphas[0] = theta[k];
    for m in 1..j_minus_1 {
        alphas[m] = alphas[m - 1] + theta[k + m].exp();
    }

    // Compute final Hessian for SEs (BHHH)
    let mut hessian = Array2::<f64>::zeros((total_params, total_params));
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let x_i = x.row(i);
        let xb: f64 = x_i.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        let c = y_idx[i];

        let z_upper = if c < j_minus_1 {
            alphas[c] - xb
        } else {
            f64::INFINITY
        };
        let z_lower = if c > 0 {
            alphas[c - 1] - xb
        } else {
            f64::NEG_INFINITY
        };
        let f_upper = if z_upper.is_finite() {
            cdf_fn(z_upper)
        } else {
            1.0
        };
        let f_lower = if z_lower.is_finite() {
            cdf_fn(z_lower)
        } else {
            0.0
        };
        let p = (f_upper - f_lower).max(1e-15);
        let pdf_upper = if z_upper.is_finite() {
            pdf_fn(z_upper)
        } else {
            0.0
        };
        let pdf_lower = if z_lower.is_finite() {
            pdf_fn(z_lower)
        } else {
            0.0
        };
        let dp_dbeta_factor = -(pdf_upper - pdf_lower) / p;

        let mut score_i = Array1::<f64>::zeros(total_params);
        for kk in 0..k {
            score_i[kk] = dp_dbeta_factor * x_i[kk];
        }
        for m in 0..j_minus_1 {
            let d_p_d_alpha_m = if c == m {
                pdf_upper / p
            } else if m + 1 == c {
                -pdf_lower / p
            } else {
                0.0
            };
            if d_p_d_alpha_m.abs() < 1e-20 {
                continue;
            }
            score_i[k] += d_p_d_alpha_m;
            for s in 1..=m {
                score_i[k + s] += d_p_d_alpha_m * theta[k + s].exp();
            }
        }

        for a in 0..total_params {
            for b in 0..total_params {
                hessian[[a, b]] -= score_i[a] * score_i[b];
            }
        }
    }

    let cov_matrix = (-&hessian)
        .inv()
        .map_err(|_| GreenersError::SingularMatrix)?;

    // Extract SEs
    let std_errors: Array1<f64> = (0..k).map(|i| cov_matrix[[i, i]].max(0.0).sqrt()).collect();
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let z_values = &beta / &std_errors.mapv(|se| if se > 1e-15 { se } else { 1.0 });
    let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));

    let threshold_ses: Vec<f64> = (0..j_minus_1)
        .map(|m| cov_matrix[[k + m, k + m]].max(0.0).sqrt())
        .collect();

    // Null LL
    let mut freq = vec![0.0; j];
    for &idx in &y_idx {
        freq[idx] += 1.0;
    }
    let ll_null: f64 = y_idx.iter().map(|&idx| (freq[idx] / n as f64).ln()).sum();
    let pseudo_r2 = 1.0 - log_likelihood / ll_null;

    let k_total = total_params as f64;
    let aic = -2.0 * log_likelihood + 2.0 * k_total;
    let bic = -2.0 * log_likelihood + k_total * (n as f64).ln();

    let model_name = if is_logit {
        "Ordered Logit".to_string()
    } else {
        "Ordered Probit".to_string()
    };

    Ok(OrderedResult {
        model_name,
        params: beta,
        std_errors,
        z_values,
        p_values,
        thresholds: alphas,
        threshold_std_errors: threshold_ses,
        log_likelihood,
        pseudo_r2,
        aic,
        bic,
        n_obs: n,
        n_categories: j,
        iterations: iter,
        converged,
        category_labels: categories,
        inference_type: InferenceType::Normal,
        variable_names,
        _x_data: x.to_owned(),
    })
}
