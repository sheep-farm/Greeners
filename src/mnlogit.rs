use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::{DataFrame, Formula, InferenceType};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result from Multinomial Logit regression.
#[derive(Debug)]
pub struct MNLogitResult {
    /// Coefficients: (k x J-1) — one column per non-base category.
    pub params: Array2<f64>,
    /// Standard errors: (k x J-1).
    pub std_errors: Array2<f64>,
    /// Z-values: (k x J-1).
    pub z_values: Array2<f64>,
    /// P-values: (k x J-1).
    pub p_values: Array2<f64>,
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
    pub omitted_vars: Vec<(usize, String)>,
    _x_data: Array2<f64>,
}

impl fmt::Display for MNLogitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Multinomial Logit Regression Results ")?;
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
            "Method:", "Newton-Raphson", "AIC:", self.aic
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Iterations:", self.iterations, "BIC:", self.bic
        )?;

        let j_minus_1 = self.n_categories - 1;
        let base_label = self.category_labels[self.n_categories - 1];

        for j in 0..j_minus_1 {
            let cat_label = self.category_labels[j];
            writeln!(
                f,
                "\n{:-^78}",
                format!(" y={} vs base y={} ", cat_label, base_label)
            )?;
            writeln!(
                f,
                "{:<12} {:>10} {:>10} {:>8} {:>8}",
                "", "coef", "std err", "z", "P>|z|"
            )?;
            writeln!(f, "{:-^78}", "")?;

            for i in 0..self.params.nrows() {
                let name = self
                    .variable_names
                    .as_ref()
                    .and_then(|n| n.get(i).cloned())
                    .unwrap_or_else(|| format!("x{}", i));
                writeln!(
                    f,
                    "{:<12} {:>10.4} {:>10.4} {:>8.3} {:>8.3}",
                    name,
                    self.params[[i, j]],
                    self.std_errors[[i, j]],
                    self.z_values[[i, j]],
                    self.p_values[[i, j]]
                )?;
            }
        }

        writeln!(f, "{:=^78}", "")?;
        for (_, name) in &self.omitted_vars {
            writeln!(f, "note: {} omitted because of collinearity", name)?;
        }
        Ok(())
    }
}

impl MNLogitResult {
    /// Predicted probabilities for each category: (n x J).
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let j = self.n_categories;
        let j_minus_1 = j - 1;
        let mut probs = Array2::<f64>::zeros((n, j));

        for i in 0..n {
            let x_i = x.row(i);
            let mut max_eta = 0.0f64; // base category eta = 0
            let mut etas = vec![0.0; j];
            #[allow(clippy::needless_range_loop)]
            for c in 0..j_minus_1 {
                etas[c] = x_i.dot(&self.params.column(c));
                max_eta = max_eta.max(etas[c]);
            }
            // base category
            etas[j_minus_1] = 0.0;
            max_eta = max_eta.max(0.0);

            // Softmax with log-sum-exp trick
            let mut sum_exp = 0.0;
            for c in 0..j {
                let e = (etas[c] - max_eta).exp();
                probs[[i, c]] = e;
                sum_exp += e;
            }
            for c in 0..j {
                probs[[i, c]] /= sum_exp;
            }
        }

        probs
    }

    /// Predicted category (argmax of probabilities).
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let probs = self.predict_proba(x);
        let n = probs.nrows();
        let mut predictions = Array1::<f64>::zeros(n);
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
            predictions[i] = self.category_labels[max_idx];
        }
        predictions
    }

    /// Relative Risk Ratios: exp(β). Shape (k x J-1).
    pub fn rrr(&self) -> Array2<f64> {
        self.params.mapv(f64::exp)
    }

    /// Model stats: (AIC, BIC, LogLik, PseudoR2).
    pub fn model_stats(&self) -> (f64, f64, f64, f64) {
        (self.aic, self.bic, self.log_likelihood, self.pseudo_r2)
    }
}

/// Multinomial Logit estimator.
pub struct MNLogit;

impl MNLogit {
    /// Fit via formula.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
    ) -> Result<MNLogitResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        let var_names = data.formula_var_names(formula)?;
        Self::fit_with_names(&y, &x, Some(var_names))
    }

    /// Fit from arrays.
    pub fn fit(y: &Array1<f64>, x: &Array2<f64>) -> Result<MNLogitResult, GreenersError> {
        Self::fit_with_names(y, x, None)
    }

    /// Fit with variable names.
    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<MNLogitResult, GreenersError> {
        let n = x.nrows();
        let _k = x.ncols();

        // Validate input
        if y.iter().any(|v| !v.is_finite()) || x.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "Input data contains NaN or Inf values".into(),
            ));
        }

        // Detect and sort unique categories
        let mut categories: Vec<f64> = y.iter().copied().collect();
        categories.sort_by(|a, b| a.partial_cmp(b).unwrap());
        categories.dedup();
        let j = categories.len();

        if j < 3 {
            return Err(GreenersError::InvalidOperation(
                "MNLogit requires at least 3 categories. Use Logit for binary outcomes.".into(),
            ));
        }

        let j_minus_1 = j - 1;

        // Map y values to category indices
        let y_idx: Vec<usize> = y
            .iter()
            .map(|val| {
                categories
                    .iter()
                    .position(|c| (c - val).abs() < 1e-10)
                    .unwrap_or(0)
            })
            .collect();

        let (x_clean, omitted_positioned, clean_var_names) = if let Some(ref names) = variable_names
        {
            let cr = crate::linalg::drop_collinear(x, names, 1e-10);
            (cr.x_clean, cr.omitted, cr.clean_names)
        } else {
            (x.clone(), vec![], vec![])
        };

        let x_use = &x_clean;
        let k_clean = x_use.ncols();

        if n <= k_clean * j_minus_1 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations for multinomial logit".into(),
            ));
        }

        // Newton-Raphson optimization
        // Parameter vector: β = [β_1; β_2; ...; β_{J-1}] of length k*(J-1)
        let total_params = k_clean * j_minus_1;
        let mut beta = Array1::<f64>::zeros(total_params);

        let tol = 1e-6;
        let max_iter = 100;
        let mut converged = false;
        let mut iter = 0;
        let mut log_likelihood = 0.0;

        for iteration in 0..max_iter {
            iter = iteration + 1;

            // Compute probabilities (softmax)
            let mut probs = Array2::<f64>::zeros((n, j));
            for i in 0..n {
                let x_i = x_use.row(i);
                let mut max_eta = 0.0f64;
                let mut etas = vec![0.0; j];
                #[allow(clippy::needless_range_loop)]
                for c in 0..j_minus_1 {
                    let beta_c = beta.slice(ndarray::s![c * k_clean..(c + 1) * k_clean]);
                    etas[c] = x_i.dot(&beta_c);
                    max_eta = max_eta.max(etas[c]);
                }
                etas[j_minus_1] = 0.0;
                max_eta = max_eta.max(0.0);

                let mut sum_exp = 0.0;
                for c in 0..j {
                    let e = (etas[c] - max_eta).exp();
                    probs[[i, c]] = e;
                    sum_exp += e;
                }
                for c in 0..j {
                    probs[[i, c]] /= sum_exp;
                    probs[[i, c]] = probs[[i, c]].clamp(1e-15, 1.0 - 1e-15);
                }
            }

            // Log-likelihood
            log_likelihood = 0.0;
            for i in 0..n {
                log_likelihood += probs[[i, y_idx[i]]].ln();
            }

            // Gradient: g_c = X' * (d_c - p_c) for each c in 0..J-1
            let mut gradient = Array1::<f64>::zeros(total_params);
            for c in 0..j_minus_1 {
                for i in 0..n {
                    let indicator = if y_idx[i] == c { 1.0 } else { 0.0 };
                    let diff = indicator - probs[[i, c]];
                    for kk in 0..k_clean {
                        gradient[c * k_clean + kk] += x_use[[i, kk]] * diff;
                    }
                }
            }

            // Hessian: H_{c,c'} = -X' diag(p_c * (δ_{cc'} - p_{c'})) X
            let mut hessian = Array2::<f64>::zeros((total_params, total_params));
            for c in 0..j_minus_1 {
                for c2 in 0..j_minus_1 {
                    // Block (c, c2) of size k_clean x k_clean
                    for i in 0..n {
                        let w = if c == c2 {
                            -probs[[i, c]] * (1.0 - probs[[i, c]])
                        } else {
                            probs[[i, c]] * probs[[i, c2]]
                        };
                        for kk in 0..k_clean {
                            for ll in 0..k_clean {
                                hessian[[c * k_clean + kk, c2 * k_clean + ll]] +=
                                    w * x_use[[i, kk]] * x_use[[i, ll]];
                            }
                        }
                    }
                }
            }

            // Newton step: delta = -H^{-1} * g
            let neg_hessian = -&hessian;
            let inv_neg_hessian = match neg_hessian.inv() {
                Ok(m) => m,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };

            let change = inv_neg_hessian.dot(&gradient);
            beta = &beta + &change;

            let diff = change.mapv(|v| v.powi(2)).sum().sqrt();
            if diff < tol {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(GreenersError::OptimizationFailed);
        }

        // Extract parameter matrices and compute standard errors
        // Recompute Hessian at final estimates for covariance
        let mut probs = Array2::<f64>::zeros((n, j));
        for i in 0..n {
            let x_i = x_use.row(i);
            let mut max_eta = 0.0f64;
            let mut etas = vec![0.0; j];
            #[allow(clippy::needless_range_loop)]
            for c in 0..j_minus_1 {
                let beta_c = beta.slice(ndarray::s![c * k_clean..(c + 1) * k_clean]);
                etas[c] = x_i.dot(&beta_c);
                max_eta = max_eta.max(etas[c]);
            }
            etas[j_minus_1] = 0.0;
            max_eta = max_eta.max(0.0);

            let mut sum_exp = 0.0;
            for c in 0..j {
                let e = (etas[c] - max_eta).exp();
                probs[[i, c]] = e;
                sum_exp += e;
            }
            for c in 0..j {
                probs[[i, c]] /= sum_exp;
                probs[[i, c]] = probs[[i, c]].clamp(1e-15, 1.0 - 1e-15);
            }
        }

        let mut hessian = Array2::<f64>::zeros((total_params, total_params));
        for c in 0..j_minus_1 {
            for c2 in 0..j_minus_1 {
                for i in 0..n {
                    let w = if c == c2 {
                        -probs[[i, c]] * (1.0 - probs[[i, c]])
                    } else {
                        probs[[i, c]] * probs[[i, c2]]
                    };
                    for kk in 0..k_clean {
                        for ll in 0..k_clean {
                            hessian[[c * k_clean + kk, c2 * k_clean + ll]] +=
                                w * x_use[[i, kk]] * x_use[[i, ll]];
                        }
                    }
                }
            }
        }

        let cov_matrix = (-&hessian).inv()?;

        // Build result matrices
        let mut params_mat = Array2::<f64>::zeros((k_clean, j_minus_1));
        let mut se_mat = Array2::<f64>::zeros((k_clean, j_minus_1));
        let mut z_mat = Array2::<f64>::zeros((k_clean, j_minus_1));
        let mut p_mat = Array2::<f64>::zeros((k_clean, j_minus_1));

        let normal_dist = Normal::new(0.0, 1.0).unwrap();

        for c in 0..j_minus_1 {
            for kk in 0..k_clean {
                let idx = c * k_clean + kk;
                params_mat[[kk, c]] = beta[idx];
                let se = cov_matrix[[idx, idx]].max(0.0).sqrt();
                se_mat[[kk, c]] = se;
                let z = if se > 1e-15 { beta[idx] / se } else { 0.0 };
                z_mat[[kk, c]] = z;
                p_mat[[kk, c]] = 2.0 * (1.0 - normal_dist.cdf(z.abs()));
            }
        }

        // Null log-likelihood (intercept only = proportional frequencies)
        let mut freq = vec![0.0; j];
        for &idx in &y_idx {
            freq[idx] += 1.0;
        }
        let ll_null: f64 = y_idx.iter().map(|&idx| (freq[idx] / n as f64).ln()).sum();

        let pseudo_r2 = 1.0 - log_likelihood / ll_null;
        let k_total = total_params as f64;
        let aic = -2.0 * log_likelihood + 2.0 * k_total;
        let bic = -2.0 * log_likelihood + k_total * (n as f64).ln();

        Ok(MNLogitResult {
            params: params_mat,
            std_errors: se_mat,
            z_values: z_mat,
            p_values: p_mat,
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
            variable_names: if !clean_var_names.is_empty() {
                Some(clean_var_names)
            } else {
                variable_names
            },
            omitted_vars: omitted_positioned,
            _x_data: x_use.clone(),
        })
    }
}
