use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::InferenceType;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;
use std::fmt;

/// Result from Conditional Logit/Poisson models.
#[derive(Debug)]
pub struct ConditionalResult {
    pub model_name: String,
    /// Coefficients (no intercept — absorbed by group FE).
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub n_groups: usize,
    pub iterations: usize,
    pub converged: bool,
    pub inference_type: InferenceType,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for ConditionalResult {
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
            "No. Groups:", self.n_groups, "AIC:", self.aic
        )?;

        writeln!(f, "\n{:-^78}", "")?;
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

        writeln!(f, "{:=^78}", "")
    }
}

impl ConditionalResult {
    /// Model stats: (AIC, BIC, LogLik).
    pub fn model_stats(&self) -> (f64, f64, f64) {
        (self.aic, self.bic, self.log_likelihood)
    }
}

/// Conditional Logit (Chamberlain's fixed-effects logit).
///
/// Conditions on the sum of y within each group, eliminating the group FE.
/// Only groups with variation in y contribute to the likelihood.
pub struct ConditionalLogit;

impl ConditionalLogit {
    /// Fit conditional logit.
    /// `groups`: group ID for each observation.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[usize],
    ) -> Result<ConditionalResult, GreenersError> {
        Self::fit_with_names(y, x, groups, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[usize],
        variable_names: Option<Vec<String>>,
    ) -> Result<ConditionalResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != groups.len() {
            return Err(GreenersError::ShapeMismatch(
                "y and groups must have same length".into(),
            ));
        }

        // Group observations
        let mut group_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &g) in groups.iter().enumerate() {
            group_map.entry(g).or_default().push(i);
        }

        // Filter groups with variation in y (sum > 0 and sum < group_size)
        let valid_groups: Vec<Vec<usize>> = group_map
            .values()
            .filter(|indices| {
                let sum: f64 = indices.iter().map(|&i| y[i]).sum();
                let n_g = indices.len() as f64;
                sum > 0.5 && sum < n_g - 0.5
            })
            .cloned()
            .collect();

        if valid_groups.is_empty() {
            return Err(GreenersError::InvalidOperation(
                "No groups with variation in y".into(),
            ));
        }

        let n_groups = valid_groups.len();

        // Newton-Raphson on conditional log-likelihood
        let mut beta = Array1::<f64>::zeros(k);
        let max_iter = 100;
        let tol = 1e-6;
        let mut converged = false;
        let mut iter = 0;
        let mut log_likelihood = 0.0;

        for iteration in 0..max_iter {
            iter = iteration + 1;

            let mut gradient = Array1::<f64>::zeros(k);
            let mut hessian = Array2::<f64>::zeros((k, k));
            log_likelihood = 0.0;

            for group_indices in &valid_groups {
                let n_g = group_indices.len();
                let s_g: usize = group_indices.iter().map(|&i| y[i] as usize).sum();

                // For small groups, enumerate all combinations of size s_g
                // For large groups, approximate via conditional Poisson
                if n_g <= 20 && s_g <= 10 {
                    // Exact enumeration
                    let xb: Vec<f64> = group_indices.iter().map(|&i| x.row(i).dot(&beta)).collect();

                    // Observed: sum of x_i where y_i = 1
                    let mut x_obs = Array1::<f64>::zeros(k);
                    for &i in group_indices {
                        if y[i] > 0.5 {
                            x_obs = &x_obs + &x.row(i).to_owned();
                        }
                    }

                    // Enumerate all combinations of s_g from n_g
                    let combos = combinations(n_g, s_g);
                    let mut e_x = Array1::<f64>::zeros(k);
                    let mut e_xx = Array2::<f64>::zeros((k, k));

                    // First pass: find max for log-sum-exp
                    let combo_sums: Vec<f64> = combos
                        .iter()
                        .map(|combo| combo.iter().map(|&j| xb[j]).sum::<f64>())
                        .collect();

                    let max_sum = combo_sums.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                    let mut total_w = 0.0;
                    let mut weighted_x = vec![Array1::<f64>::zeros(k); combos.len()];

                    for (ci, combo) in combos.iter().enumerate() {
                        let w = (combo_sums[ci] - max_sum).exp();
                        total_w += w;

                        let mut x_combo = Array1::<f64>::zeros(k);
                        for &j in combo {
                            x_combo = &x_combo + &x.row(group_indices[j]).to_owned();
                        }
                        weighted_x[ci] = x_combo;
                    }

                    let log_denom = max_sum + total_w.ln();
                    let obs_sum: f64 = group_indices
                        .iter()
                        .filter(|&&i| y[i] > 0.5)
                        .map(|&i| x.row(i).dot(&beta))
                        .sum();

                    log_likelihood += obs_sum - log_denom;

                    // E[X] and E[XX'] under the conditional distribution
                    for (ci, _combo) in combos.iter().enumerate() {
                        let w = (combo_sums[ci] - max_sum).exp() / total_w;
                        let x_c = &weighted_x[ci];
                        e_x = &e_x + &(x_c * w);

                        for a in 0..k {
                            for b in 0..k {
                                e_xx[[a, b]] += w * x_c[a] * x_c[b];
                            }
                        }
                    }

                    // Gradient: x_obs - E[X]
                    gradient = &gradient + &(&x_obs - &e_x);

                    // Hessian: -(E[XX'] - E[X]*E[X]')
                    for a in 0..k {
                        for b in 0..k {
                            hessian[[a, b]] -= e_xx[[a, b]] - e_x[a] * e_x[b];
                        }
                    }
                } else {
                    // Large group approximation: use conditional Poisson (Andersen, 1970)
                    // This is asymptotically equivalent
                    let xb: Vec<f64> = group_indices.iter().map(|&i| x.row(i).dot(&beta)).collect();
                    let exp_xb: Vec<f64> = xb.iter().map(|v| v.exp()).collect();
                    let sum_exp: f64 = exp_xb.iter().sum();

                    for (j_idx, &i) in group_indices.iter().enumerate() {
                        let p_j = exp_xb[j_idx] / sum_exp;
                        let diff = y[i] - s_g as f64 * p_j;
                        for kk in 0..k {
                            gradient[kk] += diff * x[[i, kk]];
                        }

                        for kk in 0..k {
                            for ll in 0..k {
                                hessian[[kk, ll]] -=
                                    s_g as f64 * p_j * (1.0 - p_j) * x[[i, kk]] * x[[i, ll]];
                            }
                        }
                    }

                    // Approximate log-likelihood contribution
                    for (j_idx, &i) in group_indices.iter().enumerate() {
                        if y[i] > 0.5 {
                            log_likelihood += (exp_xb[j_idx] / sum_exp).max(1e-15).ln();
                        }
                    }
                }
            }

            // Newton step
            let neg_hessian = -&hessian;
            let inv_neg_hessian = match neg_hessian.inv() {
                Ok(m) => m,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };

            let change = inv_neg_hessian.dot(&gradient);
            beta = &beta + &change;

            if change.mapv(|v| v.powi(2)).sum().sqrt() < tol {
                converged = true;
                break;
            }
        }

        // Standard errors from final Hessian
        let mut final_hessian = Array2::<f64>::zeros((k, k));
        for group_indices in &valid_groups {
            let n_g = group_indices.len();
            let s_g: usize = group_indices.iter().map(|&i| y[i] as usize).sum();

            if n_g <= 20 && s_g <= 10 {
                let xb: Vec<f64> = group_indices.iter().map(|&i| x.row(i).dot(&beta)).collect();
                let combos = combinations(n_g, s_g);
                let combo_sums: Vec<f64> = combos
                    .iter()
                    .map(|combo| combo.iter().map(|&j| xb[j]).sum::<f64>())
                    .collect();
                let max_sum = combo_sums.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                let mut total_w = 0.0;
                let mut weighted_x = vec![Array1::<f64>::zeros(k); combos.len()];
                for (ci, combo) in combos.iter().enumerate() {
                    let w = (combo_sums[ci] - max_sum).exp();
                    total_w += w;
                    let mut x_combo = Array1::<f64>::zeros(k);
                    for &j in combo {
                        x_combo = &x_combo + &x.row(group_indices[j]).to_owned();
                    }
                    weighted_x[ci] = x_combo;
                }

                let mut e_x = Array1::<f64>::zeros(k);
                let mut e_xx = Array2::<f64>::zeros((k, k));
                for (ci, _) in combos.iter().enumerate() {
                    let w = (combo_sums[ci] - max_sum).exp() / total_w;
                    let x_c = &weighted_x[ci];
                    e_x = &e_x + &(x_c * w);
                    for a in 0..k {
                        for b in 0..k {
                            e_xx[[a, b]] += w * x_c[a] * x_c[b];
                        }
                    }
                }

                for a in 0..k {
                    for b in 0..k {
                        final_hessian[[a, b]] -= e_xx[[a, b]] - e_x[a] * e_x[b];
                    }
                }
            } else {
                let exp_xb: Vec<f64> = group_indices
                    .iter()
                    .map(|&i| x.row(i).dot(&beta).exp())
                    .collect();
                let sum_exp: f64 = exp_xb.iter().sum();

                for (j_idx, &i) in group_indices.iter().enumerate() {
                    let p_j = exp_xb[j_idx] / sum_exp;
                    for kk in 0..k {
                        for ll in 0..k {
                            final_hessian[[kk, ll]] -=
                                s_g as f64 * p_j * (1.0 - p_j) * x[[i, kk]] * x[[i, ll]];
                        }
                    }
                }
            }
        }

        let cov_matrix = (-&final_hessian).inv().unwrap_or(Array2::eye(k) * 1e-4);
        let std_errors: Array1<f64> = (0..k).map(|i| cov_matrix[[i, i]].max(0.0).sqrt()).collect();

        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let z_values = &beta / std_errors.mapv(|s| if s > 1e-15 { s } else { 1.0 });
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));

        let k_f = k as f64;
        let aic = -2.0 * log_likelihood + 2.0 * k_f;
        let bic = -2.0 * log_likelihood + k_f * (n as f64).ln();

        Ok(ConditionalResult {
            model_name: "Conditional Logit".to_string(),
            params: beta,
            std_errors,
            z_values,
            p_values,
            log_likelihood,
            aic,
            bic,
            n_obs: n,
            n_groups,
            iterations: iter,
            converged,
            inference_type: InferenceType::Normal,
            variable_names,
        })
    }
}

/// Conditional Poisson (Hausman-Hall-Griliches).
///
/// Conditions on the sum of y within each group.
/// Equivalent to Poisson FE; the conditional likelihood eliminates the FE.
pub struct ConditionalPoisson;

impl ConditionalPoisson {
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[usize],
    ) -> Result<ConditionalResult, GreenersError> {
        Self::fit_with_names(y, x, groups, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[usize],
        variable_names: Option<Vec<String>>,
    ) -> Result<ConditionalResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != groups.len() {
            return Err(GreenersError::ShapeMismatch(
                "y and groups must have same length".into(),
            ));
        }

        let mut group_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &g) in groups.iter().enumerate() {
            group_map.entry(g).or_default().push(i);
        }

        // Filter groups with positive total count
        let valid_groups: Vec<Vec<usize>> = group_map
            .values()
            .filter(|indices| {
                let sum: f64 = indices.iter().map(|&i| y[i]).sum();
                sum > 0.5
            })
            .cloned()
            .collect();

        if valid_groups.is_empty() {
            return Err(GreenersError::InvalidOperation(
                "No groups with positive counts".into(),
            ));
        }

        let n_groups = valid_groups.len();

        // Newton-Raphson on conditional Poisson log-likelihood
        // L_g = Π (exp(x_it β) / Σ_s exp(x_is β))^{y_it}
        let mut beta = Array1::<f64>::zeros(k);
        let max_iter = 100;
        let tol = 1e-6;
        let mut converged = false;
        let mut iter = 0;
        let mut log_likelihood = 0.0;

        for iteration in 0..max_iter {
            iter = iteration + 1;

            let mut gradient = Array1::<f64>::zeros(k);
            let mut hessian = Array2::<f64>::zeros((k, k));
            log_likelihood = 0.0;

            for group_indices in &valid_groups {
                let s_g: f64 = group_indices.iter().map(|&i| y[i]).sum();

                let exp_xb: Vec<f64> = group_indices
                    .iter()
                    .map(|&i| x.row(i).dot(&beta).exp())
                    .collect();
                let sum_exp: f64 = exp_xb.iter().sum();

                // Log-likelihood contribution
                for (j_idx, &i) in group_indices.iter().enumerate() {
                    if y[i] > 0.0 {
                        log_likelihood += y[i] * (exp_xb[j_idx] / sum_exp).max(1e-15).ln();
                    }
                }

                // Gradient and Hessian
                let mut e_x = Array1::<f64>::zeros(k);
                for (j_idx, &i) in group_indices.iter().enumerate() {
                    let p_j = exp_xb[j_idx] / sum_exp;
                    for kk in 0..k {
                        e_x[kk] += p_j * x[[i, kk]];
                    }
                }

                // Gradient: Σ y_it (x_it - E[x])
                for &i in group_indices {
                    for kk in 0..k {
                        gradient[kk] += y[i] * (x[[i, kk]] - e_x[kk]);
                    }
                }

                // Hessian: -s_g * (E[xx'] - E[x]E[x]')
                let mut e_xx = Array2::<f64>::zeros((k, k));
                for (j_idx, &i) in group_indices.iter().enumerate() {
                    let p_j = exp_xb[j_idx] / sum_exp;
                    for a in 0..k {
                        for b in 0..k {
                            e_xx[[a, b]] += p_j * x[[i, a]] * x[[i, b]];
                        }
                    }
                }

                for a in 0..k {
                    for b in 0..k {
                        hessian[[a, b]] -= s_g * (e_xx[[a, b]] - e_x[a] * e_x[b]);
                    }
                }
            }

            let neg_hessian = -&hessian;
            let inv_neg_hessian = match neg_hessian.inv() {
                Ok(m) => m,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };

            let change = inv_neg_hessian.dot(&gradient);
            beta = &beta + &change;

            if change.mapv(|v| v.powi(2)).sum().sqrt() < tol {
                converged = true;
                break;
            }
        }

        // Final covariance
        let mut final_hessian = Array2::<f64>::zeros((k, k));
        for group_indices in &valid_groups {
            let s_g: f64 = group_indices.iter().map(|&i| y[i]).sum();
            let exp_xb: Vec<f64> = group_indices
                .iter()
                .map(|&i| x.row(i).dot(&beta).exp())
                .collect();
            let sum_exp: f64 = exp_xb.iter().sum();

            let mut e_x = Array1::<f64>::zeros(k);
            let mut e_xx = Array2::<f64>::zeros((k, k));
            for (j_idx, &i) in group_indices.iter().enumerate() {
                let p_j = exp_xb[j_idx] / sum_exp;
                for kk in 0..k {
                    e_x[kk] += p_j * x[[i, kk]];
                }
                for a in 0..k {
                    for b in 0..k {
                        e_xx[[a, b]] += p_j * x[[i, a]] * x[[i, b]];
                    }
                }
            }

            for a in 0..k {
                for b in 0..k {
                    final_hessian[[a, b]] -= s_g * (e_xx[[a, b]] - e_x[a] * e_x[b]);
                }
            }
        }

        let cov_matrix = (-&final_hessian).inv().unwrap_or(Array2::eye(k) * 1e-4);
        let std_errors: Array1<f64> = (0..k).map(|i| cov_matrix[[i, i]].max(0.0).sqrt()).collect();

        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let z_values = &beta / std_errors.mapv(|s| if s > 1e-15 { s } else { 1.0 });
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));

        let k_f = k as f64;
        let aic = -2.0 * log_likelihood + 2.0 * k_f;
        let bic = -2.0 * log_likelihood + k_f * (n as f64).ln();

        Ok(ConditionalResult {
            model_name: "Conditional Poisson".to_string(),
            params: beta,
            std_errors,
            z_values,
            p_values,
            log_likelihood,
            aic,
            bic,
            n_obs: n,
            n_groups,
            iterations: iter,
            converged,
            inference_type: InferenceType::Normal,
            variable_names,
        })
    }
}

/// Conditional Multinomial Logit (McFadden's choice model).
///
/// Softmax likelihood within each choice set (group).
/// Each group has `n_alts` alternatives; y indicates the chosen one.
pub struct ConditionalMNLogit;

impl ConditionalMNLogit {
    /// Fit conditional multinomial logit.
    ///
    /// - `y`: chosen alternative index (0-based) for each choice occasion
    /// - `x`: stacked design matrix (n_occasions * n_alts) x k
    /// - `groups`: group/choice-set ID for each row of x
    /// - `n_alts`: number of alternatives per choice set
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[usize],
        _n_alts: usize,
    ) -> Result<ConditionalResult, GreenersError> {
        Self::fit_with_names(y, x, groups, _n_alts, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[usize],
        _n_alts: usize,
        variable_names: Option<Vec<String>>,
    ) -> Result<ConditionalResult, GreenersError> {
        let n_rows = x.nrows();
        let k = x.ncols();

        if n_rows != groups.len() {
            return Err(GreenersError::ShapeMismatch(
                "x rows and groups must have same length".into(),
            ));
        }

        // Build choice sets
        let mut group_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &g) in groups.iter().enumerate() {
            group_map.entry(g).or_default().push(i);
        }

        let mut choice_sets: Vec<Vec<usize>> = group_map.values().cloned().collect();
        choice_sets.sort_by_key(|v| v[0]);

        let n_occasions = y.len();
        if choice_sets.len() != n_occasions {
            return Err(GreenersError::ShapeMismatch(
                "Number of groups must equal length of y".into(),
            ));
        }

        // Newton-Raphson
        let mut beta = Array1::<f64>::zeros(k);
        let max_iter = 100;
        let tol = 1e-6;
        let mut converged = false;
        let mut iter = 0;
        let mut log_likelihood = 0.0;

        for iteration in 0..max_iter {
            iter = iteration + 1;
            let mut gradient = Array1::<f64>::zeros(k);
            let mut hessian = Array2::<f64>::zeros((k, k));
            log_likelihood = 0.0;

            for (occ, indices) in choice_sets.iter().enumerate() {
                let chosen = y[occ] as usize;

                // Compute exp(x_j' beta) for each alternative
                let xb: Vec<f64> = indices.iter().map(|&i| x.row(i).dot(&beta)).collect();
                let max_xb = xb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp_xb: Vec<f64> = xb.iter().map(|&v| (v - max_xb).exp()).collect();
                let sum_exp: f64 = exp_xb.iter().sum();

                // Log-likelihood: xb[chosen] - log(sum_exp) - max_xb + max_xb
                if chosen < indices.len() {
                    log_likelihood += xb[chosen] - max_xb - sum_exp.ln();
                }

                // Gradient and Hessian
                let probs: Vec<f64> = exp_xb.iter().map(|&e| e / sum_exp).collect();

                // E[x] = sum p_j x_j
                let mut e_x = Array1::<f64>::zeros(k);
                for (j, &idx) in indices.iter().enumerate() {
                    let xj = x.row(idx);
                    for kk in 0..k {
                        e_x[kk] += probs[j] * xj[kk];
                    }
                }

                // gradient += x_chosen - E[x]
                if chosen < indices.len() {
                    let x_chosen = x.row(indices[chosen]);
                    for kk in 0..k {
                        gradient[kk] += x_chosen[kk] - e_x[kk];
                    }
                }

                // Hessian -= E[xx'] - E[x]E[x]'
                for (j, &idx) in indices.iter().enumerate() {
                    let xj = x.row(idx);
                    for a in 0..k {
                        for b in 0..k {
                            hessian[[a, b]] -= probs[j] * xj[a] * xj[b];
                        }
                    }
                }
                for a in 0..k {
                    for b in 0..k {
                        hessian[[a, b]] += e_x[a] * e_x[b];
                    }
                }
            }

            let neg_hessian = -&hessian;
            let inv_neg_hessian = match neg_hessian.inv() {
                Ok(m) => m,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };

            let change = inv_neg_hessian.dot(&gradient);
            beta = &beta + &change;

            if change.mapv(|v| v.powi(2)).sum().sqrt() < tol {
                converged = true;
                break;
            }
        }

        // Standard errors
        let mut final_hessian = Array2::<f64>::zeros((k, k));
        for indices in &choice_sets {
            let xb: Vec<f64> = indices.iter().map(|&i| x.row(i).dot(&beta)).collect();
            let max_xb = xb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_xb: Vec<f64> = xb.iter().map(|&v| (v - max_xb).exp()).collect();
            let sum_exp: f64 = exp_xb.iter().sum();
            let probs: Vec<f64> = exp_xb.iter().map(|&e| e / sum_exp).collect();

            let mut e_x = Array1::<f64>::zeros(k);
            for (j, &idx) in indices.iter().enumerate() {
                let xj = x.row(idx);
                for kk in 0..k {
                    e_x[kk] += probs[j] * xj[kk];
                }
            }

            for (j, &idx) in indices.iter().enumerate() {
                let xj = x.row(idx);
                for a in 0..k {
                    for b in 0..k {
                        final_hessian[[a, b]] -= probs[j] * xj[a] * xj[b];
                    }
                }
            }
            for a in 0..k {
                for b in 0..k {
                    final_hessian[[a, b]] += e_x[a] * e_x[b];
                }
            }
        }

        let cov_matrix = (-&final_hessian).inv().unwrap_or(Array2::eye(k) * 1e-4);
        let std_errors: Array1<f64> = (0..k).map(|i| cov_matrix[[i, i]].max(0.0).sqrt()).collect();

        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let z_values = &beta / std_errors.mapv(|s| if s > 1e-15 { s } else { 1.0 });
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal_dist.cdf(z.abs())));

        let k_f = k as f64;
        let n = n_rows;
        let aic = -2.0 * log_likelihood + 2.0 * k_f;
        let bic = -2.0 * log_likelihood + k_f * (n as f64).ln();

        Ok(ConditionalResult {
            model_name: "Conditional MNLogit".to_string(),
            params: beta,
            std_errors,
            z_values,
            p_values,
            log_likelihood,
            aic,
            bic,
            n_obs: n_rows,
            n_groups: choice_sets.len(),
            iterations: iter,
            converged,
            inference_type: crate::InferenceType::Normal,
            variable_names,
        })
    }
}

/// Generate all combinations of `r` elements from `0..n`.
fn combinations(n: usize, r: usize) -> Vec<Vec<usize>> {
    if r == 0 {
        return vec![vec![]];
    }
    if r > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut combo = vec![0usize; r];
    // Initialize
    for (i, item) in combo.iter_mut().enumerate().take(r) {
        *item = i;
    }

    loop {
        result.push(combo.clone());

        // Find rightmost element that can be incremented
        let mut i = r;
        loop {
            if i == 0 {
                return result;
            }
            i -= 1;
            if combo[i] < n - r + i {
                break;
            }
            if i == 0 {
                return result;
            }
        }

        combo[i] += 1;
        for j in (i + 1)..r {
            combo[j] = combo[j - 1] + 1;
        }
    }
}
