//! Markov-Switching VAR (MS-VAR).
//!
//! Multivariate VAR where the intercept (and optionally variance)
//! depends on a latent Markov chain with K regimes.
//!
//! y_t = mu_{s_t} + A_1 y_{t-1} + ... + A_p y_{t-p} + eps_t
//! eps_t ~ N(0, Sigma_{s_t})
//!
//! s_t follows a Markov chain with transition matrix P.
//!
//! Estimation: EM algorithm with forward-backward (Baum-Welch).

use crate::linalg::{LinalgDeterminant as _, LinalgInverse as _};
use crate::GreenersError;
use ndarray::{Array1, Array2, Array3};
use std::fmt;

/// Result of MS-VAR estimation.
#[derive(Debug)]
pub struct MsVarResult {
    /// Regime-specific intercepts (K x n_vars)
    pub regime_intercepts: Array2<f64>,
    /// Common AR coefficients (n_vars x (n_vars * p))
    pub ar_coeffs: Array2<f64>,
    /// Regime-specific covariance matrices (K x n_vars x n_vars)
    pub regime_covariances: Array3<f64>,
    /// Transition probability matrix (K x K)
    pub transition_matrix: Array2<f64>,
    /// Filtered regime probabilities (T x K)
    pub filtered_probs: Array2<f64>,
    /// Smoothed regime probabilities (T x K)
    pub smoothed_probs: Array2<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Number of regimes
    pub n_regimes: usize,
    /// VAR lag order
    pub lags: usize,
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for MsVarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" MS-VAR ({} regimes, {} lags) ", self.n_regimes, self.lags)
        )?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Regimes:", self.n_regimes)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;
        writeln!(f, "{:<20} {:>12.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Transition matrix:")?;
        for i in 0..self.n_regimes {
            let mut row = format!("  From regime {i}: ");
            for j in 0..self.n_regimes {
                row.push_str(&format!("{:>10.4} ", self.transition_matrix[(i, j)]));
            }
            writeln!(f, "{row}")?;
        }

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Regime intercepts:")?;
        for r in 0..self.n_regimes {
            let mut row = format!("  Regime {r}: ");
            for j in 0..self.n_vars {
                row.push_str(&format!(
                    "{:>10}={:.4} ",
                    self.var_names.get(j).map(|s| s.as_str()).unwrap_or("?"),
                    self.regime_intercepts[(r, j)]
                ));
            }
            writeln!(f, "{row}")?;
        }

        writeln!(f, "\n  Smoothed regime probabilities (selected periods):")?;
        let t_mid = self.n_obs / 2;
        for &t in &[0, t_mid, self.n_obs - 1] {
            let mut row = format!("  t={:<6} ", t + 1);
            for r in 0..self.n_regimes {
                row.push_str(&format!("R{r}={:.3} ", self.smoothed_probs[(t, r)]));
            }
            writeln!(f, "{row}")?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct MSVAR;

impl MSVAR {
    /// Estimate MS-VAR with K regimes and p lags via EM (Baum-Welch).
    ///
    /// # Arguments
    /// * `y` - Data matrix (T x n_vars)
    /// * `n_regimes` - Number of regimes (K)
    /// * `lags` - VAR lag order (p)
    /// * `var_names` - Optional variable names
    pub fn fit(
        y: &Array2<f64>,
        n_regimes: usize,
        lags: usize,
        var_names: Option<Vec<String>>,
    ) -> Result<MsVarResult, GreenersError> {
        let t = y.nrows();
        let k = y.ncols();
        if t < (lags + 1) * 3 {
            return Err(GreenersError::InvalidOperation(
                "MS-VAR: too few observations".into(),
            ));
        }
        if n_regimes < 2 {
            return Err(GreenersError::InvalidOperation(
                "MS-VAR: need at least 2 regimes".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("y{i}")).collect());
        let n_eff = t - lags;

        // Build VAR design matrix: Z_t = [1, y_{t-1}, ..., y_{t-p}]
        // We use common AR coefficients but regime-specific intercepts.
        // So we split: y_t = mu_{s_t} + A * [y_{t-1},...,y_{t-p}] + eps
        let mut z_ar = Array2::zeros((n_eff, k * lags)); // AR part only
        let mut y_dep = Array2::zeros((n_eff, k));

        for i in 0..n_eff {
            let t_i = lags + i;
            y_dep.row_mut(i).assign(&y.row(t_i));
            for p in 0..lags {
                for j in 0..k {
                    z_ar[(i, p * k + j)] = y[(t_i - 1 - p, j)];
                }
            }
        }

        // Initialize: estimate common AR via OLS, then cluster residuals into K regimes
        let zt = z_ar.t();
        let ztz = zt.dot(&z_ar);
        let ztz_reg = &ztz + Array2::eye(k * lags) * 1e-8;
        let ztz_inv = ztz_reg.inv()?;
        let zty = zt.dot(&y_dep);
        let mut ar_coeffs: Array2<f64> = ztz_inv.dot(&zty); // (k*lags) x k

        let residuals = &y_dep - z_ar.dot(&ar_coeffs);

        // Initialize regimes by k-means-like clustering on residual norms
        let mut regime_intercepts = Array2::zeros((n_regimes, k));
        let mut regime_covariances = Array3::zeros((n_regimes, k, k));
        let mut assignments = vec![0usize; n_eff];

        // Simple init: sort by residual norm, split into K groups
        let mut norms: Vec<(usize, f64)> = (0..n_eff)
            .map(|i| (i, residuals.row(i).mapv(|v| v * v).sum().sqrt()))
            .collect();
        norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (idx, &(orig_i, _)) in norms.iter().enumerate() {
            assignments[orig_i] = idx * n_regimes / n_eff;
        }

        for r in 0..n_regimes {
            let mask: Vec<bool> = assignments.iter().map(|&a| a == r).collect();
            let n_r = mask.iter().filter(|&&b| b).count();
            if n_r == 0 {
                regime_intercepts.row_mut(r).assign(&Array1::zeros(k));
                regime_covariances
                    .slice_mut(ndarray::s![r, .., ..])
                    .fill(0.0);
                for j in 0..k {
                    regime_covariances[(r, j, j)] = 1.0;
                }
                continue;
            }
            // Mean of y_dep for this regime (approximate intercept)
            let mut mean = Array1::<f64>::zeros(k);
            for i in 0..n_eff {
                if mask[i] {
                    for j in 0..k {
                        mean[j] += residuals[(i, j)];
                    }
                }
            }
            mean /= n_r as f64;
            regime_intercepts.row_mut(r).assign(&mean);

            // Covariance
            let mut cov = Array2::<f64>::zeros((k, k));
            for i in 0..n_eff {
                if mask[i] {
                    for a in 0..k {
                        for b in 0..k {
                            cov[(a, b)] +=
                                (residuals[(i, a)] - mean[a]) * (residuals[(i, b)] - mean[b]);
                        }
                    }
                }
            }
            cov /= n_r as f64;
            regime_covariances
                .slice_mut(ndarray::s![r, .., ..])
                .assign(&cov);
        }

        // Transition matrix init: 0.9 diagonal
        let mut trans = Array2::ones((n_regimes, n_regimes)) * 0.1 / (n_regimes - 1) as f64;
        for i in 0..n_regimes {
            trans[(i, i)] = 0.9;
        }

        // EM iterations
        for _em in 0..50 {
            // E-step: forward-backward
            let (filtered, smoothed, ll) = Self::forward_backward(
                &y_dep,
                &z_ar,
                &regime_intercepts,
                &ar_coeffs,
                &regime_covariances,
                &trans,
                n_regimes,
            )?;

            // M-step: update parameters
            // Transition matrix
            for i in 0..n_regimes {
                let row_sum: f64 = (0..n_regimes).map(|j| trans[(i, j)]).sum();
                for j in 0..n_regimes {
                    let mut num = 0.0;
                    for t_i in 0..n_eff - 1 {
                        num += smoothed[(t_i, i)] * filtered[(t_i + 1, j)];
                    }
                    trans[(i, j)] = if row_sum > 1e-15 {
                        num / row_sum
                    } else {
                        1.0 / n_regimes as f64
                    };
                }
                // Normalize
                let s: f64 = (0..n_regimes).map(|j| trans[(i, j)]).sum();
                if s > 1e-15 {
                    for j in 0..n_regimes {
                        trans[(i, j)] /= s;
                    }
                }
            }

            // Update intercepts and AR coefficients
            // Weighted OLS: y_t = mu_{s_t} + A * z_t + eps
            // For simplicity, update intercepts as weighted means of (y - A*z)
            for r in 0..n_regimes {
                let mut weighted_res = Array1::<f64>::zeros(k);
                let mut weight_sum = 0.0;
                for i in 0..n_eff {
                    let w = smoothed[(i, r)];
                    let pred_i = z_ar.row(i).dot(&ar_coeffs);
                    for j in 0..k {
                        weighted_res[j] += (y_dep[(i, j)] - pred_i[j]) * w;
                    }
                    weight_sum += w;
                }
                if weight_sum > 1e-15 {
                    regime_intercepts
                        .row_mut(r)
                        .assign(&(&weighted_res / weight_sum));
                }

                // Update covariance
                let mut cov = Array2::<f64>::zeros((k, k));
                for i in 0..n_eff {
                    let w = smoothed[(i, r)];
                    let pred_i = z_ar.row(i).dot(&ar_coeffs);
                    for a in 0..k {
                        for b in 0..k {
                            let da = y_dep[(i, a)] - regime_intercepts[(r, a)] - pred_i[a];
                            let db = y_dep[(i, b)] - regime_intercepts[(r, b)] - pred_i[b];
                            cov[(a, b)] += da * db * w;
                        }
                    }
                }
                if weight_sum > 1e-15 {
                    cov /= weight_sum;
                    regime_covariances
                        .slice_mut(ndarray::s![r, .., ..])
                        .assign(&cov);
                }
            }

            // Update AR coefficients (common across regimes)
            // Weighted least squares with regime-specific intercepts
            let mut xtw = Array2::zeros((n_eff, k * lags));
            let mut ytw = Array2::zeros((n_eff, k));
            for i in 0..n_eff {
                for r in 0..n_regimes {
                    let w = smoothed[(i, r)].sqrt();
                    for j in 0..k * lags {
                        xtw[(i, j)] += w * z_ar[(i, j)];
                    }
                    for j in 0..k {
                        ytw[(i, j)] += w * (y_dep[(i, j)] - regime_intercepts[(r, j)]);
                    }
                }
            }
            let xtwt = xtw.t();
            let xtwx = xtwt.dot(&xtw);
            let xtwx_reg = &xtwx + Array2::eye(k * lags) * 1e-8;
            let xtwx_inv = xtwx_reg.inv()?;
            let xtwy = xtwt.dot(&ytw);
            ar_coeffs.assign(&xtwx_inv.dot(&xtwy));

            if ll.is_nan() {
                break;
            }
        }
        // Final forward-backward for clean probabilities
        let (filtered, smoothed, ll) = Self::forward_backward(
            &y_dep,
            &z_ar,
            &regime_intercepts,
            &ar_coeffs,
            &regime_covariances,
            &trans,
            n_regimes,
        )?;
        let log_likelihood = ll;

        let n_params =
            n_regimes * k + k * lags * k + n_regimes * k * (k + 1) / 2 + n_regimes * n_regimes;
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + (n_eff as f64) * n_params as f64;

        Ok(MsVarResult {
            regime_intercepts,
            ar_coeffs,
            regime_covariances,
            transition_matrix: trans,
            filtered_probs: filtered,
            smoothed_probs: smoothed,
            log_likelihood,
            aic,
            bic,
            n_obs: n_eff,
            n_vars: k,
            n_regimes,
            lags,
            var_names: names,
        })
    }

    fn forward_backward(
        y_dep: &Array2<f64>,
        z_ar: &Array2<f64>,
        intercepts: &Array2<f64>,
        ar_coeffs: &Array2<f64>,
        covariances: &Array3<f64>,
        trans: &Array2<f64>,
        n_regimes: usize,
    ) -> Result<(Array2<f64>, Array2<f64>, f64), GreenersError> {
        let n = y_dep.nrows();
        let k = y_dep.ncols();

        // Compute emission probabilities for each regime at each time
        let mut log_emissions = Array2::zeros((n, n_regimes));
        for r in 0..n_regimes {
            let cov = covariances.slice(ndarray::s![r, .., ..]).to_owned();
            let cov_inv = cov.inv().unwrap_or_else(|_| Array2::<f64>::eye(k));
            let det = cov.det().unwrap_or(1.0).max(1e-300);
            let log_det = det.ln();
            let pi_k = (2.0 * std::f64::consts::PI).powi(k as i32);

            for i in 0..n {
                let pred = z_ar.row(i).dot(ar_coeffs);
                let d: Array1<f64> =
                    &y_dep.row(i).to_owned() - &intercepts.row(r).to_owned() - &pred;
                let mahal = d.dot(&cov_inv.dot(&d));
                log_emissions[(i, r)] = -0.5 * (pi_k * det).ln() - 0.5 * mahal - log_det * 0.0;
            }
        }

        // Forward pass
        let mut log_alpha = Array2::zeros((n, n_regimes));
        let mut log_trans = Array2::zeros((n_regimes, n_regimes));
        for i in 0..n_regimes {
            for j in 0..n_regimes {
                log_trans[(i, j)] = trans[(i, j)].max(1e-300).ln();
            }
        }

        // Initialize
        for r in 0..n_regimes {
            log_alpha[(0, r)] = log_emissions[(0, r)] - (n_regimes as f64).ln();
        }

        for i in 1..n {
            for j in 0..n_regimes {
                let mut max_val = f64::NEG_INFINITY;
                for r in 0..n_regimes {
                    let val = log_alpha[(i - 1, r)] + log_trans[(r, j)];
                    if val > max_val {
                        max_val = val;
                    }
                }
                let mut sum = 0.0;
                for r in 0..n_regimes {
                    sum += (log_alpha[(i - 1, r)] + log_trans[(r, j)] - max_val).exp();
                }
                log_alpha[(i, j)] = log_emissions[(i, j)] + max_val + sum.ln();
            }
        }

        // Log-likelihood
        let mut max_ll = f64::NEG_INFINITY;
        for r in 0..n_regimes {
            if log_alpha[(n - 1, r)] > max_ll {
                max_ll = log_alpha[(n - 1, r)];
            }
        }
        let mut sum_exp = 0.0;
        for r in 0..n_regimes {
            sum_exp += (log_alpha[(n - 1, r)] - max_ll).exp();
        }
        let log_likelihood = max_ll + sum_exp.ln();

        // Backward pass
        let mut log_beta = Array2::zeros((n, n_regimes));
        for r in 0..n_regimes {
            log_beta[(n - 1, r)] = 0.0;
        }
        for i in (0..n - 1).rev() {
            for j in 0..n_regimes {
                let mut max_val = f64::NEG_INFINITY;
                for r in 0..n_regimes {
                    let val = log_beta[(i + 1, r)] + log_trans[(j, r)] + log_emissions[(i + 1, r)];
                    if val > max_val {
                        max_val = val;
                    }
                }
                let mut sum = 0.0;
                for r in 0..n_regimes {
                    sum += (log_beta[(i + 1, r)] + log_trans[(j, r)] + log_emissions[(i + 1, r)]
                        - max_val)
                        .exp();
                }
                log_beta[(i, j)] = max_val + sum.ln();
            }
        }

        // Smoothed and filtered probabilities
        let mut filtered = Array2::zeros((n, n_regimes));
        let mut smoothed = Array2::zeros((n, n_regimes));
        for i in 0..n {
            // Filtered: normalize log_alpha
            let mut max_a = f64::NEG_INFINITY;
            for r in 0..n_regimes {
                if log_alpha[(i, r)] > max_a {
                    max_a = log_alpha[(i, r)];
                }
            }
            let mut sum_a = 0.0;
            for r in 0..n_regimes {
                sum_a += (log_alpha[(i, r)] - max_a).exp();
            }
            for r in 0..n_regimes {
                filtered[(i, r)] = (log_alpha[(i, r)] - max_a).exp() / sum_a;
            }

            // Smoothed: log_alpha + log_beta
            let mut max_s = f64::NEG_INFINITY;
            for r in 0..n_regimes {
                let val = log_alpha[(i, r)] + log_beta[(i, r)];
                if val > max_s {
                    max_s = val;
                }
            }
            let mut sum_s = 0.0;
            for r in 0..n_regimes {
                sum_s += (log_alpha[(i, r)] + log_beta[(i, r)] - max_s).exp();
            }
            for r in 0..n_regimes {
                smoothed[(i, r)] = (log_alpha[(i, r)] + log_beta[(i, r)] - max_s).exp() / sum_s;
            }
        }

        Ok((filtered, smoothed, log_likelihood))
    }
}
