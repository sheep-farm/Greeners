//! Dynamic Factor Model (DFM) via state-space + EM.
//!
//! Extracts a small number of common factors from a large panel of
//! macroeconomic indicators. The factors follow a VAR(1) process:
//!
//! f_t = A * f_{t-1} + eta_t,  eta_t ~ N(0, Q)
//! x_t = Lambda * f_t + e_t,   e_t ~ N(0, R)
//!
//! where x_t is (n_series x 1), f_t is (n_factors x 1), Lambda is
//! (n_series x n_factors), A is (n_factors x n_factors).
//!
//! Estimation: EM algorithm (Dempster et al. 1977) with Kalman
//! filter/smoother. R is assumed diagonal for tractability.

use crate::linalg::LinalgDeterminant as _;
use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2, Array3};
use std::fmt;

/// Result of Dynamic Factor Model estimation.
#[derive(Debug)]
pub struct DfmResult {
    /// Extracted factors (T x n_factors)
    pub factors: Array2<f64>,
    /// Factor loadings (n_series x n_factors)
    pub loadings: Array2<f64>,
    /// Factor transition matrix A (n_factors x n_factors)
    pub factor_ar: Array2<f64>,
    /// Factor innovation covariance Q (n_factors x n_factors)
    pub factor_cov: Array2<f64>,
    /// Observation noise variances (diagonal of R) (n_series)
    pub obs_variances: Array1<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of series
    pub n_series: usize,
    /// Number of factors
    pub n_factors: usize,
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for DfmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Dynamic Factor Model (DFM) ")?;
        writeln!(f, "State-space + EM (Kalman filter/smoother)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Series:", self.n_series)?;
        writeln!(f, "{:<20} {:>12}", "Factors:", self.n_factors)?;
        writeln!(f, "{:<20} {:>12.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Factor loadings (Lambda):")?;
        write!(f, "  {:<12} ", "Series")?;
        for j in 0..self.n_factors {
            let label = format!("F{}", j + 1);
            write!(f, "{:>10} ", label)?;
        }
        writeln!(f)?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.n_series {
            let name = self.var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
            write!(f, "  {:<12} ", name)?;
            for j in 0..self.n_factors {
                write!(f, "{:>10.4} ", self.loadings[(i, j)])?;
            }
            writeln!(f)?;
        }

        writeln!(f, "\n  Factor transition matrix (A):")?;
        for i in 0..self.n_factors {
            let label = format!("F{}:", i + 1);
            write!(f, "  {:<10} ", label)?;
            for j in 0..self.n_factors {
                write!(f, "{:>10.4} ", self.factor_ar[(i, j)])?;
            }
            writeln!(f)?;
        }

        writeln!(f, "\n  Observation noise variances:")?;
        for i in 0..self.n_series {
            let name = self.var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
            writeln!(f, "  {name:<12} {:>10.6}", self.obs_variances[i])?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct DFM;

impl DFM {
    /// Estimate DFM via EM with Kalman filter/smoother.
    ///
    /// # Arguments
    /// * `x` - Data matrix (T x n_series), standardized internally
    /// * `n_factors` - Number of factors to extract
    /// * `max_iter` - Maximum EM iterations
    /// * `var_names` - Optional variable names
    pub fn fit(
        x: &Array2<f64>,
        n_factors: usize,
        max_iter: usize,
        var_names: Option<Vec<String>>,
    ) -> Result<DfmResult, GreenersError> {
        let t = x.nrows();
        let n = x.ncols();
        if t < n_factors + 5 {
            return Err(GreenersError::InvalidOperation(
                "DFM: too few observations".into(),
            ));
        }
        if n_factors == 0 {
            return Err(GreenersError::InvalidOperation(
                "DFM: n_factors must be >= 1".into(),
            ));
        }
        if n_factors >= n {
            return Err(GreenersError::InvalidOperation(
                "DFM: n_factors must be < n_series".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..n).map(|i| format!("x{}", i)).collect());

        // Standardize X
        let x_std = Self::standardize(x);

        // Initialize via PCA (power iteration)
        let (factors_init, loadings_init) = Self::pca_init(&x_std, n_factors)?;

        let mut factors = factors_init.clone();
        let mut loadings = loadings_init.clone();
        let mut factor_ar = Array2::eye(n_factors) * 0.5; // AR coefficient init
        let mut factor_cov = Array2::eye(n_factors) * 0.1;
        let mut obs_variances = Array1::ones(n) * 0.1;

        let mut log_likelihood = f64::NEG_INFINITY;

        for _em in 0..max_iter {
            // E-step: Kalman filter + smoother
            let (smoothed_factors, smoothed_cov, ll) = Self::kalman_smoother(
                &x_std,
                &loadings,
                &factor_ar,
                &factor_cov,
                &obs_variances,
                n_factors,
                n,
                t,
            )?;

            factors = smoothed_factors;
            log_likelihood = ll;

            // M-step: update parameters

            // 1. Update loadings: Lambda = (sum x_t * f_t') * (sum f_t * f_t')^{-1}
            let mut s_xf: Array2<f64> = Array2::zeros((n, n_factors));
            let mut s_ff: Array2<f64> = Array2::zeros((n_factors, n_factors));
            for tt in 0..t {
                let f_t = factors.row(tt);
                let x_t = x_std.row(tt);
                for i in 0..n {
                    for j in 0..n_factors {
                        s_xf[(i, j)] += x_t[i] * f_t[j];
                    }
                }
                for a in 0..n_factors {
                    for b in 0..n_factors {
                        s_ff[(a, b)] += f_t[a] * f_t[b] + smoothed_cov[(tt, a, b)];
                    }
                }
            }
            let s_ff_inv = (&s_ff + Array2::eye(n_factors) * 1e-8).inv()?;
            loadings = s_xf.dot(&s_ff_inv);

            // 2. Update obs_variances (diagonal R)
            for i in 0..n {
                let mut sum_sq = 0.0;
                for tt in 0..t {
                    let pred_i = (0..n_factors)
                        .map(|j| loadings[(i, j)] * factors[(tt, j)])
                        .sum::<f64>();
                    let resid = x_std[(tt, i)] - pred_i;
                    sum_sq += resid * resid;
                    // Add variance contribution from factor uncertainty
                    for a in 0..n_factors {
                        for b in 0..n_factors {
                            sum_sq +=
                                loadings[(i, a)] * loadings[(i, b)] * smoothed_cov[(tt, a, b)];
                        }
                    }
                }
                obs_variances[i] = (sum_sq / t as f64).max(1e-10);
            }

            // 3. Update factor AR matrix A
            // A = (sum f_t * f_{t-1}') * (sum f_{t-1} * f_{t-1}')^{-1}
            let mut s_f1f0: Array2<f64> = Array2::zeros((n_factors, n_factors));
            let mut s_f0f0: Array2<f64> = Array2::zeros((n_factors, n_factors));
            for tt in 1..t {
                let f_t = factors.row(tt);
                let f_prev = factors.row(tt - 1);
                for a in 0..n_factors {
                    for b in 0..n_factors {
                        s_f1f0[(a, b)] += f_t[a] * f_prev[b];
                        s_f0f0[(a, b)] += f_prev[a] * f_prev[b] + smoothed_cov[(tt - 1, a, b)];
                    }
                }
            }
            let s_f0f0_inv = (&s_f0f0 + Array2::eye(n_factors) * 1e-8).inv()?;
            factor_ar = s_f1f0.dot(&s_f0f0_inv);

            // 4. Update factor covariance Q
            let mut sum_q = Array2::zeros((n_factors, n_factors));
            for tt in 1..t {
                let f_t = factors.row(tt);
                let f_pred = factor_ar.dot(&factors.row(tt - 1));
                let d = &f_t - &f_pred;
                for a in 0..n_factors {
                    for b in 0..n_factors {
                        sum_q[(a, b)] += d[a] * d[b];
                    }
                }
            }
            factor_cov = &sum_q / (t - 1) as f64;
            // Add small ridge
            factor_cov = &factor_cov + Array2::eye(n_factors) * 1e-8;
        }

        let n_params = n * n_factors + n_factors * n_factors + n_factors * (n_factors + 1) / 2 + n;
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + (t as f64) * n_params as f64;

        Ok(DfmResult {
            factors,
            loadings,
            factor_ar,
            factor_cov,
            obs_variances,
            log_likelihood,
            aic,
            bic,
            n_obs: t,
            n_series: n,
            n_factors,
            var_names: names,
        })
    }

    fn standardize(x: &Array2<f64>) -> Array2<f64> {
        let t = x.nrows();
        let n = x.ncols();
        let mut x_std = Array2::zeros((t, n));
        for j in 0..n {
            let mean: f64 = x.column(j).mean().unwrap_or(0.0);
            let std_val: f64 = x.column(j).std(0.0).max(1e-10);
            for i in 0..t {
                x_std[(i, j)] = (x[(i, j)] - mean) / std_val;
            }
        }
        x_std
    }

    fn pca_init(
        x: &Array2<f64>,
        n_factors: usize,
    ) -> Result<(Array2<f64>, Array2<f64>), GreenersError> {
        let t = x.nrows();
        let n = x.ncols();

        // Covariance (n x n)
        let xt = x.t();
        let cov = xt.dot(x) / t as f64;

        // Power iteration with deflation
        let mut loadings = Array2::zeros((n, n_factors));
        let mut factors = Array2::zeros((t, n_factors));
        let mut remaining = cov.clone();

        for f in 0..n_factors {
            let mut v = Array1::ones(n) / (n as f64).sqrt();
            for _ in 0..100 {
                let v_new = remaining.dot(&v);
                let norm = v_new.mapv(|x| x * x).sum().sqrt().max(1e-10);
                v = v_new / norm;
            }
            let lambda = v.dot(&remaining.dot(&v));
            loadings.column_mut(f).assign(&v);
            // Factor scores
            for i in 0..t {
                factors[(i, f)] = x.row(i).dot(&v);
            }
            // Deflate
            for a in 0..n {
                for b in 0..n {
                    remaining[(a, b)] -= lambda * v[a] * v[b];
                }
            }
        }

        Ok((factors, loadings))
    }

    #[allow(clippy::too_many_arguments)]
    fn kalman_smoother(
        x: &Array2<f64>,
        loadings: &Array2<f64>,
        factor_ar: &Array2<f64>,
        factor_cov: &Array2<f64>,
        obs_variances: &Array1<f64>,
        n_factors: usize,
        n_series: usize,
        t: usize,
    ) -> Result<(Array2<f64>, Array3<f64>, f64), GreenersError> {
        // Forward filter
        let mut f_filt: Vec<Array1<f64>> = Vec::with_capacity(t);
        let mut p_filt: Vec<Array2<f64>> = Vec::with_capacity(t);
        let mut f_pred: Vec<Array1<f64>> = Vec::with_capacity(t);
        let mut p_pred: Vec<Array2<f64>> = Vec::with_capacity(t);

        let mut f_t = Array1::zeros(n_factors);
        let mut p_t = Array2::eye(n_factors) * 1.0;
        let mut ll = 0.0_f64;

        for tt in 0..t {
            // Predict
            let f_p = factor_ar.dot(&f_t);
            let p_p: Array2<f64> = factor_ar.dot(&p_t).dot(&factor_ar.t()) + factor_cov;

            // Innovation
            let x_t = x.row(tt);
            let y_pred = loadings.dot(&f_p);
            let innov = &x_t - &y_pred;

            // Innovation covariance: S = Lambda * P_pred * Lambda' + R
            let mut s_mat: Array2<f64> = loadings.dot(&p_p).dot(&loadings.t());
            for i in 0..n_series {
                s_mat[(i, i)] += obs_variances[i];
            }
            let s_inv = (&s_mat + Array2::eye(n_series) * 1e-8).inv()?;

            // Kalman gain: K = P_pred * Lambda' * S^{-1}
            let k_gain: Array2<f64> = p_p.dot(&loadings.t()).dot(&s_inv);

            // Update
            f_t = &f_p + k_gain.dot(&innov);
            p_t = &p_p - k_gain.dot(loadings).dot(&p_p);

            f_filt.push(f_t.clone());
            p_filt.push(p_t.clone());
            f_pred.push(f_p);
            p_pred.push(p_p);

            // Log-likelihood
            let s_det = s_mat.det().unwrap_or(1e-300).max(1e-300);
            let mahal = innov.dot(&s_inv.dot(&innov));
            ll += -0.5 * (n_series as f64 * (2.0 * std::f64::consts::PI).ln() + s_det.ln() + mahal);
        }

        // Backward smoother (RTS)
        let mut smoothed_factors = Array2::zeros((t, n_factors));
        let mut smoothed_cov = Array3::zeros((t, n_factors, n_factors));

        let mut f_smooth = f_filt[t - 1].clone();
        let mut p_smooth = p_filt[t - 1].clone();
        smoothed_factors.row_mut(t - 1).assign(&f_smooth);
        smoothed_cov
            .slice_mut(ndarray::s![t - 1, .., ..])
            .assign(&p_smooth);

        for tt in (0..t - 1).rev() {
            // J = P_filt[tt] * A' * P_pred[tt+1]^{-1}
            let p_pred_inv = (&p_pred[tt + 1] + Array2::eye(n_factors) * 1e-8).inv()?;
            let j_mat = p_filt[tt].dot(&factor_ar.t()).dot(&p_pred_inv);

            f_smooth = &f_filt[tt] + j_mat.dot(&(&f_smooth - &f_pred[tt + 1]));
            p_smooth = &p_filt[tt] + j_mat.dot(&(&p_smooth - &p_pred[tt + 1])).dot(&j_mat.t());

            smoothed_factors.row_mut(tt).assign(&f_smooth);
            smoothed_cov
                .slice_mut(ndarray::s![tt, .., ..])
                .assign(&p_smooth);
        }

        Ok((smoothed_factors, smoothed_cov, ll))
    }
}
