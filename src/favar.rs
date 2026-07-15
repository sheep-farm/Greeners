//! FAVAR (Factor-Augmented VAR).
//!
//! Combines factor analysis with VAR. A large set of macroeconomic
//! indicators is summarized into a few common factors via PCA, then
//! a VAR is estimated on the factors plus observed policy variables.
//!
//! y_t = [F_t', R_t']' where F_t are extracted factors and R_t are
//! observed variables (e.g., federal funds rate).
//!
//! F_t = Lambda_f * F_{t-1} + ... + eps_t  (VAR part)
//!
//! Estimation: two-step (Bernanke et al. 2005):
//!   1. PCA to extract factors from large panel X
//!   2. VAR on [F, R]

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2, Array3, Axis};
use std::fmt;

/// Result of FAVAR estimation.
#[derive(Debug)]
pub struct FavarResult {
    /// Extracted factors (T x n_factors)
    pub factors: Array2<f64>,
    /// Factor loadings (n_series x n_factors)
    pub loadings: Array2<f64>,
    /// VAR coefficients on [F, R] ((1 + k*lags) x k) where k = n_factors + n_observed
    pub var_coeffs: Array2<f64>,
    /// VAR residual covariance (k x k)
    pub var_sigma: Array2<f64>,
    /// IRF (steps x k x k)
    pub irf: Array3<f64>,
    /// Variance explained by each factor
    pub variance_explained: Array1<f64>,
    /// Total variance explained ratio
    pub total_variance_explained: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of series in panel
    pub n_series: usize,
    /// Number of factors extracted
    pub n_factors: usize,
    /// Number of observed policy variables
    pub n_observed: usize,
    /// VAR lags
    pub lags: usize,
    /// Factor names
    pub factor_names: Vec<String>,
    /// Observed variable names
    pub observed_names: Vec<String>,
}

impl fmt::Display for FavarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " FAVAR (Factor-Augmented VAR) ")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Series (panel):", self.n_series)?;
        writeln!(f, "{:<20} {:>12}", "Factors:", self.n_factors)?;
        writeln!(f, "{:<20} {:>12}", "Observed vars:", self.n_observed)?;
        writeln!(f, "{:<20} {:>12}", "VAR lags:", self.lags)?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "Total var explained:", self.total_variance_explained
        )?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Variance explained by factor:")?;
        for i in 0..self.n_factors {
            writeln!(
                f,
                "  Factor {:<6} {:>10.4}%  ({})",
                i + 1,
                self.variance_explained[i] * 100.0,
                self.factor_names.get(i).map(|s| s.as_str()).unwrap_or("")
            )?;
        }

        writeln!(f, "\n  VAR coefficients (first row = intercept):")?;
        let k = self.n_factors + self.n_observed;
        let mut header = String::from("  {:<12} ");
        for j in 0..k {
            let name = if j < self.n_factors {
                self.factor_names.get(j).map(|s| s.as_str()).unwrap_or("?")
            } else {
                self.observed_names
                    .get(j - self.n_factors)
                    .map(|s| s.as_str())
                    .unwrap_or("?")
            };
            header.push_str(&format!("{:>10} ", name));
        }
        writeln!(f, "{header}")?;
        for i in 0..self.var_coeffs.nrows() {
            let label = if i == 0 {
                "const".to_string()
            } else {
                let lag = (i - 1) / k + 1;
                let var_idx = (i - 1) % k;
                let name = if var_idx < self.n_factors {
                    self.factor_names
                        .get(var_idx)
                        .map(|s| s.as_str())
                        .unwrap_or("?")
                } else {
                    self.observed_names
                        .get(var_idx - self.n_factors)
                        .map(|s| s.as_str())
                        .unwrap_or("?")
                };
                format!("L{lag}.{name}")
            };
            let mut row = format!("  {:<12} ", label);
            for j in 0..k {
                row.push_str(&format!("{:>10.4} ", self.var_coeffs[(i, j)]));
            }
            writeln!(f, "{row}")?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct FAVAR;

impl FAVAR {
    /// Estimate FAVAR via two-step PCA + VAR.
    ///
    /// # Arguments
    /// * `x` - Panel data (T x n_series), standardized internally
    /// * `observed` - Observed policy variables (T x n_observed)
    /// * `n_factors` - Number of factors to extract
    /// * `lags` - VAR lag order
    /// * `irf_steps` - IRF horizon (0 = no IRF)
    pub fn fit(
        x: &Array2<f64>,
        observed: &Array2<f64>,
        n_factors: usize,
        lags: usize,
        irf_steps: usize,
        factor_names: Option<Vec<String>>,
        observed_names: Option<Vec<String>>,
    ) -> Result<FavarResult, GreenersError> {
        let t = x.nrows();
        let n_series = x.ncols();
        let n_obs = observed.ncols();
        if observed.nrows() != t {
            return Err(GreenersError::ShapeMismatch(
                "FAVAR: x and observed must have same number of rows".into(),
            ));
        }
        if n_factors == 0 || lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "FAVAR: n_factors and lags must be >= 1".into(),
            ));
        }
        if t < (lags + 1) * 3 {
            return Err(GreenersError::InvalidOperation(
                "FAVAR: too few observations".into(),
            ));
        }

        // Step 1: Standardize X and extract factors via PCA
        let x_std = Self::standardize(x);
        let (factors, loadings, variance_explained, total_var) = Self::pca(&x_std, n_factors)?;

        // Step 2: Build combined matrix [F, R] and estimate VAR
        let k = n_factors + n_obs;
        let mut combined = Array2::zeros((t, k));
        for i in 0..t {
            for j in 0..n_factors {
                combined[(i, j)] = factors[(i, j)];
            }
            for j in 0..n_obs {
                combined[(i, n_factors + j)] = observed[(i, j)];
            }
        }

        // VAR estimation via OLS
        let n_eff = t - lags;
        let mut z = Array2::zeros((n_eff, 1 + k * lags));
        let mut y_dep = Array2::zeros((n_eff, k));
        for i in 0..n_eff {
            let t_i = lags + i;
            y_dep.row_mut(i).assign(&combined.row(t_i));
            z[(i, 0)] = 1.0;
            for p in 0..lags {
                for j in 0..k {
                    z[(i, 1 + p * k + j)] = combined[(t_i - 1 - p, j)];
                }
            }
        }

        let zt = z.t();
        let ztz = zt.dot(&z);
        let ztz_reg = &ztz + Array2::eye(1 + k * lags) * 1e-8;
        let ztz_inv = ztz_reg.inv()?;
        let zty = zt.dot(&y_dep);
        let var_coeffs: Array2<f64> = ztz_inv.dot(&zty);

        let residuals = &y_dep - z.dot(&var_coeffs);
        let var_sigma: Array2<f64> = residuals.t().dot(&residuals) / n_eff as f64;

        // IRF via Cholesky
        let irf = if irf_steps > 0 {
            Self::compute_irf(&var_coeffs, &var_sigma, lags, k, irf_steps)?
        } else {
            Array3::zeros((0, k, k))
        };

        // AIC/BIC
        let n_params = (1 + k * lags) * k;
        let log_det_sigma = var_sigma
            .inv()
            .ok()
            .map(|inv| {
                let det = inv.mapv(|v| 1.0 / v).product();
                det.ln()
            })
            .unwrap_or(0.0);
        let ll = -(n_eff as f64) / 2.0
            * (k as f64 * (2.0 * std::f64::consts::PI).ln() + log_det_sigma)
            - 0.5 * residuals.mapv(|v| v * v).sum();
        let aic = -2.0 * ll + 2.0 * n_params as f64;
        let bic = -2.0 * ll + (n_eff as f64) * n_params as f64;

        let f_names =
            factor_names.unwrap_or_else(|| (0..n_factors).map(|i| format!("F{}", i + 1)).collect());
        let o_names =
            observed_names.unwrap_or_else(|| (0..n_obs).map(|i| format!("R{}", i + 1)).collect());

        Ok(FavarResult {
            factors,
            loadings,
            var_coeffs,
            var_sigma,
            irf,
            variance_explained,
            total_variance_explained: total_var,
            aic,
            bic,
            n_obs: n_eff,
            n_series,
            n_factors,
            n_observed: n_obs,
            lags,
            factor_names: f_names,
            observed_names: o_names,
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

    #[allow(clippy::type_complexity)]
    fn pca(
        x: &Array2<f64>,
        n_factors: usize,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, f64), GreenersError> {
        let t = x.nrows();
        let n = x.ncols();

        // Covariance matrix (n x n)
        let xt = x.t();
        let cov = xt.dot(x) / t as f64;

        // Power iteration for top n_factors eigenvectors
        let mut factors = Array2::zeros((t, n_factors));
        let mut loadings = Array2::zeros((n, n_factors));
        let mut eigenvalues = Array1::zeros(n_factors);

        let mut remaining = cov.clone();
        let total_variance: f64 = cov.diag().sum();
        let mut explained_sum = 0.0;

        for f in 0..n_factors {
            let (eigval, eigvec) = Self::power_iteration(&remaining, 100);
            eigenvalues[f] = eigval;
            loadings.column_mut(f).assign(&eigvec);
            // Factor scores: X * eigvec
            for i in 0..t {
                factors[(i, f)] = x.row(i).dot(&eigvec);
            }
            // Deflate
            remaining = &remaining
                - &eigvec
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&eigvec.clone().insert_axis(Axis(0)))
                    * eigval;
            explained_sum += eigval;
        }

        let variance_explained = eigenvalues.mapv(|v| v / total_variance.max(1e-10));
        let total_var = explained_sum / total_variance.max(1e-10);

        Ok((factors, loadings, variance_explained, total_var))
    }

    fn power_iteration(mat: &Array2<f64>, n_iter: usize) -> (f64, Array1<f64>) {
        let n = mat.ncols();
        let mut v = Array1::ones(n) / (n as f64).sqrt();
        for _ in 0..n_iter {
            let v_new = mat.dot(&v);
            let norm = v_new.mapv(|x| x * x).sum().sqrt().max(1e-10);
            v = v_new / norm;
        }
        let lambda = v.dot(&mat.dot(&v));
        (lambda, v)
    }

    fn compute_irf(
        coeffs: &Array2<f64>,
        sigma: &Array2<f64>,
        lags: usize,
        k: usize,
        steps: usize,
    ) -> Result<Array3<f64>, GreenersError> {
        // Cholesky decomposition of sigma (simplified: use eigenvalue approach)
        let chol = Self::cholesky_simple(sigma, k)?;

        // Build companion form
        let mut phi = Array2::zeros((k * lags, k * lags));
        for p in 0..lags {
            for j in 0..k {
                for i in 0..k {
                    phi[(p * k + j, i)] = coeffs[(1 + p * k + i, j)];
                }
            }
        }
        if lags > 1 {
            for p in 1..lags {
                for j in 0..k {
                    phi[(p * k + j, (p - 1) * k + j)] = 1.0;
                }
            }
        }

        // IRF
        let mut irf = Array3::zeros((steps, k, k));
        let mut phi_power = Array2::eye(k * lags);
        for h in 0..steps {
            // Phi_h (top k rows)
            let phi_h = phi_power.slice(ndarray::s![0..k, 0..k]).to_owned();
            // IRF[h] = Phi_h * Chol
            irf.slice_mut(ndarray::s![h, .., ..])
                .assign(&phi_h.dot(&chol));
            phi_power = phi.dot(&phi_power);
        }

        Ok(irf)
    }

    fn cholesky_simple(sigma: &Array2<f64>, k: usize) -> Result<Array2<f64>, GreenersError> {
        let mut chol = Array2::zeros((k, k));
        for i in 0..k {
            for j in 0..=i {
                let mut sum = 0.0;
                for l in 0..j {
                    sum += chol[(i, l)] * chol[(j, l)];
                }
                if i == j {
                    let val = sigma[(i, i)] - sum;
                    if val < 0.0 {
                        chol[(i, j)] = 1e-10;
                    } else {
                        chol[(i, j)] = val.sqrt();
                    }
                } else {
                    let denom = chol[(j, j)].max(1e-10);
                    chol[(i, j)] = (sigma[(i, j)] - sum) / denom;
                }
            }
        }
        Ok(chol)
    }
}
