//! DCC-GARCH (Dynamic Conditional Correlation GARCH).
//!
//! Engle (2002). Models time-varying correlations in multivariate
//! volatility. Two-step estimation:
//!
//! 1. Estimate univariate GARCH(1,1) for each series
//! 2. Model the conditional correlation matrix via a GARCH-like
//!    process on the standardized residuals:
//!
//!    Q_t = (1 - a - b) * Q_bar + a * eps_{t-1} * eps_{t-1}' + b * Q_{t-1}
//!    R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
//!
//! where Q_bar is the unconditional correlation of standardized
//! residuals, and R_t is the dynamic conditional correlation matrix.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2, Array3};
use std::fmt;

/// Result of DCC-GARCH estimation.
#[derive(Debug)]
pub struct DccGarchResult {
    /// DCC parameters: alpha (a), beta (b)
    pub dcc_alpha: f64,
    pub dcc_beta: f64,
    /// Univariate GARCH parameters per series: [omega, alpha, beta] x k
    pub garch_params: Array2<f64>,
    /// Conditional volatilities (T x k)
    pub conditional_vols: Array2<f64>,
    /// Dynamic conditional correlation matrices (T x k x k)
    pub dcc_correlations: Array3<f64>,
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
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for DccGarchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " DCC-GARCH ")?;
        writeln!(f, "Engle (2002) — Dynamic Conditional Correlation")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Series:", self.n_series)?;
        writeln!(f, "{:<20} {:>12.6}", "DCC alpha:", self.dcc_alpha)?;
        writeln!(f, "{:<20} {:>12.6}", "DCC beta:", self.dcc_beta)?;
        writeln!(f, "{:<20} {:>12.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Univariate GARCH(1,1) parameters:")?;
        writeln!(
            f,
            "  {:<10} {:>10} {:>10} {:>10}",
            "Series", "omega", "alpha", "beta"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.n_series {
            let name = self.var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
            writeln!(
                f,
                "  {:<10} {:>10.6} {:>10.6} {:>10.6}",
                name,
                self.garch_params[(i, 0)],
                self.garch_params[(i, 1)],
                self.garch_params[(i, 2)]
            )?;
        }

        // Show correlation at selected periods
        let t_mid = self.n_obs / 2;
        writeln!(f, "\n  Conditional correlations (selected periods):")?;
        for &t in &[0, t_mid, self.n_obs - 1] {
            if t < self.dcc_correlations.dim().0 {
                writeln!(f, "  t={:<6}", t + 1)?;
                for i in 0..self.n_series {
                    let mut row = format!(
                        "    {}:",
                        self.var_names.get(i).map(|s| s.as_str()).unwrap_or("?")
                    );
                    for j in 0..self.n_series {
                        row.push_str(&format!(" {:>8.4}", self.dcc_correlations[(t, i, j)]));
                    }
                    writeln!(f, "{row}")?;
                }
            }
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct DCCGARCH;

impl DCCGARCH {
    /// Estimate DCC-GARCH(1,1).
    ///
    /// # Arguments
    /// * `returns` - Return matrix (T x k)
    /// * `var_names` - Optional variable names
    pub fn fit(
        returns: &Array2<f64>,
        var_names: Option<Vec<String>>,
    ) -> Result<DccGarchResult, GreenersError> {
        let t = returns.nrows();
        let k = returns.ncols();
        if t < 20 {
            return Err(GreenersError::InvalidOperation(
                "DCC-GARCH: need at least 20 observations".into(),
            ));
        }
        if k < 2 {
            return Err(GreenersError::InvalidOperation(
                "DCC-GARCH: need at least 2 series".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("r{}", i)).collect());

        // Step 1: Estimate univariate GARCH(1,1) for each series
        let mut garch_params = Array2::zeros((k, 3));
        let mut conditional_vols = Array2::zeros((t, k));
        let mut std_resids = Array2::zeros((t, k));

        for j in 0..k {
            let r_j = returns.column(j).to_owned();
            let (params, vols, sresid) = Self::garch11(&r_j)?;
            garch_params[(j, 0)] = params[0]; // omega
            garch_params[(j, 1)] = params[1]; // alpha
            garch_params[(j, 2)] = params[2]; // beta
            for i in 0..t {
                conditional_vols[(i, j)] = vols[i];
                std_resids[(i, j)] = sresid[i];
            }
        }

        // Step 2: Estimate DCC parameters
        // Q_bar = unconditional correlation of standardized residuals
        let mut q_bar = Array2::zeros((k, k));
        for i in 0..t {
            let s = std_resids.row(i);
            for a in 0..k {
                for b in 0..k {
                    q_bar[(a, b)] += s[a] * s[b];
                }
            }
        }
        q_bar /= t as f64;

        // Grid search over (alpha, beta) with constraint alpha + beta < 1
        let mut best_alpha = 0.01_f64;
        let mut best_beta = 0.95_f64;
        let mut best_ll = f64::NEG_INFINITY;

        let n_grid = 15;
        for i in 0..n_grid {
            for j in 0..n_grid {
                let alpha = 0.01 + 0.48 * i as f64 / (n_grid - 1) as f64;
                let beta = 0.01 + 0.48 * j as f64 / (n_grid - 1) as f64;
                if alpha + beta >= 0.99 {
                    continue;
                }
                let ll = Self::dcc_loglik(&std_resids, &q_bar, alpha, beta, k, t);
                if ll > best_ll {
                    best_ll = ll;
                    best_alpha = alpha;
                    best_beta = beta;
                }
            }
        }

        // Compute dynamic correlations with best parameters
        let dcc_correlations = Self::compute_dcc(&std_resids, &q_bar, best_alpha, best_beta, k, t);

        // Total log-likelihood (GARCH + DCC)
        let mut garch_ll = 0.0_f64;
        for j in 0..k {
            for i in 0..t {
                let vol = conditional_vols[(i, j)].max(1e-10);
                let r = returns[(i, j)];
                garch_ll +=
                    -0.5 * (2.0 * std::f64::consts::PI).ln() - vol.ln() - 0.5 * (r / vol).powi(2);
            }
        }
        let total_ll = garch_ll + best_ll;

        let n_params = k * 3 + 2; // GARCH params + DCC alpha, beta
        let aic = -2.0 * total_ll + 2.0 * n_params as f64;
        let bic = -2.0 * total_ll + (t as f64) * n_params as f64;

        Ok(DccGarchResult {
            dcc_alpha: best_alpha,
            dcc_beta: best_beta,
            garch_params,
            conditional_vols,
            dcc_correlations,
            log_likelihood: total_ll,
            aic,
            bic,
            n_obs: t,
            n_series: k,
            var_names: names,
        })
    }

    /// Estimate univariate GARCH(1,1) via MLE (grid search).
    #[allow(clippy::type_complexity)]
    fn garch11(r: &Array1<f64>) -> Result<([f64; 3], Vec<f64>, Vec<f64>), GreenersError> {
        let t = r.len();
        let var_init = r.mapv(|v| v * v).mean().unwrap_or(0.01);

        // Grid search over (omega, alpha, beta)
        let mut best_params = [0.01_f64, 0.05, 0.90];
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_vols = vec![var_init; t];

        let n_grid = 8;
        for oi in 0..n_grid {
            let omega = 0.001 + 0.1 * oi as f64 / (n_grid - 1) as f64 * var_init;
            for ai in 0..n_grid {
                let alpha = 0.01 + 0.3 * ai as f64 / (n_grid - 1) as f64;
                for bi in 0..n_grid {
                    let beta = 0.5 + 0.48 * bi as f64 / (n_grid - 1) as f64;
                    if alpha + beta >= 0.99 {
                        continue;
                    }
                    let mut vols = vec![var_init; t];
                    let mut ll = 0.0_f64;
                    for i in 0..t {
                        if i > 0 {
                            vols[i] = omega + alpha * r[i - 1] * r[i - 1] + beta * vols[i - 1];
                        }
                        let vol = vols[i].max(1e-10);
                        ll += -0.5 * (2.0 * std::f64::consts::PI).ln()
                            - vol.ln()
                            - 0.5 * (r[i] * r[i]) / vol;
                    }
                    if ll > best_ll {
                        best_ll = ll;
                        best_params = [omega, alpha, beta];
                        best_vols = vols;
                    }
                }
            }
        }

        let std_resid: Vec<f64> = (0..t)
            .map(|i| {
                let vol = best_vols[i].max(1e-10);
                r[i] / vol.sqrt()
            })
            .collect();

        Ok((best_params, best_vols, std_resid))
    }

    /// DCC log-likelihood for given (alpha, beta).
    fn dcc_loglik(
        std_resids: &Array2<f64>,
        q_bar: &Array2<f64>,
        alpha: f64,
        beta: f64,
        k: usize,
        t: usize,
    ) -> f64 {
        let mut q_prev = q_bar.clone();
        let mut ll = 0.0_f64;

        for i in 0..t {
            // Q_t = (1-a-b)*Q_bar + a*eps*eps' + b*Q_{t-1}
            let s = std_resids.row(i);
            let mut q_t = Array2::<f64>::zeros((k, k));
            for a in 0..k {
                for b in 0..k {
                    q_t[(a, b)] = (1.0 - alpha - beta) * q_bar[(a, b)]
                        + alpha * s[a] * s[b]
                        + beta * q_prev[(a, b)];
                }
            }

            // R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
            let mut diag_inv_sqrt = Array1::zeros(k);
            for a in 0..k {
                diag_inv_sqrt[a] = 1.0 / q_t[(a, a)].sqrt().max(1e-10);
            }
            let mut r_t = Array2::zeros((k, k));
            for a in 0..k {
                for b in 0..k {
                    r_t[(a, b)] = diag_inv_sqrt[a] * q_t[(a, b)] * diag_inv_sqrt[b];
                }
            }

            // Log-likelihood: -0.5 * (log|R_t| + eps' * R_t^{-1} * eps)
            let r_det = Self::det_2d(&r_t).max(1e-300);
            let r_inv = Self::inv_2d(&r_t, k);
            let eps = s.to_owned();
            let quad = eps.dot(&r_inv.dot(&eps));
            ll += -0.5 * (r_det.ln() + quad);

            q_prev = q_t;
        }

        ll
    }

    /// Compute DCC correlation matrices.
    fn compute_dcc(
        std_resids: &Array2<f64>,
        q_bar: &Array2<f64>,
        alpha: f64,
        beta: f64,
        k: usize,
        t: usize,
    ) -> Array3<f64> {
        let mut q_prev = q_bar.clone();
        let mut correlations = Array3::zeros((t, k, k));

        for i in 0..t {
            let s = std_resids.row(i);
            let mut q_t = Array2::<f64>::zeros((k, k));
            for a in 0..k {
                for b in 0..k {
                    q_t[(a, b)] = (1.0 - alpha - beta) * q_bar[(a, b)]
                        + alpha * s[a] * s[b]
                        + beta * q_prev[(a, b)];
                }
            }

            let mut diag_inv_sqrt = Array1::zeros(k);
            for a in 0..k {
                diag_inv_sqrt[a] = 1.0 / q_t[(a, a)].sqrt().max(1e-10);
            }
            for a in 0..k {
                for b in 0..k {
                    correlations[(i, a, b)] = diag_inv_sqrt[a] * q_t[(a, b)] * diag_inv_sqrt[b];
                }
            }

            q_prev = q_t;
        }

        correlations
    }

    fn det_2d(m: &Array2<f64>) -> f64 {
        let n = m.nrows();
        if n == 1 {
            return m[(0, 0)];
        }
        if n == 2 {
            return m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
        }
        // LU decomposition
        let mut a = m.clone();
        let mut det = 1.0;
        for i in 0..n {
            let mut max_row = i;
            for r in (i + 1)..n {
                if a[(r, i)].abs() > a[(max_row, i)].abs() {
                    max_row = r;
                }
            }
            if max_row != i {
                for j in 0..n {
                    let tmp = a[(i, j)];
                    a[(i, j)] = a[(max_row, j)];
                    a[(max_row, j)] = tmp;
                }
                det = -det;
            }
            if a[(i, i)].abs() < 1e-300 {
                return 0.0;
            }
            det *= a[(i, i)];
            for r in (i + 1)..n {
                let factor = a[(r, i)] / a[(i, i)];
                for j in i..n {
                    a[(r, j)] -= factor * a[(i, j)];
                }
            }
        }
        det
    }

    fn inv_2d(m: &Array2<f64>, k: usize) -> Array2<f64> {
        (m.clone() + Array2::<f64>::eye(k) * 1e-8)
            .inv()
            .unwrap_or_else(|_| Array2::eye(k))
    }
}
