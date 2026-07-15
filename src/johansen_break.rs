//! Johansen cointegration test with structural breaks.
//!
//! Extends the standard Johansen test to allow for known structural
//! breaks in the deterministic components (intercept/trend). Based on
//! Johansen et al. (2000) and Giles & Godwin (2012).
//!
//! The VECM with breaks:
//!   Delta y_t = Pi * y_{t-1} + sum Gamma_j * Delta y_{t-j}
//!               + mu_0 + mu_1 * t + delta * D_t + eps_t
//!
//! where D_t are break dummies (shift in level or trend).
//!
//! Trace and Lambda-max statistics are computed for r = 0, 1, ..., k-1.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2, Axis};
use std::fmt;

/// Result of Johansen test with structural breaks.
#[derive(Debug)]
pub struct JohansenBreakResult {
    /// Trace statistics for each rank r = 0, 1, ..., k-1
    pub trace_stats: Array1<f64>,
    /// Lambda-max statistics for each rank
    pub lambda_max_stats: Array1<f64>,
    /// Critical values (5%) for trace test
    pub trace_cv_5: Array1<f64>,
    /// Critical values (5%) for lambda-max test
    pub lambda_max_cv_5: Array1<f64>,
    /// Eigenvalues of the cointegration matrix
    pub eigenvalues: Array1<f64>,
    /// Cointegrating vectors (eigenvectors), (k x r) for the selected rank
    pub cointegrating_vectors: Array2<f64>,
    /// Selected cointegration rank (at 5% level)
    pub cointegration_rank: usize,
    /// Number of variables
    pub n_vars: usize,
    /// VAR lag order
    pub lags: usize,
    /// Number of structural breaks
    pub n_breaks: usize,
    /// Break points (time indices)
    pub break_points: Vec<usize>,
    /// Number of observations
    pub n_obs: usize,
}

impl fmt::Display for JohansenBreakResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            " Johansen Cointegration Test (with Structural Breaks) "
        )?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Structural breaks:", self.n_breaks)?;
        if self.n_breaks > 0 {
            let bp: Vec<String> = self.break_points.iter().map(|b| b.to_string()).collect();
            writeln!(f, "{:<20} {}", "Break points:", bp.join(", "))?;
        }

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<6} {:>12} {:>12} {:>12} {:>12} {:>10}",
            "Rank", "Trace", "Trace 5%", "L-max", "L-max 5%", "Sig?"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for r in 0..self.n_vars {
            let sig = if self.trace_stats[r] > self.trace_cv_5[r] {
                "**"
            } else {
                ""
            };
            writeln!(
                f,
                "{:<6} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>10}",
                r,
                self.trace_stats[r],
                self.trace_cv_5[r],
                self.lambda_max_stats[r],
                self.lambda_max_cv_5[r],
                sig
            )?;
        }

        writeln!(
            f,
            "\n  Selected cointegration rank: {} (at 5% level)",
            self.cointegration_rank
        )?;

        if self.cointegration_rank > 0 {
            writeln!(f, "\n  Cointegrating vectors:")?;
            for j in 0..self.cointegration_rank {
                let mut row = format!("  Vector {j}: ");
                for i in 0..self.n_vars {
                    row.push_str(&format!("{:>10.4} ", self.cointegrating_vectors[(i, j)]));
                }
                writeln!(f, "{row}")?;
            }
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct JohansenBreak;

impl JohansenBreak {
    /// Johansen cointegration test with structural breaks.
    ///
    /// # Arguments
    /// * `y` - Data matrix (T x k)
    /// * `lags` - VAR lag order for the VECM
    /// * `break_points` - Vector of break indices (time periods where breaks occur)
    pub fn fit(
        y: &Array2<f64>,
        lags: usize,
        break_points: &[usize],
    ) -> Result<JohansenBreakResult, GreenersError> {
        let t = y.nrows();
        let k = y.ncols();
        if t < (lags + k + 1) * 2 {
            return Err(GreenersError::InvalidOperation(
                "Johansen: too few observations".into(),
            ));
        }
        if lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "Johansen: lags must be >= 1".into(),
            ));
        }

        let n_eff = t - lags - 1;
        let n_breaks = break_points.len();

        // Build break dummies: shift dummies (1 from break point onward)
        let mut d_shift = Array2::zeros((t, n_breaks));
        for (b, &bp) in break_points.iter().enumerate() {
            for i in bp..t {
                d_shift[(i, b)] = 1.0;
            }
        }

        // Build differenced data
        let mut dy = Array2::zeros((n_eff, k));
        let mut y_lag1 = Array2::zeros((n_eff, k)); // y_{t-1} (level)
        let mut dy_lags = Array2::zeros((n_eff, k * (lags - 1).max(1))); // lagged differences
        let mut d_shift_reg = Array2::zeros((n_eff, n_breaks));

        for i in 0..n_eff {
            let t_i = lags + 1 + i;
            dy.row_mut(i).assign(&(&y.row(t_i) - &y.row(t_i - 1)));
            y_lag1.row_mut(i).assign(&y.row(t_i - 1));
            for p in 1..lags {
                for j in 0..k {
                    dy_lags[(i, (p - 1) * k + j)] = y[(t_i - p, j)] - y[(t_i - p - 1, j)];
                }
            }
            for b in 0..n_breaks {
                d_shift_reg[(i, b)] = d_shift[(t_i, b)];
            }
        }

        // Regressors: [constant, dy_lags, break dummies]
        let n_reg = 1 + k * (lags - 1).max(1) + n_breaks;
        let mut z = Array2::zeros((n_eff, n_reg));
        for i in 0..n_eff {
            z[(i, 0)] = 1.0;
            for j in 0..k * (lags - 1).max(1) {
                z[(i, 1 + j)] = dy_lags[(i, j)];
            }
            for b in 0..n_breaks {
                z[(i, 1 + k * (lags - 1).max(1) + b)] = d_shift_reg[(i, b)];
            }
        }

        // Residuals from regressing dy on Z (R0) and y_lag1 on Z (R1)
        let zt = z.t();
        let ztz = zt.dot(&z);
        let ztz_reg = &ztz + Array2::eye(n_reg) * 1e-8;
        let ztz_inv = ztz_reg.inv()?;

        let r0 = &dy - z.dot(&ztz_inv.dot(&zt.dot(&dy)));
        let r1 = &y_lag1 - z.dot(&ztz_inv.dot(&zt.dot(&y_lag1)));

        // S_ij = R_i' * R_j / n_eff
        let s00 = r0.t().dot(&r0) / n_eff as f64;
        let s01 = r0.t().dot(&r1) / n_eff as f64;
        let s10 = r1.t().dot(&r0) / n_eff as f64;
        let s11 = r1.t().dot(&r1) / n_eff as f64;

        // Solve generalized eigenvalue problem: |lambda * S11 - S10 * S00^{-1} * S01| = 0
        let s00_inv = (&s00 + Array2::eye(k) * 1e-10).inv()?;
        let m = s10.dot(&s00_inv).dot(&s01);

        let s11_inv = (&s11 + Array2::eye(k) * 1e-10).inv()?;
        let eig_matrix = s11_inv.dot(&m);

        // Eigenvalues via power iteration with deflation
        let (eigenvalues, eigenvectors) = Self::eigen_decomposition(&eig_matrix, k);

        // Sort eigenvalues in descending order
        let mut sorted: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let sorted_eigenvalues: Array1<f64> =
            sorted.iter().map(|&(_, v)| v.clamp(0.0, 1.0)).collect();
        let mut sorted_eigenvectors = Array2::zeros((k, k));
        for (new_idx, &(orig_idx, _)) in sorted.iter().enumerate() {
            for i in 0..k {
                sorted_eigenvectors[(i, new_idx)] = eigenvectors[(i, orig_idx)];
            }
        }

        // Trace and Lambda-max statistics
        let mut trace_stats = Array1::zeros(k);
        let mut lambda_max_stats = Array1::zeros(k);

        for r in 0..k {
            // Trace: -n * sum_{i=r+1}^{k} ln(1 - lambda_i)
            let mut trace = 0.0;
            for i in r..k {
                let lambda = sorted_eigenvalues[i].min(0.99999);
                trace += (1.0 - lambda).ln();
            }
            trace_stats[r] = -(n_eff as f64) * trace;

            // Lambda-max: -n * ln(1 - lambda_{r+1})
            if r < k {
                let lambda = sorted_eigenvalues[r].min(0.99999);
                lambda_max_stats[r] = -(n_eff as f64) * (1.0 - lambda).ln();
            }
        }

        // Critical values (5%) — adjusted for breaks
        // Standard Johansen CV adjusted by: + 0.5 * n_breaks per break
        // (approximate adjustment, see Johansen et al. 2000)
        let break_adj = n_breaks as f64 * 0.5;
        let trace_cv_5 = Self::trace_cv_5(k, break_adj);
        let lambda_max_cv_5 = Self::lambda_max_cv_5(k, break_adj);

        // Determine cointegration rank
        let mut rank = 0;
        for r in 0..k {
            if trace_stats[r] > trace_cv_5[r] {
                rank = r + 1;
            }
        }

        // Cointegrating vectors for selected rank
        let cointegrating_vectors = if rank > 0 {
            sorted_eigenvectors
                .slice(ndarray::s![.., 0..rank])
                .to_owned()
        } else {
            Array2::zeros((k, 0))
        };

        Ok(JohansenBreakResult {
            trace_stats,
            lambda_max_stats,
            trace_cv_5,
            lambda_max_cv_5,
            eigenvalues: sorted_eigenvalues,
            cointegrating_vectors,
            cointegration_rank: rank,
            n_vars: k,
            lags,
            n_breaks,
            break_points: break_points.to_vec(),
            n_obs: n_eff,
        })
    }

    /// Trace test 5% critical values (Osterwald-Lenum 1992, with break adjustment).
    fn trace_cv_5(k: usize, break_adj: f64) -> Array1<f64> {
        // Standard CV for model with constant (no trend)
        let cv_table: Vec<Vec<f64>> = vec![
            vec![
                27.58, 29.96, 32.45, 34.87, 37.52, 40.15, 42.68, 45.10, 47.51, 49.85,
            ],
            vec![
                13.31, 15.09, 16.85, 18.63, 20.39, 22.17, 23.95, 25.72, 27.49, 29.25,
            ],
            vec![2.71, 3.76, 4.13, 4.40, 4.65, 4.88, 5.09, 5.30, 5.50, 5.69],
        ];
        let mut cv = Array1::zeros(k);
        for r in 0..k {
            let idx = (k - r - 1).min(9);
            let row = if k - r <= 1 {
                2
            } else if k - r <= 2 {
                1
            } else {
                0
            };
            cv[r] = cv_table[row.min(2)][idx] + break_adj * (k - r) as f64;
        }
        cv
    }

    /// Lambda-max 5% critical values (with break adjustment).
    fn lambda_max_cv_5(k: usize, break_adj: f64) -> Array1<f64> {
        let cv_table: Vec<Vec<f64>> = vec![
            vec![
                12.25, 14.26, 16.26, 18.17, 20.16, 22.12, 24.04, 25.90, 27.73, 29.52,
            ],
            vec![
                9.24, 11.22, 13.17, 15.09, 17.04, 18.97, 20.89, 22.79, 24.67, 26.54,
            ],
            vec![2.71, 3.76, 4.13, 4.40, 4.65, 4.88, 5.09, 5.30, 5.50, 5.69],
        ];
        let mut cv = Array1::zeros(k);
        for r in 0..k {
            let idx = (k - r - 1).min(9);
            let row = if k - r <= 1 {
                2
            } else if k - r <= 2 {
                1
            } else {
                0
            };
            cv[r] = cv_table[row.min(2)][idx] + break_adj;
        }
        cv
    }

    /// Simple eigenvalue decomposition via power iteration + deflation.
    fn eigen_decomposition(mat: &Array2<f64>, k: usize) -> (Array1<f64>, Array2<f64>) {
        let mut eigenvalues = Array1::zeros(k);
        let mut eigenvectors = Array2::zeros((k, k));
        let mut remaining = mat.clone();

        for i in 0..k {
            let (eigval, eigvec) = Self::power_iteration(&remaining, 200);
            eigenvalues[i] = eigval;
            for j in 0..k {
                eigenvectors[(j, i)] = eigvec[j];
            }
            // Deflate
            remaining = &remaining
                - &eigvec
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&eigvec.clone().insert_axis(Axis(0)))
                    * eigval;
        }

        (eigenvalues, eigenvectors)
    }

    fn power_iteration(mat: &Array2<f64>, n_iter: usize) -> (f64, Array1<f64>) {
        let k = mat.ncols();
        let mut v = Array1::ones(k) / (k as f64).sqrt();
        for _ in 0..n_iter {
            let v_new = mat.dot(&v);
            let norm = v_new.mapv(|x| x * x).sum().sqrt().max(1e-10);
            v = v_new / norm;
        }
        let lambda = v.dot(&mat.dot(&v));
        (lambda, v)
    }
}
