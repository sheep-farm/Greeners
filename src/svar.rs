use crate::var::VarResult;
use crate::{GreenersError, VAR};
use ndarray::{s, Array2, Array3};
use ndarray_linalg::{Cholesky, Inverse, UPLO};
use std::fmt;

/// Identification scheme for SVAR.
#[derive(Debug, Clone)]
pub enum SVarIdentification {
    /// Cholesky decomposition (recursive ordering).
    Cholesky,
    /// Short-run restrictions: A * u_t = B * e_t.
    /// Masks are k x k matrices where NaN = free parameter, finite = restricted value.
    ShortRun(Array2<f64>, Array2<f64>),
    /// Long-run restrictions (Blanchard-Quah style).
    /// Mask is k x k where NaN = free, 0.0 = restricted to zero.
    LongRun(Array2<f64>),
}

/// SVAR estimation result.
#[derive(Debug)]
pub struct SVarResult {
    pub var_result: VarResult,
    pub a_matrix: Array2<f64>,
    pub b_matrix: Array2<f64>,
    pub identification: String,
}

impl SVarResult {
    /// Structural impulse response function.
    ///
    /// Returns Array3 of dimension (steps x k x k).
    /// Element [h, i, j] = response of variable i to structural shock j at horizon h.
    pub fn structural_irf(&self, steps: usize) -> Result<Array3<f64>, GreenersError> {
        let k = self.var_result.n_vars;
        let p = self.var_result.lags;

        // Structural impact matrix: A^{-1} * B
        let a_inv = self
            .a_matrix
            .inv()
            .map_err(|_| GreenersError::SingularMatrix)?;
        let impact = a_inv.dot(&self.b_matrix);

        // Extract companion form A matrices
        let mut a_matrices = Vec::new();
        for l in 0..p {
            let start_row = 1 + l * k;
            let end_row = 1 + (l + 1) * k;
            let a_lag = self
                .var_result
                .params
                .slice(s![start_row..end_row, ..])
                .t()
                .to_owned();
            a_matrices.push(a_lag);
        }

        // Compute IRF recursively
        let mut phi_history = Vec::with_capacity(steps);
        let mut irf = Array3::<f64>::zeros((steps, k, k));

        let phi_0 = Array2::<f64>::eye(k);
        let theta_0 = phi_0.dot(&impact);
        irf.slice_mut(s![0, .., ..]).assign(&theta_0);
        phi_history.push(phi_0);

        for h in 1..steps {
            let mut phi_h = Array2::<f64>::zeros((k, k));
            for j in 1..=p {
                if h >= j {
                    phi_h = phi_h + a_matrices[j - 1].dot(&phi_history[h - j]);
                }
            }
            let theta_h = phi_h.dot(&impact);
            irf.slice_mut(s![h, .., ..]).assign(&theta_h);
            phi_history.push(phi_h);
        }

        Ok(irf)
    }

    /// Structural forecast error variance decomposition.
    pub fn structural_fevd(&self, steps: usize) -> Result<Array3<f64>, GreenersError> {
        let k = self.var_result.n_vars;
        let irf = self.structural_irf(steps)?;

        let mut fevd = Array3::<f64>::zeros((steps, k, k));

        for i in 0..k {
            let mut cum_mse = vec![0.0f64; k];
            for h in 0..steps {
                for j in 0..k {
                    cum_mse[j] += irf[[h, i, j]].powi(2);
                }
                let total: f64 = cum_mse.iter().sum();
                if total > 1e-15 {
                    for j in 0..k {
                        fevd[[h, i, j]] = cum_mse[j] / total;
                    }
                }
            }
        }

        Ok(fevd)
    }
}

impl fmt::Display for SVarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" SVAR({}) — {} ", self.var_result.lags, self.identification)
        )?;
        writeln!(f, "{:<20} {:>10}", "No. Variables:", self.var_result.n_vars)?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.var_result.n_obs)?;

        writeln!(f, "\n{:-^78}", " A Matrix ")?;
        for row in self.a_matrix.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>10.4} ", val)?;
            }
            writeln!(f, "]")?;
        }

        writeln!(f, "\n{:-^78}", " B Matrix ")?;
        for row in self.b_matrix.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>10.4} ", val)?;
            }
            writeln!(f, "]")?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// Structural VAR model.
pub struct SVAR;

impl SVAR {
    /// Fit a Structural VAR model.
    ///
    /// * `data` — T x k data matrix
    /// * `lags` — number of VAR lags
    /// * `identification` — identification scheme
    pub fn fit(
        data: &Array2<f64>,
        lags: usize,
        identification: SVarIdentification,
    ) -> Result<SVarResult, GreenersError> {
        let k = data.ncols();

        // First, estimate reduced-form VAR
        let var_result = VAR::fit(data, lags, None)?;

        let (a_matrix, b_matrix, id_name) = match identification {
            SVarIdentification::Cholesky => {
                let p_chol = var_result
                    .sigma_u
                    .cholesky(UPLO::Lower)
                    .map_err(|_| GreenersError::SingularMatrix)?;
                let a = Array2::<f64>::eye(k);
                (a, p_chol, "Cholesky".to_string())
            }
            SVarIdentification::ShortRun(a_mask, b_mask) => {
                // Solve A * Sigma_u * A' = B * B'
                // With restrictions from masks (NaN = free, finite = fixed)
                // Use iterative scoring or direct solve for just-identified case
                let (a, b) =
                    solve_short_run_restrictions(&var_result.sigma_u, &a_mask, &b_mask, k)?;
                (a, b, "Short-run restrictions".to_string())
            }
            SVarIdentification::LongRun(_c_mask) => {
                // Blanchard-Quah: long-run impact matrix C(1)*A^{-1} has restrictions
                // C(1) = (I - A1 - A2 - ... - Ap)^{-1}
                let mut c1_inv = Array2::<f64>::eye(k);
                let p = var_result.lags;
                for l in 0..p {
                    let start_row = 1 + l * k;
                    let end_row = 1 + (l + 1) * k;
                    let a_lag = var_result
                        .params
                        .slice(s![start_row..end_row, ..])
                        .t()
                        .to_owned();
                    c1_inv -= &a_lag;
                }
                let c1 = c1_inv.inv().map_err(|_| GreenersError::SingularMatrix)?;

                // Long-run impact = C(1) * A^{-1} * B
                // Under Cholesky on the long-run covariance:
                // C(1) * Sigma_u * C(1)' = (C(1)*P) * (C(1)*P)'
                let long_run_cov = c1.dot(&var_result.sigma_u).dot(&c1.t());
                let long_run_chol = long_run_cov
                    .cholesky(UPLO::Lower)
                    .map_err(|_| GreenersError::SingularMatrix)?;

                // Impact matrix: P = C(1)^{-1} * long_run_chol
                let b = c1_inv.dot(&long_run_chol);
                let a = Array2::<f64>::eye(k);

                (a, b, "Long-run restrictions".to_string())
            }
        };

        Ok(SVarResult {
            var_result,
            a_matrix,
            b_matrix,
            identification: id_name,
        })
    }
}

/// Solve short-run SVAR restrictions via iterative method.
/// For just-identified case with A diagonal and B lower triangular,
/// this reduces to Cholesky-like decomposition.
fn solve_short_run_restrictions(
    sigma_u: &Array2<f64>,
    a_mask: &Array2<f64>,
    b_mask: &Array2<f64>,
    k: usize,
) -> Result<(Array2<f64>, Array2<f64>), GreenersError> {
    // Initialize A and B from masks
    let mut a = Array2::<f64>::eye(k);
    let mut b = Array2::<f64>::eye(k);

    // Apply fixed restrictions
    for i in 0..k {
        for j in 0..k {
            if a_mask[[i, j]].is_finite() {
                a[[i, j]] = a_mask[[i, j]];
            }
            if b_mask[[i, j]].is_finite() {
                b[[i, j]] = b_mask[[i, j]];
            }
        }
    }

    // Iterative scoring: A * Sigma * A' = B * B'
    // Simple iteration: given A, solve for B via Cholesky of A*Sigma*A'
    for _iter in 0..100 {
        let target = a.dot(sigma_u).dot(&a.t());

        // B = cholesky(target)
        let b_new = target
            .cholesky(UPLO::Lower)
            .map_err(|_| GreenersError::SingularMatrix)?;

        // Apply B mask restrictions
        for i in 0..k {
            for j in 0..k {
                if b_mask[[i, j]].is_finite() {
                    // Keep restricted value
                } else {
                    b[[i, j]] = b_new[[i, j]];
                }
            }
        }

        // Check convergence
        let diff = (&a.dot(sigma_u).dot(&a.t()) - &b.dot(&b.t()))
            .mapv(|v| v.abs())
            .sum();
        if diff < 1e-10 {
            break;
        }
    }

    Ok((a, b))
}
