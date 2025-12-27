use crate::GreenersError;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Eig, Inverse}; // Adicionado Cholesky
use num_complex::Complex64;
use std::fmt;

#[derive(Debug)]
pub struct VecmResult {
    pub alpha: Array2<f64>,
    pub beta: Array2<f64>,
    pub gamma: Array2<f64>,
    pub rank: usize,
    pub n_vars: usize,
    pub n_obs: usize,
    pub eigenvalues: Array1<f64>,
}

impl fmt::Display for VecmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" VECM (Johansen ML) - Rank {} ", self.rank)
        )?;
        writeln!(f, "{:<20} {:>10}", "No. Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;

        writeln!(f, "\n{:-^78}", " Cointegration Vectors (Beta) ")?;
        writeln!(
            f,
            "Interpret: Long-run equilibrium relationships (The 'Leash')"
        )?;
        for row in self.beta.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>10.4} ", val)?;
            }
            writeln!(f, "]")?;
        }

        writeln!(f, "\n{:-^78}", " Adjustment Coefficients (Alpha) ")?;
        writeln!(f, "Interpret: Speed of correction towards equilibrium")?;
        for row in self.alpha.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>10.4} ", val)?;
            }
            writeln!(f, "]")?;
        }

        writeln!(f, "\n{:-^78}", " Johansen Eigenvalues (Lambda) ")?;
        for val in &self.eigenvalues {
            write!(f, "{:>10.4} ", val)?;
        }
        writeln!(f, "\n{:=^78}", "")
    }
}

pub struct VECM;

impl VECM {
    pub fn fit(data: &Array2<f64>, lags: usize, rank: usize) -> Result<VecmResult, GreenersError> {
        let t_total = data.nrows();
        let k = data.ncols();

        if rank == 0 || rank >= k {
            return Err(GreenersError::ShapeMismatch(
                "Rank must be between 1 and k-1".into(),
            ));
        }

        let _n_obs = t_total - lags;

        // 1. Preparar Dados (Delta Y_t)
        let mut dy = Array2::<f64>::zeros((t_total - 1, k));
        for i in 1..t_total {
            let diff = &data.row(i) - &data.row(i - 1);
            dy.row_mut(i - 1).assign(&diff);
        }

        let n_eff = t_total - lags;
        let p_vecm = lags - 1;
        let n_z_cols = k * p_vecm + 1;

        let mut z_mat = Array2::<f64>::zeros((n_eff, n_z_cols));
        let mut dy_target = Array2::<f64>::zeros((n_eff, k));
        let mut y_lag_level = Array2::<f64>::zeros((n_eff, k));

        for i in 0..n_eff {
            let t_original = lags + i;

            let dy_row = &data.row(t_original) - &data.row(t_original - 1);
            dy_target.row_mut(i).assign(&dy_row);

            y_lag_level.row_mut(i).assign(&data.row(t_original - 1));

            z_mat[[i, 0]] = 1.0; // Intercepto

            for l in 1..=p_vecm {
                let lag_time = t_original - l;
                let dy_lag = &data.row(lag_time) - &data.row(lag_time - 1);

                let start_col = 1 + (l - 1) * k;
                for j in 0..k {
                    z_mat[[i, start_col + j]] = dy_lag[j];
                }
            }
        }

        // 2. Regressões Auxiliares
        let ztz = z_mat.t().dot(&z_mat);
        let ztz_inv = ztz.inv().map_err(|_| GreenersError::SingularMatrix)?;

        let beta_0 = ztz_inv.dot(&z_mat.t()).dot(&dy_target);
        let r0 = &dy_target - &z_mat.dot(&beta_0);

        let beta_1 = ztz_inv.dot(&z_mat.t()).dot(&y_lag_level);
        let r1 = &y_lag_level - &z_mat.dot(&beta_1);

        // 3. Matrizes de Momento
        let t_float = n_eff as f64;
        let s00 = r0.t().dot(&r0) / t_float;
        let s11 = r1.t().dot(&r1) / t_float;
        let s01 = r0.t().dot(&r1) / t_float;
        let s10 = s01.t();

        // Resolver autovalores generalizados
        let s11_chol = s11
            .cholesky(ndarray_linalg::UPLO::Lower)
            .map_err(|_| GreenersError::SingularMatrix)?;

        let s11_inv_chol = s11_chol.inv().map_err(|_| GreenersError::SingularMatrix)?;

        let s00_inv = s00.inv().map_err(|_| GreenersError::SingularMatrix)?;

        let temp = s11_inv_chol
            .dot(&s10)
            .dot(&s00_inv)
            .dot(&s01)
            .dot(&s11_inv_chol.t());

        let (eigvals_complex, eigvecs_complex) =
            temp.eig().map_err(|_| GreenersError::OptimizationFailed)?;

        // 4. Filtrar e Ordenar (CORREÇÃO DE TIPOS AQUI)
        let mut pairs: Vec<(f64, Array1<f64>)> = eigvals_complex
            .iter()
            .enumerate()
            .map(|(i, v)| {
                // Explicitamos que 'c' é Complex64 para ajudar o compilador
                let vec_real = eigvecs_complex.column(i).mapv(|c: Complex64| c.re);
                (v.re, vec_real)
            })
            .collect();

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let sorted_eigenvalues: Array1<f64> = Array1::from_vec(pairs.iter().map(|p| p.0).collect());

        // 5. Estimar Beta e Alpha
        let mut beta_est = Array2::<f64>::zeros((k, rank));
        for r in 0..rank {
            let v = &pairs[r].1;
            let beta_vec = s11_inv_chol.t().dot(v);
            beta_est.column_mut(r).assign(&beta_vec);
        }

        let cointegration_term = r1.dot(&beta_est);

        let alpha_est = r0.t().dot(&cointegration_term).dot(
            &cointegration_term
                .t()
                .dot(&cointegration_term)
                .inv()
                .unwrap(),
        );

        let error_correction = y_lag_level.dot(&beta_est).dot(&alpha_est.t());
        let dy_clean = &dy_target - &error_correction;
        let gamma_full = ztz_inv.dot(&z_mat.t()).dot(&dy_clean);

        Ok(VecmResult {
            alpha: alpha_est,
            beta: beta_est,
            gamma: gamma_full.t().to_owned(),
            rank,
            n_vars: k,
            n_obs: n_eff,
            eigenvalues: sorted_eigenvalues,
        })
    }
}
