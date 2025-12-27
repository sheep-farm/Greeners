use crate::GreenersError; // Removidos OLS, CovarianceType
use ndarray::{s, Array2, Array3}; // Removidos Array1, Axis
use ndarray_linalg::{Cholesky, Determinant, Inverse}; // <--- Adicionado Determinant
use std::fmt;

#[derive(Debug)]
pub struct VarResult {
    pub params: Array2<f64>,  // Matriz (1 + k*p) x k
    pub sigma_u: Array2<f64>, // Covariância dos resíduos (k x k)
    pub aic: f64,
    pub bic: f64,
    pub lags: usize,
    pub n_vars: usize,
    pub n_obs: usize,
    pub var_names: Vec<String>,
}

impl VarResult {
    /// Calcula a Função de Impulso-Resposta (IRF) Ortogonalizada.
    /// Usa Decomposição de Cholesky para identificar os choques estruturais.
    /// Retorna Array3 de dimensão (steps x k x k).
    /// Elemento [h, i, j] = Resposta da variável i ao choque na variável j no tempo h.
    pub fn irf(&self, steps: usize) -> Result<Array3<f64>, GreenersError> {
        let k = self.n_vars;
        let p = self.lags;

        // 1. Identificação de Choques (Cholesky)
        // P * P' = Sigma_u. P é triangular inferior.
        let p_chol = self
            .sigma_u
            .cholesky(ndarray_linalg::UPLO::Lower)
            .map_err(|_| GreenersError::SingularMatrix)?;

        // 2. Extrair matrizes A_1, ..., A_p dos parâmetros achatados
        let mut a_matrices = Vec::new();
        for l in 0..p {
            let start_row = 1 + l * k;
            let end_row = 1 + (l + 1) * k;
            // Transposta para ficar na forma y_t = A * y_{t-1}
            let a_lag = self.params.slice(s![start_row..end_row, ..]).t().to_owned();
            a_matrices.push(a_lag);
        }

        // 3. Calcular IRF Recursivamente (VMA representation)
        let mut phi_history = Vec::with_capacity(steps);
        let mut irf_tensor = Array3::<f64>::zeros((steps, k, k));

        // t=0
        let phi_0 = Array2::<f64>::eye(k);
        let theta_0 = phi_0.dot(&p_chol);
        irf_tensor.slice_mut(s![0, .., ..]).assign(&theta_0);
        phi_history.push(phi_0);

        for h in 1..steps {
            let mut phi_h = Array2::<f64>::zeros((k, k));

            // Soma ponderada pelos lags passados
            for j in 1..=p {
                if h >= j {
                    let a_j = &a_matrices[j - 1];
                    let phi_prev = &phi_history[h - j];
                    phi_h = phi_h + a_j.dot(phi_prev);
                }
            }

            // Ortogonalizar
            let theta_h = phi_h.dot(&p_chol);
            irf_tensor.slice_mut(s![h, .., ..]).assign(&theta_h);
            phi_history.push(phi_h);
        }

        Ok(irf_tensor)
    }

    /// Teste de Causalidade de Granger (Placeholder).
    /// Retorna erro por enquanto, pois requer reestimação do modelo restrito.
    pub fn granger_causality(
        &self,
        _cause_idx: usize,
        _effect_idx: usize,
    ) -> Result<(f64, f64), GreenersError> {
        // Prefixamos com _ para silenciar warnings de "unused variables"
        Err(GreenersError::ShapeMismatch(
            "Granger Causality requires restricted model estimation (Not implemented in v0.1)"
                .into(),
        ))
    }
}

impl fmt::Display for VarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" Vector Autoregression (VAR({})) ", self.lags)
        )?;
        writeln!(f, "{:<15} {:>10}", "No. Variables:", self.n_vars)?;
        writeln!(f, "{:<15} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<15} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<15} {:>10.4}", "BIC:", self.bic)?;
        writeln!(f, "{:=^78}", "")?;

        writeln!(f, "\n{:-^78}", " Residual Covariance (Sigma_u) ")?;
        for row in self.sigma_u.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>10.4} ", val)?;
            }
            writeln!(f, "]")?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct VAR;

impl VAR {
    pub fn fit(
        data: &Array2<f64>,
        lags: usize,
        var_names: Option<Vec<String>>,
    ) -> Result<VarResult, GreenersError> {
        let t_total = data.nrows();
        let k = data.ncols();

        if t_total <= lags {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations for lags".into(),
            ));
        }

        // 1. Criar Matrizes Y e X (Lags)
        // Y efetivo: t = p até T
        let y_eff = data.slice(s![lags.., ..]).to_owned();
        let n_obs = y_eff.nrows();

        // X matrix: [1, y_{t-1}, ..., y_{t-p}]
        // Dimensão: N x (1 + k*p)
        let n_cols_x = 1 + k * lags;
        let mut x_mat = Array2::<f64>::zeros((n_obs, n_cols_x));

        // Preencher Intercepto
        x_mat.column_mut(0).fill(1.0);

        // Preencher Lags
        for i in 0..n_obs {
            // A linha i do X corresponde ao tempo (lags + i)
            let current_time_idx = lags + i;

            for l in 1..=lags {
                let lag_idx = current_time_idx - l;
                let lag_data = data.row(lag_idx);

                let start_col = 1 + (l - 1) * k;
                for var_idx in 0..k {
                    x_mat[[i, start_col + var_idx]] = lag_data[var_idx];
                }
            }
        }

        // 2. Estimação OLS Multivariada (Algebra direta)
        // B = (X'X)^-1 X'Y
        let xt_x = x_mat.t().dot(&x_mat);
        let xt_x_inv = xt_x.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let xt_y = x_mat.t().dot(&y_eff);

        let params = xt_x_inv.dot(&xt_y); // (1+kp) x k

        // 3. Resíduos e Sigma
        let preds = x_mat.dot(&params);
        let residuals = &y_eff - &preds;

        let sigma_u = residuals.t().dot(&residuals) / ((n_obs - n_cols_x) as f64);

        // 4. Critérios de Informação
        // Para usar .det(), importamos a trait Determinant
        let det_sigma = sigma_u.det().unwrap_or(1.0).max(1e-10);
        let log_det = det_sigma.ln();

        let t_float = n_obs as f64;

        // AIC = ln(|Sigma|) + 2*K_total/T
        let aic = log_det + (2.0 * (k * n_cols_x) as f64) / t_float;
        let bic = log_det + ((k * n_cols_x) as f64 * t_float.ln()) / t_float;

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("Var{}", i)).collect());

        Ok(VarResult {
            params,
            sigma_u,
            aic,
            bic,
            lags,
            n_vars: k,
            n_obs,
            var_names: names,
        })
    }
}
