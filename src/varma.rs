use ndarray::{Array2, s};
use ndarray_linalg::{Inverse, Determinant};
use crate::GreenersError;
use std::fmt;

#[derive(Debug)]
pub struct VarmaResult {
    pub ar_params: Array2<f64>, // Matriz A (AR)
    pub ma_params: Array2<f64>, // Matriz M (MA)
    pub sigma_u: Array2<f64>,
    pub aic: f64,
    pub bic: f64,
    pub p_lags: usize,
    pub q_lags: usize,
    pub n_vars: usize,
    pub n_obs: usize,
}

impl fmt::Display for VarmaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", format!(" VARMA({}, {}) via Hannan-Rissanen ", self.p_lags, self.q_lags))?;
        writeln!(f, "{:<15} {:>10}", "No. Variables:", self.n_vars)?;
        writeln!(f, "{:<15} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<15} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<15} {:>10.4}", "BIC:", self.bic)?;
        
        writeln!(f, "\n{:-^78}", " Residual Covariance (Sigma) ")?;
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

pub struct VARMA;

impl VARMA {
    /// Estima um modelo VARMA(p, q).
    /// Método: Hannan-Rissanen (2-step).
    /// 1. Estima um "Long VAR" para recuperar os resíduos.
    /// 2. Usa os resíduos defasados como regressores para a parte MA.
    pub fn fit(
        data: &Array2<f64>,
        p: usize, // Lags AR
        q: usize  // Lags MA
    ) -> Result<VarmaResult, GreenersError> {
        let t_total = data.nrows();
        let k = data.ncols();

        // Heurística para o "Long VAR": p_long > p e q.
        // Geralmente usamos algo maior para capturar a estrutura MA via AR.
        let p_long = (p + q).max((t_total as f64).powf(0.25) as usize + 2).max(4); 
        
        if t_total <= p_long + 1 {
            return Err(GreenersError::ShapeMismatch("Not enough observations for Hannan-Rissanen".into()));
        }

        // --- PASSO 1: LONG VAR para estimar resíduos (u_hat) ---
        // Y = A_long * Y_lags + u
        
        // Y efetivo começa em p_long
        let y_long = data.slice(s![p_long.., ..]).to_owned();
        let n_obs_long = y_long.nrows();
        
        // Montar X para o Long VAR
        let n_cols_long = 1 + k * p_long;
        let mut x_long = Array2::<f64>::zeros((n_obs_long, n_cols_long));
        x_long.column_mut(0).fill(1.0); // Intercepto

        for i in 0..n_obs_long {
            let t_idx = p_long + i;
            for l in 1..=p_long {
                let lag_row = data.row(t_idx - l);
                let start_col = 1 + (l - 1) * k;
                for j in 0..k {
                    x_long[[i, start_col + j]] = lag_row[j];
                }
            }
        }

        // Resolver Long VAR
        let xtx_long = x_long.t().dot(&x_long);
        let xtx_long_inv = xtx_long.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let xty_long = x_long.t().dot(&y_long);
        let params_long = xtx_long_inv.dot(&xty_long);

        // Calcular Resíduos Estimados (u_hat)
        // Precisamos alinhar: u_hat tem tamanho n_obs_long
        // Correspondendo aos tempos t = p_long .. T
        let preds_long = x_long.dot(&params_long);
        let u_hat = &y_long - &preds_long;

        // --- PASSO 2: Regressão VARMA Real ---
        // Agora regredimos Y contra Y_lags (AR) e u_hat_lags (MA)
        // Precisamos perder mais observações pq agora dependemos de lags de u_hat
        // O "tempo zero" da segunda regressão deve garantir que temos u_hat_{t-q}
        
        // O vetor u_hat começa no índice temporal original 'p_long'.
        // Para ter q lags de u_hat, precisamos começar q passos depois.
        // Novo início efetivo: t = p_long + q
        
        if t_total <= p_long + q {
             return Err(GreenersError::ShapeMismatch("Not enough obs for step 2".into()));
        }

        let start_t_step2 = p_long + q;
        let y_final = data.slice(s![start_t_step2.., ..]).to_owned();
        let n_obs_final = y_final.nrows();

        // Colunas: Intercepto (1) + AR (p*k) + MA (q*k)
        let n_cols_final = 1 + (p * k) + (q * k);
        let mut x_final = Array2::<f64>::zeros((n_obs_final, n_cols_final));
        x_final.column_mut(0).fill(1.0);

        for i in 0..n_obs_final {
            let t_real = start_t_step2 + i; // Índice no 'data' original

            // Preencher Lags AR (Y_{t-1} ... Y_{t-p})
            for l in 1..=p {
                let lag_row = data.row(t_real - l);
                let start_col = 1 + (l - 1) * k;
                for j in 0..k {
                    x_final[[i, start_col + j]] = lag_row[j];
                }
            }

            // Preencher Lags MA (u_hat_{t-1} ... u_hat_{t-q})
            // u_hat começa no tempo real 'p_long'.
            // O índice 0 de u_hat corresponde a data[p_long].
            // Queremos u_hat no tempo (t_real - l).
            // O índice no array u_hat será: (t_real - l) - p_long
            for l in 1..=q {
                let u_idx = (t_real - l) - p_long;
                let u_row = u_hat.row(u_idx);
                
                let start_col = 1 + (p * k) + (l - 1) * k; // Pula os ARs
                for j in 0..k {
                    x_final[[i, start_col + j]] = u_row[j];
                }
            }
        }

        // Resolver OLS Final
        let xtx = x_final.t().dot(&x_final);
        let xtx_inv = xtx.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let xty = x_final.t().dot(&y_final);
        let params_final = xtx_inv.dot(&xty);

        // --- Pós-Processamento ---
        // Separar parâmetros AR e MA
        // params structure: [Intercept (1) | AR lags (p*k) | MA lags (q*k)]
        
        // Extrair AR (pegando todos os lags e empilhando, simplificado aqui retornamos raw rows)
        // Para simplificar a struct, vamos retornar a matriz de coeficientes 'achatada' por tipo
        // AR Params Matrix: (1 + p*k) rows (incluindo intercepto)
        let split_idx = 1 + p * k;
        let ar_params = params_final.slice(s![0..split_idx, ..]).to_owned();
        let ma_params = params_final.slice(s![split_idx.., ..]).to_owned();

        // Calcular Sigma Final e Critérios
        let preds = x_final.dot(&params_final);
        let residuals = &y_final - &preds;
        let sigma_u = residuals.t().dot(&residuals) / ((n_obs_final - n_cols_final) as f64);
        
        let det_sigma = sigma_u.det().unwrap_or(1.0).max(1e-10);
        let log_det = det_sigma.ln();
        let t_float = n_obs_final as f64;
        
        let aic = log_det + (2.0 * (k * n_cols_final) as f64) / t_float;
        let bic = log_det + ((k * n_cols_final) as f64 * t_float.ln()) / t_float;

        Ok(VarmaResult {
            ar_params,
            ma_params,
            sigma_u,
            aic,
            bic,
            p_lags: p,
            q_lags: q,
            n_vars: k,
            n_obs: n_obs_final,
        })
    }
}