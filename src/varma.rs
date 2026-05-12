use crate::linalg::{LinalgDeterminant as _, LinalgInverse as _};
use crate::GreenersError;
use ndarray::{s, Array2};
use std::fmt;

#[derive(Debug)]
pub struct VarmaResult {
    pub ar_params: Array2<f64>,           // Matriz A (AR)
    pub ma_params: Array2<f64>,           // Matriz M (MA)
    pub exog_params: Option<Array2<f64>>, // Exogenous coefficients
    pub sigma_u: Array2<f64>,
    pub aic: f64,
    pub bic: f64,
    pub p_lags: usize,
    pub q_lags: usize,
    pub n_vars: usize,
    pub n_exog: usize,
    pub n_obs: usize,
}

impl fmt::Display for VarmaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(
                " VARMA{}({}, {}) via Hannan-Rissanen ",
                if self.n_exog > 0 { "X" } else { "" },
                self.p_lags,
                self.q_lags
            )
        )?;
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
        q: usize, // Lags MA
    ) -> Result<VarmaResult, GreenersError> {
        Self::fit_with_exog(data, p, q, None)
    }

    /// Estima um modelo VARMAX(p, q) with exogenous regressors.
    pub fn fit_with_exog(
        data: &Array2<f64>,
        p: usize,
        q: usize,
        exog: Option<&Array2<f64>>,
    ) -> Result<VarmaResult, GreenersError> {
        let t_total = data.nrows();
        let k = data.ncols();
        let n_exog = exog.map_or(0, |e| e.ncols());

        if let Some(e) = exog {
            if e.nrows() != t_total {
                return Err(GreenersError::ShapeMismatch(
                    "exog rows must match data rows".into(),
                ));
            }
        }

        // Heurística para o "Long VAR": p_long > p e q.
        let p_long = (p + q).max((t_total as f64).powf(0.25) as usize + 2).max(4);

        if t_total <= p_long + 1 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations for Hannan-Rissanen".into(),
            ));
        }

        // --- PASSO 1: LONG VAR para estimar resíduos (u_hat) ---
        let y_long = data.slice(s![p_long.., ..]).to_owned();
        let n_obs_long = y_long.nrows();

        let n_cols_long = 1 + k * p_long + n_exog;
        let mut x_long = Array2::<f64>::zeros((n_obs_long, n_cols_long));
        x_long.column_mut(0).fill(1.0);

        for i in 0..n_obs_long {
            let t_idx = p_long + i;
            for l in 1..=p_long {
                let lag_row = data.row(t_idx - l);
                let start_col = 1 + (l - 1) * k;
                for j in 0..k {
                    x_long[[i, start_col + j]] = lag_row[j];
                }
            }
            // Exogenous columns
            if let Some(e) = exog {
                let exog_start = 1 + k * p_long;
                for j in 0..n_exog {
                    x_long[[i, exog_start + j]] = e[[t_idx, j]];
                }
            }
        }

        let xtx_long = x_long.t().dot(&x_long);
        let xtx_long_inv = xtx_long.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let xty_long = x_long.t().dot(&y_long);
        let params_long = xtx_long_inv.dot(&xty_long);

        let preds_long = x_long.dot(&params_long);
        let u_hat = &y_long - &preds_long;

        // --- PASSO 2: Regressão VARMA Real ---
        if t_total <= p_long + q {
            return Err(GreenersError::ShapeMismatch(
                "Not enough obs for step 2".into(),
            ));
        }

        let start_t_step2 = p_long + q;
        let y_final = data.slice(s![start_t_step2.., ..]).to_owned();
        let n_obs_final = y_final.nrows();

        // Colunas: Intercepto (1) + AR (p*k) + MA (q*k) + Exog (n_exog)
        let n_cols_final = 1 + (p * k) + (q * k) + n_exog;
        let mut x_final = Array2::<f64>::zeros((n_obs_final, n_cols_final));
        x_final.column_mut(0).fill(1.0);

        for i in 0..n_obs_final {
            let t_real = start_t_step2 + i;

            // AR lags
            for l in 1..=p {
                let lag_row = data.row(t_real - l);
                let start_col = 1 + (l - 1) * k;
                for j in 0..k {
                    x_final[[i, start_col + j]] = lag_row[j];
                }
            }

            // MA lags
            for l in 1..=q {
                let u_idx = (t_real - l) - p_long;
                let u_row = u_hat.row(u_idx);
                let start_col = 1 + (p * k) + (l - 1) * k;
                for j in 0..k {
                    x_final[[i, start_col + j]] = u_row[j];
                }
            }

            // Exogenous
            if let Some(e) = exog {
                let exog_start = 1 + (p * k) + (q * k);
                for j in 0..n_exog {
                    x_final[[i, exog_start + j]] = e[[t_real, j]];
                }
            }
        }

        let xtx = x_final.t().dot(&x_final);
        let xtx_inv = xtx.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let xty = x_final.t().dot(&y_final);
        let params_final = xtx_inv.dot(&xty);

        // Separate parameters
        let split_ar = 1 + p * k;
        let split_ma = split_ar + q * k;
        let ar_params = params_final.slice(s![0..split_ar, ..]).to_owned();
        let ma_params = params_final.slice(s![split_ar..split_ma, ..]).to_owned();
        let exog_params = if n_exog > 0 {
            Some(params_final.slice(s![split_ma.., ..]).to_owned())
        } else {
            None
        };

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
            exog_params,
            sigma_u,
            aic,
            bic,
            p_lags: p,
            q_lags: q,
            n_vars: k,
            n_exog,
            n_obs: n_obs_final,
        })
    }
}
