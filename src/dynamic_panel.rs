use ndarray::{Array1, Array2, s, Axis};
use ndarray_linalg::Inverse;
use crate::{GreenersError, OLS, CovarianceType};
use std::fmt;

#[derive(Debug)]
pub struct ArellanoBondResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub sargan_stat: f64, // Teste de validade dos instrumentos
    pub n_obs: usize,
}

impl fmt::Display for ArellanoBondResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Dynamic Panel (Arellano-Bond / Diff-GMM) ")?;
        writeln!(f, "{:<20} {:>15}", "Estimator:", "IV-GMM on First Differences")?;
        writeln!(f, "{:<20} {:>15.4}", "Sargan Test:", self.sargan_stat)?;
        writeln!(f, "{:<20} {:>15}", "Obs (Effective):", self.n_obs)?;
        
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "{:<15} | {:>10} | {:>10} | {:>8} | {:>8}", 
            "Variable", "Coef", "Std Err", "t", "P>|t|")?;
        writeln!(f, "{:-^78}", "")?;
        
        // O primeiro coeficiente é sempre o Lag da Dependente
        writeln!(f, "{:<15} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}", 
            "L1.DepVar", self.params[0], self.std_errors[0], self.t_values[0], self.p_values[0]
        )?;

        for i in 1..self.params.len() {
            writeln!(f, "D.x{:<11} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}", 
                i-1, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct ArellanoBond;

impl ArellanoBond {
    /// Estima um painel dinâmico: y_it = rho * y_{i,t-1} + beta * x_it + alpha_i + e_it
    /// Método: Diferença Primeira instrumentada por y_{i,t-2} (Anderson-Hsiao / Diff GMM).
    /// Assume que x_it são estritamente exógenos (instrumentam a si mesmos na diferença).
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &Array1<i64>,
        time_ids: &Array1<i64>
    ) -> Result<ArellanoBondResult, GreenersError> {
        let n_total = y.len();
        
        if entity_ids.len() != n_total || time_ids.len() != n_total {
            return Err(GreenersError::ShapeMismatch("IDs mismatch".into()));
        }

        // 1. Organizar dados (Sort by Entity, then Time)
        // Precisamos garantir a ordem para calcular lags e diferenças corretamente
        let mut indices: Vec<usize> = (0..n_total).collect();
        indices.sort_by_key(|&i| (entity_ids[i], time_ids[i]));

        // Reconstruir arrays ordenados
        let y_sorted = y.select(Axis(0), &indices);
        let x_sorted = x.select(Axis(0), &indices);
        let id_sorted = entity_ids.select(Axis(0), &indices);
        let _t_sorted = time_ids.select(Axis(0), &indices);

        // 2. Construir Variáveis Transformadas (D.y, D.lag_y, D.x, lag2_y)
        // Perderemos 2 observações por indivíduo (uma pelo lag, uma pela diferença)
        
        let mut dy_vec = Vec::new();       // Dependente: Delta y_t
        let mut dlag_y_vec = Vec::new();   // Regressor Endógeno: Delta y_{t-1}
        let mut dx_vec = Vec::new();       // Regressores Exógenos: Delta x_t
        let mut inst_lag2_vec = Vec::new();// Instrumento: y_{t-2} (Nível)

        // Precisamos percorrer por indivíduo
        let mut start = 0;
        while start < n_total {
            let current_id = id_sorted[start];
            let mut end = start;
            while end < n_total && id_sorted[end] == current_id {
                end += 1;
            }
            
            // Slice do indivíduo
            let y_i = y_sorted.slice(s![start..end]);
            let x_i = x_sorted.slice(s![start..end, ..]);
            // let t_i = t_sorted.slice(s![start..end]); // Assumindo continuidade temporal para simplificar
            
            let t_len = end - start;
            if t_len >= 3 { // Precisa de pelo menos 3 períodos (t, t-1, t-2)
                for t in 2..t_len {
                    // Y: Delta y_t = y_t - y_{t-1}
                    let dy = y_i[t] - y_i[t-1];
                    
                    // X Endógeno: Delta y_{t-1} = y_{t-1} - y_{t-2}
                    let dlag_y = y_i[t-1] - y_i[t-2];

                    // Instrumento: y_{t-2} (Nível)
                    // Este é o pulo do gato do Arellano-Bond: usar nível passado para instrumentar diferença futura
                    let z_level = y_i[t-2];

                    // X Exógeno: Delta x_t = x_t - x_{t-1}
                    // Loop pelas colunas de X
                    let mut dx_row = Vec::new();
                    for k in 0..x_sorted.ncols() {
                        dx_row.push(x_i[[t, k]] - x_i[[t-1, k]]);
                    }

                    // Push nos vetores globais
                    dy_vec.push(dy);
                    dlag_y_vec.push(dlag_y);
                    inst_lag2_vec.push(z_level);
                    dx_vec.extend(dx_row);
                }
            }
            start = end;
        }

        // 3. Montar Matrizes Finais
        let n_eff = dy_vec.len();
        if n_eff == 0 {
            return Err(GreenersError::ShapeMismatch("Not enough time periods (need T>=3)".into()));
        }

        let k_x = x.ncols();
        let dy_final = Array1::from(dy_vec);
        
        // Matriz de Regressores W = [D.y_{t-1}, D.X]
        let mut w_data = Vec::with_capacity(n_eff * (1 + k_x));
        for i in 0..n_eff {
            w_data.push(dlag_y_vec[i]); // Primeira coluna: Endógena
            for j in 0..k_x {
                w_data.push(dx_vec[i * k_x + j]); // Colunas seguintes: Exógenas
            }
        }
        let w_matrix = Array2::from_shape_vec((n_eff, 1 + k_x), w_data)
             .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

        // Matriz de Instrumentos Z = [y_{t-2}, D.X]
        // Assumimos que D.X instrumenta a si mesmo (exógeno estrito)
        let mut z_data = Vec::with_capacity(n_eff * (1 + k_x));
        for i in 0..n_eff {
            z_data.push(inst_lag2_vec[i]); // Instrumento principal
            for j in 0..k_x {
                z_data.push(dx_vec[i * k_x + j]); // Instrumentos exógenos
            }
        }
        let z_matrix = Array2::from_shape_vec((n_eff, 1 + k_x), z_data)
             .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

        // 4. Estimação 2SLS (GMM)
        // beta_iv = (W' P_z W)^-1 W' P_z y
        // Onde P_z = Z (Z'Z)^-1 Z'
        
        // Hat Matrix: X_hat = Z * (Z'Z)^-1 * Z'W
        let zt_z = z_matrix.t().dot(&z_matrix);
        let zt_z_inv = zt_z.inv().map_err(|_| GreenersError::SingularMatrix)?;
        
        let projection = z_matrix.dot(&zt_z_inv).dot(&z_matrix.t());
        let w_hat = projection.dot(&w_matrix); // Instrumentando os regressores

        // Rodar OLS de dy contra w_hat
        let ols_iv = OLS::fit(&dy_final, &w_hat, CovarianceType::NonRobust)?;

        // Recalcular erros padrão corretos (usando resíduos originais, não projetados)
        let params = ols_iv.params;
        let residuals = &dy_final - &w_matrix.dot(&params); // Resíduos reais u = y - W*beta
        let sigma2 = residuals.mapv(|x| x.powi(2)).sum() / (n_eff as f64 - params.len() as f64);
        
        // Var(beta) = sigma2 * (W_hat' W_hat)^-1
        let wt_w_inv = w_hat.t().dot(&w_hat).inv().map_err(|_| GreenersError::SingularMatrix)?;
        let var_beta = wt_w_inv * sigma2;
        let std_errors = var_beta.diag().mapv(f64::sqrt);

        // Estatísticas t e p
        let t_values = &params / &std_errors;
        let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        use statrs::distribution::ContinuousCDF;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        // Teste de Sargan (Validade dos Instrumentos)
        // Como o número de inst (Z) == número de vars (W) neste caso simplificado,
        // o Sargan será zero (identificado exatamente).
        // Em um AB completo com mais lags, isso seria > 0.
        let sargan = 0.0; // Placeholder para esta implementação exactly-identified

        Ok(ArellanoBondResult {
            params,
            std_errors,
            t_values,
            p_values,
            r_squared: ols_iv.r_squared, // Note: R2 em modelos diferençados é pouco informativo
            sargan_stat: sargan,
            n_obs: n_eff,
        })
    }
}