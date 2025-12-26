use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use crate::GreenersError;
use std::fmt;
use statrs::distribution::ContinuousCDF;

/// Estrutura para definir uma única equação do sistema
#[derive(Clone)]
pub struct Equation {
    pub y: Array1<f64>,
    pub x: Array2<f64>, // Inclui endógenas e exógenas
    pub name: String,
}

/// Resultado do Sistema 3SLS
#[derive(Debug)]
pub struct ThreeSLSResult {
    pub equations: Vec<EquationResult>,
    pub sigma_cross: Array2<f64>, // Matriz de covariância dos erros entre equações
    pub system_r2: f64, // R2 de McElroy (Opcional, mas chique)
}

#[derive(Debug)]
pub struct EquationResult {
    pub name: String,
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
}

impl fmt::Display for ThreeSLSResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Three-Stage Least Squares (3SLS) System ")?;
        writeln!(f, "Number of Equations: {}", self.equations.len())?;
        
        // Mostrar a matriz de correlação de resíduos (Cross-Equation Correlation)
        writeln!(f, "\n{:-^78}", " Residual Covariance Matrix (Sigma) ")?;
        for row in self.sigma_cross.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>10.4} ", val)?;
            }
            writeln!(f, "]")?;
        }

        for eq in &self.equations {
            writeln!(f, "\n{:-^78}", format!(" Equation: {} ", eq.name))?;
            writeln!(f, "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}", 
                "Variable", "Coef", "Std Err", "t", "P>|t|")?;
            writeln!(f, "{:-^78}", "")?;
            
            for i in 0..eq.params.len() {
                writeln!(f, "x{:<9} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}", 
                    i, eq.params[i], eq.std_errors[i], eq.t_values[i], eq.p_values[i]
                )?;
            }
            writeln!(f, "R-squared: {:.4}", eq.r_squared)?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct ThreeSLS;

impl ThreeSLS {
    /// Estima um sistema de equações simultâneas via 3SLS.
    /// 
    /// # Arguments
    /// * `equations` - Vetor de structs `Equation` (cada uma com y e X).
    /// * `z_instruments` - Matriz global de instrumentos (união de todas as exógenas).
    pub fn fit(
        equations: &[Equation],
        z_instruments: &Array2<f64>
    ) -> Result<ThreeSLSResult, GreenersError> {
        let n_obs = z_instruments.nrows();
        let n_eq = equations.len();

        // --- STAGE 1: Reduced Form & Projection ---
        // Projetar cada X no espaço de Z para obter X_hat = Z(Z'Z)^-1 Z'X
        // X_hat é a versão "limpa" das endógenas.
        
        // Pré-calcular P_z = Z (Z'Z)^-1 Z'
        // Para eficiência, calculamos apenas a parte (Z'Z)^-1 Z' e multiplicamos depois
        let z_t = z_instruments.t();
        let ztz = z_t.dot(z_instruments);
        let ztz_inv = ztz.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let projection_matrix_part = z_instruments.dot(&ztz_inv).dot(&z_t); // N x N (Cuidado com memória aqui se N for huge)
        
        let mut x_hat_list = Vec::new();
        let mut residuals_2sls = Array2::<f64>::zeros((n_obs, n_eq));

        // --- STAGE 2: 2SLS Equation-by-Equation ---
        for (i, eq) in equations.iter().enumerate() {
            // X_hat = P_z * X
            let x_hat = projection_matrix_part.dot(&eq.x);
            
            // Beta_2sls = (X_hat' X)^-1 X_hat' y
            // Nota: Em 2SLS clássico, usamos X_hat' X_hat ou X_hat' X, é equivalente.
            let xt_x = x_hat.t().dot(&eq.x);
            let xt_x_inv = xt_x.inv().map_err(|_| GreenersError::SingularMatrix)?;
            let xt_y = x_hat.t().dot(&eq.y);
            let beta_2sls = xt_x_inv.dot(&xt_y);

            // Resíduos u = y - X * beta (Usamos o X original para resíduos!)
            let pred = eq.x.dot(&beta_2sls);
            let u = &eq.y - &pred;
            
            // Guardar para o passo seguinte
            residuals_2sls.column_mut(i).assign(&u);
            x_hat_list.push(x_hat);
        }

        // Calcular Matriz de Covariância dos Erros (Sigma)
        // Sigma_ij = (u_i' u_j) / N
        let sigma = residuals_2sls.t().dot(&residuals_2sls) / (n_obs as f64);
        let sigma_inv = sigma.inv().map_err(|_| GreenersError::SingularMatrix)?;

        // --- STAGE 3: GLS Estimation on the System ---
        // Resolver o sistema gigante: [X_hat' (Sigma^-1 ox I) X_hat] Beta = X_hat' (Sigma^-1 ox I) y
        
        // 1. Contar total de parâmetros
        let mut k_total = 0;
        let mut k_per_eq = Vec::new();
        for eq in equations {
            let k = eq.x.ncols();
            k_per_eq.push(k);
            k_total += k;
        }

        // 2. Construir Matriz LHS (Hessiana do Sistema) e Vetor RHS
        // Usamos construção em bloco para evitar Kronecker explícito.
        let mut lhs_system = Array2::<f64>::zeros((k_total, k_total));
        let mut rhs_system = Array1::<f64>::zeros(k_total);

        let mut start_i = 0;
        for i in 0..n_eq {
            let ki = k_per_eq[i];
            let x_hat_i = &x_hat_list[i];
            
            let mut start_j = 0;
            for j in 0..n_eq {
                let kj = k_per_eq[j];
                let x_hat_j = &x_hat_list[j];
                
                // Elemento Sigma^{ij} (escalar)
                let s_ij = sigma_inv[[i, j]];
                
                // Bloco LHS = s_ij * (X_hat_i' * X_hat_j)
                let block = x_hat_i.t().dot(x_hat_j) * s_ij;
                
                // Inserir na matriz grandona
                lhs_system.slice_mut(ndarray::s![start_i..start_i+ki, start_j..start_j+kj]).assign(&block);

                // Parte do RHS (apenas quando loop j roda, acumula para o i)
                // RHS_i = sum_j (s_ij * X_hat_i' * y_j)
                let y_j = &equations[j].y;
                let vec_part = x_hat_i.t().dot(y_j) * s_ij;
                
                // Somar ao vetor RHS na posição i
                let mut target_slice = rhs_system.slice_mut(ndarray::s![start_i..start_i+ki]);
                target_slice += &vec_part;

                start_j += kj;
            }
            start_i += ki;
        }

        // 3. Resolver Beta 3SLS
        let lhs_inv = lhs_system.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let beta_3sls_all = lhs_inv.dot(&rhs_system);

        // --- PÓS-ESTIMAÇÃO: Separar resultados e Estatísticas ---
        let mut final_results = Vec::new();
        let mut cursor = 0;

        for (i, eq) in equations.iter().enumerate() {
            let k = k_per_eq[i];
            let params = beta_3sls_all.slice(ndarray::s![cursor..cursor+k]).to_owned();
            
            // Variância Assintótica dos coeficientes dessa equação
            // É o bloco diagonal correspondente da inversa do sistema
            let cov_params = lhs_inv.slice(ndarray::s![cursor..cursor+k, cursor..cursor+k]).to_owned();
            let std_errors = cov_params.diag().mapv(f64::sqrt);
            
            // Estatísticas T e P
            let t_values = &params / &std_errors;
            let p_values = t_values.mapv(|t| 2.0 * (1.0 - statrs::distribution::Normal::new(0.0, 1.0).unwrap().cdf(t.abs())));

            // R2 (Usando resíduos finais do 3SLS)
            let pred = eq.x.dot(&params);
            let res = &eq.y - &pred;
            let sst = (&eq.y - eq.y.mean().unwrap()).mapv(|v| v.powi(2)).sum();
            let ssr = res.mapv(|v| v.powi(2)).sum();
            let r2 = 1.0 - (ssr / sst);

            final_results.push(EquationResult {
                name: eq.name.clone(),
                params,
                std_errors,
                t_values,
                p_values,
                r_squared: r2,
            });

            cursor += k;
        }

        Ok(ThreeSLSResult {
            equations: final_results,
            sigma_cross: sigma,
            system_r2: 0.0, // Placeholder
        })
    }
}