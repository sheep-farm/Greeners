use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use crate::{GreenersError, OLS, CovarianceType};
use std::fmt;

/// Estrutura de entrada para o SUR
#[derive(Clone)]
pub struct SurEquation {
    pub y: Array1<f64>,
    pub x: Array2<f64>,
    pub name: String,
}

#[derive(Debug)]
pub struct SurResult {
    pub equations: Vec<SurEquationResult>,
    pub sigma_cross: Array2<f64>,
    pub system_r2: f64,
}

#[derive(Debug)]
pub struct SurEquationResult {
    pub name: String,
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
}

impl fmt::Display for SurResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Seemingly Unrelated Regressions (SUR) ")?;
        writeln!(f, "Zellner's Efficient Estimator")?;
        
        writeln!(f, "\n{:-^78}", " Cross-Equation Error Correlation (Sigma) ")?;
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

pub struct SUR;

impl SUR {
    /// Estima o modelo Zellner's SUR (Feasible GLS on System).
    /// Melhora a eficiência sobre o OLS quando os erros das equações são correlacionados.
    pub fn fit(equations: &[SurEquation]) -> Result<SurResult, GreenersError> {
        let n_obs = equations[0].y.len();
        let n_eq = equations.len();

        // 1. Passo OLS: Obter resíduos iniciais para estimar Sigma
        let mut residuals_ols = Array2::<f64>::zeros((n_obs, n_eq));
        
        for (i, eq) in equations.iter().enumerate() {
            if eq.y.len() != n_obs {
                return Err(GreenersError::ShapeMismatch("All equations must have same N observations".into()));
            }
            
            // Rodar OLS simples
            let ols = OLS::fit(&eq.y, &eq.x, CovarianceType::NonRobust)?;
            
            // Recalcular resíduos (y - Xb)
            let pred = eq.x.dot(&ols.params);
            let u = &eq.y - &pred;
            residuals_ols.column_mut(i).assign(&u);
        }

        // 2. Estimar Matriz de Covariância dos Erros (Sigma)
        // Sigma = (u'u) / N
        let sigma = residuals_ols.t().dot(&residuals_ols) / (n_obs as f64);
        
        // Inverter Sigma para usar no GLS
        let sigma_inv = sigma.inv().map_err(|_| GreenersError::SingularMatrix)?;

        // 3. Montar Sistema GLS (Kronecker Product implícito por blocos)
        // [X' (Sigma^-1 ox I) X] Beta = X' (Sigma^-1 ox I) y
        
        let mut k_total = 0;
        let mut k_per_eq = Vec::new();
        for eq in equations {
            let k = eq.x.ncols();
            k_per_eq.push(k);
            k_total += k;
        }

        let mut lhs = Array2::<f64>::zeros((k_total, k_total));
        let mut rhs = Array1::<f64>::zeros(k_total);

        let mut start_i = 0;
        for i in 0..n_eq {
            let ki = k_per_eq[i];
            let xi = &equations[i].x;
            
            let mut start_j = 0;
            for j in 0..n_eq {
                let kj = k_per_eq[j];
                let xj = &equations[j].x;
                
                // Elemento s^{ij} da inversa de Sigma
                let s_ij = sigma_inv[[i, j]];
                
                // Bloco LHS = s_ij * (Xi' * Xj)
                let block = xi.t().dot(xj) * s_ij;
                lhs.slice_mut(ndarray::s![start_i..start_i+ki, start_j..start_j+kj]).assign(&block);

                // Bloco RHS (acumulado na linha i)
                // RHS_i += s_ij * (Xi' * yj)
                let yj = &equations[j].y;
                let vec_part = xi.t().dot(yj) * s_ij;
                let mut target_slice = rhs.slice_mut(ndarray::s![start_i..start_i+ki]);
                target_slice += &vec_part;

                start_j += kj;
            }
            start_i += ki;
        }

        // 4. Resolver
        let lhs_inv = lhs.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let beta_sur = lhs_inv.dot(&rhs);

        // 5. Empacotar Resultados
        let mut final_results = Vec::new();
        let mut cursor = 0;
        let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        use statrs::distribution::ContinuousCDF;

        for (i, eq) in equations.iter().enumerate() {
            let k = k_per_eq[i];
            let params = beta_sur.slice(ndarray::s![cursor..cursor+k]).to_owned();
            
            // Variância: Bloco diagonal da inversa da Hessiana
            let cov_params = lhs_inv.slice(ndarray::s![cursor..cursor+k, cursor..cursor+k]).to_owned();
            let std_errors = cov_params.diag().mapv(f64::sqrt);
            
            let t_values = &params / &std_errors;
            let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

            // R2
            let pred = eq.x.dot(&params);
            let res = &eq.y - &pred;
            let sst = (&eq.y - eq.y.mean().unwrap()).mapv(|v| v.powi(2)).sum();
            let ssr = res.mapv(|v| v.powi(2)).sum();
            let r2 = 1.0 - (ssr / sst);

            final_results.push(SurEquationResult {
                name: eq.name.clone(),
                params,
                std_errors,
                t_values,
                p_values,
                r_squared: r2,
            });

            cursor += k;
        }

        Ok(SurResult {
            equations: final_results,
            sigma_cross: sigma,
            system_r2: 0.0, 
        })
    }
}