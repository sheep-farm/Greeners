use ndarray::{Array1, Array2, Axis, s};
use crate::{OLS, CovarianceType, GreenersError};
use std::fmt;

/// Resultado do FGLS
#[derive(Debug)]
pub struct FglsResult {
    pub method: String,     // "WLS" ou "Cochrane-Orcutt"
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub rho: Option<f64>,   // Apenas para Cochrane-Orcutt
    pub iter: Option<usize>,// Iterações até convergência
}

impl fmt::Display for FglsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", format!(" FGLS Estimation ({}) ", self.method))?;
        if let Some(r) = self.rho {
            writeln!(f, "Autocorrelation (rho): {:>10.4}", r)?;
        }
        writeln!(f, "{:<20} {:>15.4}", "R-squared:", self.r_squared)?;
        
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}", 
            "Variable", "coef", "std err", "t", "P>|t|")?;
        writeln!(f, "{:-^78}", "")?;
        
        for i in 0..self.params.len() {
            writeln!(f, "x{:<9} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}", 
                i, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct FGLS;

impl FGLS {
    /// Weighted Least Squares (WLS)
    /// Usado quando existe heterocedasticidade conhecida ou estimada.
    /// Weights (w) devem ser inversamente proporcionais à variância: w_i = 1 / sigma_i^2
    pub fn wls(
        y: &Array1<f64>,
        x: &Array2<f64>,
        weights: &Array1<f64>
    ) -> Result<FglsResult, GreenersError> {
        let n = y.len();
        if weights.len() != n {
            return Err(GreenersError::ShapeMismatch("Weights length mismatch".into()));
        }

        // Transformação GLS: Multiplicar X e y pela raiz quadrada dos pesos
        // y* = sqrt(w) * y
        // X* = sqrt(w) * X
        let sqrt_w = weights.mapv(f64::sqrt);
        
        let y_transformed = y * &sqrt_w;
        
        // Multiplicação broadcast da coluna de pesos pelas linhas de X
        let mut x_transformed = x.clone();
        for (i, mut row) in x_transformed.axis_iter_mut(Axis(0)).enumerate() {
            row *= sqrt_w[i];
        }

        // Rodar OLS nos dados transformados
        let ols = OLS::fit(&y_transformed, &x_transformed, CovarianceType::NonRobust)?;

        Ok(FglsResult {
            method: "WLS".to_string(),
            params: ols.params,
            std_errors: ols.std_errors,
            t_values: ols.t_values,
            p_values: ols.p_values,
            r_squared: ols.r_squared,
            rho: None,
            iter: None,
        })
    }

    /// Cochrane-Orcutt Iterative Procedure (AR(1))
    /// Resolve correlação serial: u_t = rho * u_{t-1} + e_t
    /// Recupera a eficiência (BLUE) que o OLS perde.
    pub fn cochrane_orcutt(
        y: &Array1<f64>,
        x: &Array2<f64>
    ) -> Result<FglsResult, GreenersError> {
        let n = y.len();
        // let k = x.ncols();
        let tol = 1e-6;
        let max_iter = 100;

        // 1. OLS Inicial para pegar resíduos
        let initial_ols = OLS::fit(y, x, CovarianceType::NonRobust)?;
        let mut residuals = y - &x.dot(&initial_ols.params);
        
        let mut rho = 0.0;
        let mut iter = 0;
        let mut diff = 1.0;
        let mut final_ols = initial_ols; // Placeholder

        while diff > tol && iter < max_iter {
            let old_rho = rho;

            // 2. Estimar Rho regressando resid[t] em resid[t-1]
            let u_t = residuals.slice(s![1..]).to_owned();
            let u_tm1 = residuals.slice(s![..n-1]).to_owned();
            
            // Regressão simples sem intercepto: rho = (u_{t-1}' u_{t-1})^-1 u_{t-1}' u_t
            let num = u_tm1.dot(&u_t);
            let den = u_tm1.dot(&u_tm1);
            rho = num / den;

            // 3. Transformação Cochrane-Orcutt (Quase-Diferença)
            // y*_t = y_t - rho * y_{t-1}
            // x*_t = x_t - rho * x_{t-1}
            // Nota: Perdemos a primeira observação (n vira n-1)
            let y_t = y.slice(s![1..]);
            let y_tm1 = y.slice(s![..n-1]);
            let y_star = &y_t - &(&y_tm1 * rho);

            let x_t = x.slice(s![1.., ..]);
            let x_tm1 = x.slice(s![..n-1, ..]);
            let x_star = &x_t - &(&x_tm1 * rho);

            // 4. Re-estimar OLS nos dados transformados
            final_ols = OLS::fit(&y_star.to_owned(), &x_star.to_owned(), CovarianceType::NonRobust)?;
            
            // Atualizar resíduos ORIGINAIS (não transformados) para próxima iteração
            residuals = y - &x.dot(&final_ols.params);

            diff = (rho - old_rho).abs();
            iter += 1;
        }

        Ok(FglsResult {
            method: "Cochrane-Orcutt AR(1)".to_string(),
            params: final_ols.params,
            std_errors: final_ols.std_errors,
            t_values: final_ols.t_values,
            p_values: final_ols.p_values,
            r_squared: final_ols.r_squared,
            rho: Some(rho),
            iter: Some(iter),
        })
    }
}