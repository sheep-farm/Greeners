use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use rand::distributions::Distribution;
use crate::{GreenersError, OLS, CovarianceType, DataFrame, Formula};
use std::fmt;

#[derive(Debug)]
pub struct QuantileResult {
    pub tau: f64,
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64, // Pseudo-R2 (Koenker & Machado)
    pub iterations: usize,
}

impl fmt::Display for QuantileResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", format!(" Quantile Regression (tau={:.2}) ", self.tau))?;
        writeln!(f, "{:<20} {:>15.4} (Pseudo)", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>15}", "Iterations:", self.iterations)?;
        
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}", 
            "Variable", "Coef", "Std Err", "t", "P>|t|")?;
        writeln!(f, "{:-^78}", "")?;
        
        for i in 0..self.params.len() {
            writeln!(f, "x{:<9} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}", 
                i, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct QuantileReg;

impl QuantileReg {
    /// Estimates Quantile Regression using a formula and DataFrame.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        tau: f64,
        n_boot: usize,
    ) -> Result<QuantileResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        Self::fit(&y, &x, tau, n_boot)
    }

    /// Estima a Regressão Quantílica via IRLS.
    ///
    /// # Arguments
    /// * `tau` - O quantil desejado (ex: 0.5 para mediana, 0.9 para decil superior).
    /// * `n_boot` - Número de repetições de Bootstrap para erro padrão (rec: 200+).
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        tau: f64,
        n_boot: usize
    ) -> Result<QuantileResult, GreenersError> {
        if tau <= 0.0 || tau >= 1.0 {
            return Err(GreenersError::OptimizationFailed); // "Tau must be in (0, 1)"
        }

        // 1. Estimação Pontual (IRLS)
        let (params, iter) = Self::irls_solver(y, x, tau)?;

        // 2. Erros Padrão via Bootstrap (Pairs Bootstrap)
        let std_errors = Self::bootstrap_se(y, x, tau, n_boot)?;

        // 3. Estatísticas Finais
        let t_values = &params / &std_errors;
        let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        use statrs::distribution::ContinuousCDF;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        // 4. Pseudo-R2 (Goodness of Fit)
        // R1 = 1 - (Loss(Model) / Loss(Null))
        // Null Model: Apenas intercepto (quantil da variável Y bruta)
        let loss_model = Self::check_loss(y, &x.dot(&params), tau);
        
        // Null model (apenas constante)
        let n = y.len();
        let ones = Array2::<f64>::ones((n, 1));
        let (params_null, _) = Self::irls_solver(y, &ones, tau)?;
        let loss_null = Self::check_loss(y, &ones.dot(&params_null), tau);
        
        let pseudo_r2 = 1.0 - (loss_model / loss_null);

        Ok(QuantileResult {
            tau,
            params,
            std_errors,
            t_values,
            p_values,
            r_squared: pseudo_r2,
            iterations: iter,
        })
    }

    /// Iteratively Reweighted Least Squares solver
    fn irls_solver(
        y: &Array1<f64>,
        x: &Array2<f64>,
        tau: f64
    ) -> Result<(Array1<f64>, usize), GreenersError> {
        let n = y.len();
        let max_iter = 1000;
        let tol = 1e-6;
        let epsilon = 1e-6; // Para evitar divisão por zero

        // Chute inicial: OLS
        let ols = OLS::fit(y, x, CovarianceType::NonRobust)?;
        let mut beta = ols.params;
        let mut diff = 1.0;
        let mut iter = 0;

        while diff > tol && iter < max_iter {
            let old_beta = beta.clone();
            
            // Calcular Resíduos
            let pred = x.dot(&beta);
            let residuals = y - &pred;

            // Calcular Pesos IRLS para Quantil
            // w_i = tau / |u| se u > 0
            // w_i = (1-tau) / |u| se u < 0
            // Aproximação suave: w_i = 1 / max(epsilon, |u|) * (tau if u>0 else 1-tau)
            
            let mut weights = Array1::<f64>::zeros(n);
            for i in 0..n {
                let u = residuals[i];
                let abs_u = u.abs().max(epsilon); // Evitar 0
                
                let w = if u >= 0.0 {
                    tau / abs_u
                } else {
                    (1.0 - tau) / abs_u
                };
                weights[i] = w;
            }

            // Rodar WLS (Weighted Least Squares)
            // Beta = (X' W X)^-1 X' W y
            // Truque algébrico: Multiplicar X e y por sqrt(w) e rodar OLS
            let sqrt_w = weights.mapv(f64::sqrt);
            let y_w = y * &sqrt_w;
            let mut x_w = x.clone();
            for (i, mut row) in x_w.axis_iter_mut(Axis(0)).enumerate() {
                row *= sqrt_w[i];
            }

            let wls_res = OLS::fit(&y_w, &x_w, CovarianceType::NonRobust);
            
            // Se WLS falhar (matriz singular devido a pesos extremos), retornar erro ou parar
            if let Ok(res) = wls_res {
                beta = res.params;
            } else {
                return Err(GreenersError::OptimizationFailed);
            }

            // Convergência baseada na mudança dos coeficientes
            diff = (&beta - &old_beta).mapv(|v| v.abs()).sum();
            iter += 1;
        }

        Ok((beta, iter))
    }

    /// Pairs Bootstrap para Erro Padrão
    fn bootstrap_se(
        y: &Array1<f64>,
        x: &Array2<f64>,
        tau: f64,
        n_boot: usize
    ) -> Result<Array1<f64>, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        let mut boot_betas = Array2::<f64>::zeros((n_boot, k));
        
        let mut rng = rand::thread_rng();

        for b in 0..n_boot {
            // Reamostragem com reposição
            let indices = Array1::from_iter((0..n).map(|_| Uniform::new(0, n).sample(&mut rng)));
            
            let mut y_boot_vec = Vec::with_capacity(n);
            let mut x_boot_vec = Vec::with_capacity(n * k);

            for &idx in &indices {
                y_boot_vec.push(y[idx]);
                for val in x.row(idx) {
                    x_boot_vec.push(*val);
                }
            }
            
            let y_boot = Array1::from(y_boot_vec);
            let x_boot = Array2::from_shape_vec((n, k), x_boot_vec)
                .map_err(|_| GreenersError::ShapeMismatch("Bootstrap shape error".into()))?;

            // Estimar no bootstrap
            // Se falhar (raro), apenas ignoramos essa iteração copiando a anterior (simplificação)
            if let Ok((beta_b, _)) = Self::irls_solver(&y_boot, &x_boot, tau) {
                boot_betas.row_mut(b).assign(&beta_b);
            }
        }

        // Calcular Desvio Padrão das estimativas bootstrap
        let mut std_errs = Array1::<f64>::zeros(k);
        for j in 0..k {
            let col = boot_betas.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var = col.mapv(|v| (v - mean).powi(2)).sum() / ((n_boot - 1) as f64);
            std_errs[j] = var.sqrt();
        }

        Ok(std_errs)
    }

    /// Função de Perda (Check Loss)
    fn check_loss(y: &Array1<f64>, y_pred: &Array1<f64>, tau: f64) -> f64 {
        let res = y - y_pred;
        res.mapv(|u| if u >= 0.0 { tau * u } else { (tau - 1.0) * u }).sum()
    }
}