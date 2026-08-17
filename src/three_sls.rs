use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2, Axis};
use statrs::distribution::ContinuousCDF;
use std::fmt;

/// Structure to define a single system equation
#[derive(Clone)]
pub struct Equation {
    pub y: Array1<f64>,
    pub x: Array2<f64>, //Includes endogenous and exogenous
    pub name: String,
    pub var_names: Vec<String>,
}

/// 3SLS System Result
#[derive(Debug)]
pub struct ThreeSLSResult {
    pub equations: Vec<EquationResult>,
    pub sigma_cross: Array2<f64>, //Covariance matrix of errors between equations
    pub system_r2: f64,           //McElroy's R2 (Optional but chic)
}

#[derive(Debug)]
pub struct EquationResult {
    pub name: String,
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub var_names: Vec<String>,
}

impl fmt::Display for ThreeSLSResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Three-Stage Least Squares (3SLS) System ")?;
        writeln!(f, "Number of Equations: {}", self.equations.len())?;

        //Show the Cross-Equation Correlation Correlation matrix
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
            writeln!(
                f,
                "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
                "Variable", "Coef", "Std Err", "t", "P>|t|"
            )?;
            writeln!(f, "{:-^78}", "")?;

            for i in 0..eq.params.len() {
                let label = eq
                    .var_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("x{i}"));
                writeln!(
                    f,
                    "{:<10} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                    label, eq.params[i], eq.std_errors[i], eq.t_values[i], eq.p_values[i]
                )?;
            }
            writeln!(f, "R-squared: {:.4}", eq.r_squared)?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct ThreeSLS;

/// Check if the instrument matrix already has a constant column.
/// If not, add a 1s column at the beginning. That makes it
/// projections of a first stage constant are accurate, allowing
/// which 3SLS equations include intercept when the frontend so specifies.
fn ensure_constant_instruments(z: &Array2<f64>) -> Array2<f64> {
    let n = z.nrows();
    if n == 0 {
        return z.clone();
    }

    let has_const = z
        .axis_iter(Axis(1))
        .any(|col| col.iter().all(|&v| (v - 1.0).abs() < 1e-12));

    if has_const {
        z.clone()
    } else {
        let mut z_out = Array2::<f64>::ones((n, z.ncols() + 1));
        z_out.slice_mut(ndarray::s![.., 1..]).assign(z);
        z_out
    }
}

impl ThreeSLS {
    /// Estimates a system of simultaneous equations via 3SLS.
    ///
    /// # Arguments
    /// * `equations` - Vector of structures `Equation` (each with y and X).
    /// * `z_instruments` - Matriz global de instrumentos (união de todas as exógenas).
    pub fn fit(
        equations: &[Equation],
        z_instruments: &Array2<f64>,
    ) -> Result<ThreeSLSResult, GreenersError> {
        let n_obs = z_instruments.nrows();
        let n_eq = equations.len();

        // --- STAGE 1: Reduced Form & Projection ---
        // Projetar cada X no espaço de Z para obter X_hat = Z(Z'Z)^-1 Z'X
        // X_hat é a versão "limpa" das endógenas.

        //Ensures that the instrument matrix includes a constant, so that
        //intercept projections are exact when the structural equation has
        //a column of 1s.
        let z_instruments = ensure_constant_instruments(z_instruments);

        // Pré-calcular P_z = Z (Z'Z)^-1 Z'
        // Para eficiência, calculamos apenas a parte (Z'Z)^-1 Z' e multiplicamos depois
        let z_t = z_instruments.t();
        let ztz = z_t.dot(&z_instruments);
        let ztz_inv = ztz.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let projection_matrix_part = z_instruments.dot(&ztz_inv).dot(&z_t); //N x N (Beware of memory here if N is huge)

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

            //residuals u = y - X * beta (We use the original X for residuals!)
            let pred = eq.x.dot(&beta_2sls);
            let u = &eq.y - &pred;

            //Save to next step
            residuals_2sls.column_mut(i).assign(&u);
            x_hat_list.push(x_hat);
        }

        //Calculate Error Covariance Matrix (Sigma)
        // Sigma_ij = (u_i' u_j) / N
        let sigma = residuals_2sls.t().dot(&residuals_2sls) / (n_obs as f64);
        let sigma_inv = sigma.inv().map_err(|_| GreenersError::SingularMatrix)?;

        // --- STAGE 3: GLS Estimation on the System ---
        // Resolver o sistema gigante: [X_hat' (Sigma^-1 ox I) X_hat] Beta = X_hat' (Sigma^-1 ox I) y

        //1. Count total parameters
        let mut k_total = 0;
        let mut k_per_eq = Vec::new();
        for eq in equations {
            let k = eq.x.ncols();
            k_per_eq.push(k);
            k_total += k;
        }

        //2. Build LHS Matrix (System Hessian) and RHS Vector
        //We use block construction to avoid explicit Kronecker.
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
                lhs_system
                    .slice_mut(ndarray::s![start_i..start_i + ki, start_j..start_j + kj])
                    .assign(&block);

                //Part of RHS (only when loop j runs, accumulates for i)
                // RHS_i = sum_j (s_ij * X_hat_i' * y_j)
                let y_j = &equations[j].y;
                let vec_part = x_hat_i.t().dot(y_j) * s_ij;

                //Add to RHS vector in position i
                let mut target_slice = rhs_system.slice_mut(ndarray::s![start_i..start_i + ki]);
                target_slice += &vec_part;

                start_j += kj;
            }
            start_i += ki;
        }

        // 3. Resolver Beta 3SLS
        let lhs_inv = lhs_system
            .inv()
            .map_err(|_| GreenersError::SingularMatrix)?;
        let beta_3sls_all = lhs_inv.dot(&rhs_system);

        //--- POST-STIMATION: Separate results and Statistics ---
        let mut final_results = Vec::new();
        let mut cursor = 0;

        for (i, eq) in equations.iter().enumerate() {
            let k = k_per_eq[i];
            let params = beta_3sls_all
                .slice(ndarray::s![cursor..cursor + k])
                .to_owned();

            //Variance Asymptotic coefficients of this equation
            //It is the corresponding diagonal block of the system inverse
            let cov_params = lhs_inv
                .slice(ndarray::s![cursor..cursor + k, cursor..cursor + k])
                .to_owned();
            let std_errors = cov_params.diag().mapv(f64::sqrt);

            //Statistics T and P
            let t_values = &params / &std_errors;
            let p_values = t_values
                .mapv(|t| 2.0 * (1.0 - statrs::distribution::Normal::standard().cdf(t.abs())));

            //R2 (Using 3SLS final residuals)
            let pred = eq.x.dot(&params);
            let res = &eq.y - &pred;
            let sst = (&eq.y
                - eq.y.mean().ok_or_else(|| {
                    GreenersError::InvalidOperation("Empty dependent variable".to_string())
                })?)
            .mapv(|v| v.powi(2))
            .sum();
            let ssr = res.mapv(|v| v.powi(2)).sum();
            let r2 = 1.0 - (ssr / sst);

            final_results.push(EquationResult {
                name: eq.name.clone(),
                params,
                std_errors,
                t_values,
                p_values,
                r_squared: r2,
                var_names: eq.var_names.clone(),
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
