use crate::error::GreenersError;
use ndarray as nd;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::fmt;

#[derive(Debug)]
pub struct GmmResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub j_stat: f64,    // Estatística de Hansen (J-test)
    pub j_p_value: f64, // P-valor do teste J
    pub n_obs: usize,
    pub df_model: usize,
    pub df_overid: usize, // Graus de liberdade de sobre-identificação
}

impl fmt::Display for GmmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " GMM (Two-Step Efficient) Results ")?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Dep. Variable:", "y", "J-Statistic:", self.j_stat
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Estimator:", "GMM", "Prob(J-Stat):", self.j_p_value
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "No. Observations:", self.n_obs, "DF Overid:", self.df_overid
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            writeln!(
                f,
                "x{:<9} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                i, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct GMM;

impl GMM {
    /// Fits a Linear GMM model using Two-Step Efficient weighting.
    /// Robust to Heteroskedasticity (White's Matrix for moments).
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        z: &Array2<f64>,
    ) -> Result<GmmResult, GreenersError> {
        let n = x.nrows();
        let k = x.ncols(); // Regressores
        let l = z.ncols(); // Instrumentos (Momentos)

        if l < k {
            return Err(GreenersError::ShapeMismatch(format!(
                "GMM requires L >= K (Instruments >= Params). Got L={}, K={}",
                l, k
            )));
        }

        // --- PREPARAÇÃO DAS MATRIZES ---
        // Momentos Amostrais: g = (1/n) * Z'u
        // Precisamos das matrizes cruzadas
        // let x_t = x.t();
        let z_t = z.t();

        // S_xz = (1/n) * Z'X
        let s_zx = z_t.dot(x) / (n as f64);
        // S_zy = (1/n) * Z'y
        let s_zy = z_t.dot(y) / (n as f64);

        // --- STEP 1: Matriz de Pesos Inicial (2SLS) ---
        // W1 = (Z'Z / n)^-1
        let s_zz = z_t.dot(z) / (n as f64);
        let w1 = s_zz.inv()?;

        // Estimador Beta 1 = ( (S_zx' W1 S_zx)^-1 ) * (S_zx' W1 S_zy)
        let s_zx_t = s_zx.t();
        let lhs_1 = s_zx_t.dot(&w1).dot(&s_zx);
        let rhs_1 = s_zx_t.dot(&w1).dot(&s_zy);
        let beta1 = lhs_1.inv()?.dot(&rhs_1);

        // Resíduos do Passo 1
        let pred1 = x.dot(&beta1);
        let resid1 = y - &pred1;

        // --- STEP 2: Matriz de Pesos Ótima (Hansen's S Matrix) ---
        // S = Lim Var (sqrt(n) * g_bar)
        // Estimador Robusto (White): S = (1/n) * sum( u_i^2 * z_i * z_i' )

        // Calcular S eficientemente: Z' * Diag(u^2) * Z
        let u_sq = resid1.mapv(|r| r.powi(2));
        let mut z_weighted = z.clone();
        for (i, mut row) in z_weighted.axis_iter_mut(nd::Axis(0)).enumerate() {
            row *= u_sq[i];
        }
        let s_matrix = z_t.dot(&z_weighted) / (n as f64);

        // W_opt = S^-1
        let w_opt = match s_matrix.inv() {
            Ok(mat) => mat,
            Err(_) => return Err(GreenersError::OptimizationFailed), // S singular
        };

        // --- STEP 3: Estimador Final GMM ---
        // Beta_GMM = (S_zx' W_opt S_zx)^-1 * S_zx' W_opt S_zy
        let lhs_final = s_zx_t.dot(&w_opt).dot(&s_zx);
        let rhs_final = s_zx_t.dot(&w_opt).dot(&s_zy);

        // Matriz de Variância Assintótica = (1/n) * (S_zx' W_opt S_zx)^-1
        // Note: Em GMM eficiente, a variância é o inverso da matriz LHS.
        let var_beta_matrix = lhs_final.inv()? / (n as f64);

        let beta_final = var_beta_matrix.dot(&rhs_final) * (n as f64); // Ajuste algébrico

        // --- ESTATÍSTICAS ---
        let std_errors = var_beta_matrix.diag().mapv(f64::sqrt);
        let t_values = &beta_final / &std_errors;

        // P-values (Normal padrão para assintótico)
        let dist = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        let p_values = t_values.mapv(|z| 2.0 * (1.0 - dist.cdf(z.abs())));

        // --- J-TEST (Overidentifying Restrictions) ---
        // J = n * g_bar(beta)' * W_opt * g_bar(beta)
        // g_bar = (1/n) * Z' * u_final
        let pred_final = x.dot(&beta_final);
        let resid_final = y - &pred_final;
        let g_bar = z_t.dot(&resid_final) / (n as f64);

        let j_stat = (n as f64) * g_bar.t().dot(&w_opt).dot(&g_bar);

        // Graus de liberdade do J-test = L - K
        let df_overid = l - k;
        let j_p_value = if df_overid > 0 {
            let chi2 =
                ChiSquared::new(df_overid as f64).map_err(|_| GreenersError::OptimizationFailed)?;
            1.0 - chi2.cdf(j_stat)
        } else {
            f64::NAN // Exact identified model, J should be ~0
        };

        Ok(GmmResult {
            params: beta_final,
            std_errors,
            t_values,
            p_values,
            j_stat,
            j_p_value,
            n_obs: n,
            df_model: k,
            df_overid,
        })
    }
}
