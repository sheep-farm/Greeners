use crate::{CovarianceType, GreenersError, OLS};
use ndarray::{Array1, Array2};
use std::fmt;

/// Resultado do estimador Difference-in-Differences (Canônico 2x2)
#[derive(Debug)]
pub struct DidResult {
    pub att: f64,       // O efeito do tratamento (Coeficiente da interação)
    pub std_error: f64, // Erro padrão do ATT
    pub t_stat: f64,
    pub p_value: f64,
    pub n_obs: usize,
    pub r_squared: f64,
    pub control_pre_mean: f64,  // Média Controle (Pré)
    pub control_post_mean: f64, // Média Controle (Pós)
    pub treated_pre_mean: f64,  // Média Tratado (Pré)
    pub treated_post_mean: f64, // Média Tratado (Pós - Counterfactual vs Real)
    pub cov_type: CovarianceType,
}

impl fmt::Display for DidResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            " Difference-in-Differences (2x2 Canonical) "
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "ATT (Effect):", self.att, "R-squared:", self.r_squared
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "Std. Error:", self.std_error, "P-value:", self.p_value
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15}",
            "t-statistic:", self.t_stat, "Observations:", self.n_obs
        )?;

        writeln!(f, "\n{:-^78}", " Group Means ")?;
        writeln!(
            f,
            "Control Group (Pre):  {:>10.4} | Control Group (Post): {:>10.4}",
            self.control_pre_mean, self.control_post_mean
        )?;
        writeln!(
            f,
            "Treated Group (Pre):  {:>10.4} | Treated Group (Post): {:>10.4}",
            self.treated_pre_mean, self.treated_post_mean
        )?;
        writeln!(
            f,
            "Parallel Trend Diff:  {:>10.4} (If > 0, Control grew more than Treated Pre-trend)",
            (self.control_post_mean - self.control_pre_mean)
                - (self.treated_post_mean - self.treated_pre_mean - self.att)
        )?;
        writeln!(f, "{:=^78}", "")
    }
}

pub struct DiffInDiff;

impl DiffInDiff {
    /// Estima o modelo Canônico 2x2 DiD.
    ///
    /// # Arguments
    /// * `y` - Variável de resultado (Outcome).
    /// * `treated` - Dummy: 1 se pertence ao grupo de tratamento, 0 caso contrário.
    /// * `post` - Dummy: 1 se está no período pós-intervenção, 0 caso contrário.
    /// * `cov_type` - Tipo de covariância (Recomendado: HC1 ou Cluster se tivéssemos cluster ID).
    pub fn fit(
        y: &Array1<f64>,
        treated: &Array1<f64>,
        post: &Array1<f64>,
        cov_type: CovarianceType,
    ) -> Result<DidResult, GreenersError> {
        let n = y.len();
        if treated.len() != n || post.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "Input arrays must have same length".into(),
            ));
        }

        // 1. Construir Matriz X [Intercept, Treated, Post, Interaction]
        let mut x_mat = Array2::<f64>::zeros((n, 4));
        let mut interaction = Array1::<f64>::zeros(n);

        // Médias para display (Manual calculation for performance)
        let mut sum_c_pre = 0.0;
        let mut n_c_pre = 0.0;
        let mut sum_c_post = 0.0;
        let mut n_c_post = 0.0;
        let mut sum_t_pre = 0.0;
        let mut n_t_pre = 0.0;
        let mut sum_t_post = 0.0;
        let mut n_t_post = 0.0;

        for i in 0..n {
            let t = treated[i];
            let p = post[i];
            let inter = t * p; // D * T

            x_mat[[i, 0]] = 1.0; // Beta0
            x_mat[[i, 1]] = t; // Beta1 (Group Fixed Effect)
            x_mat[[i, 2]] = p; // Beta2 (Time Fixed Effect)
            x_mat[[i, 3]] = inter; // Delta (ATT)

            interaction[i] = inter;

            // Acumular médias
            let val = y[i];
            if t == 0.0 && p == 0.0 {
                sum_c_pre += val;
                n_c_pre += 1.0;
            } else if t == 0.0 && p == 1.0 {
                sum_c_post += val;
                n_c_post += 1.0;
            } else if t == 1.0 && p == 0.0 {
                sum_t_pre += val;
                n_t_pre += 1.0;
            } else if t == 1.0 && p == 1.0 {
                sum_t_post += val;
                n_t_post += 1.0;
            }
        }

        // 2. Rodar OLS
        let ols = OLS::fit(y, &x_mat, cov_type)?;

        // O ATT é o coeficiente da interação (índice 3)
        let att = ols.params[3];
        let std_error = ols.std_errors[3];
        let t_stat = ols.t_values[3];
        let p_value = ols.p_values[3];

        Ok(DidResult {
            att,
            std_error,
            t_stat,
            p_value,
            n_obs: n,
            r_squared: ols.r_squared,
            control_pre_mean: if n_c_pre > 0.0 {
                sum_c_pre / n_c_pre
            } else {
                0.0
            },
            control_post_mean: if n_c_post > 0.0 {
                sum_c_post / n_c_post
            } else {
                0.0
            },
            treated_pre_mean: if n_t_pre > 0.0 {
                sum_t_pre / n_t_pre
            } else {
                0.0
            },
            treated_post_mean: if n_t_post > 0.0 {
                sum_t_post / n_t_post
            } else {
                0.0
            },
            cov_type,
        })
    }
}
