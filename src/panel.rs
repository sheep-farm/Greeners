use crate::{CovarianceType, DataFrame, Formula, GreenersError, OLS};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

// ===========================================================================
// FIXED EFFECTS (WITHIN ESTIMATOR)
// ===========================================================================

/// Struct to hold Fixed Effects estimation results.
#[derive(Debug)]
pub struct PanelResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64, // "Within" R-squared
    pub n_obs: usize,
    pub n_entities: usize, // Number of unique groups (N)
    pub df_resid: usize,   // Corrected degrees of freedom
    pub sigma: f64,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for PanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Fixed Effects (Within) Regression ")?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Dep. Variable:", "y", "Within R-sq:", self.r_squared
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Estimator:", "Fixed Effects", "No. Entities:", self.n_entities
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4e}",
            "No. Observations:", self.n_obs, "Sigma:", self.sigma
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let var_name = if let Some(ref names) = self.variable_names {
                if i < names.len() {
                    names[i].clone()
                } else {
                    format!("x{}", i)
                }
            } else {
                format!("x{}", i)
            };

            writeln!(
                f,
                "{:<10} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                var_name, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct FixedEffects;

impl FixedEffects {
    /// Estimates Fixed Effects model using a formula and DataFrame.
    /// Requires entity_ids to be passed separately.
    pub fn from_formula<T>(
        formula: &Formula,
        data: &DataFrame,
        entity_ids: &[T],
    ) -> Result<PanelResult, GreenersError>
    where
        T: Eq + Hash + Clone,
    {
        let (y, x) = data.to_design_matrix(formula)?;

        // Build variable names from formula (no intercept in FE)
        let var_names: Vec<String> = formula.independents.iter().cloned().map(|s| s.clone()).collect();

        Self::fit_with_names(&y, &x, entity_ids, Some(var_names))
    }

    /// Performs the "Within Transformation" (Demeaning) on a matrix/vector.
    /// x_dem = x_it - mean(x_i)
    fn within_transform<T>(data: &Array2<f64>, groups: &[T]) -> Result<Array2<f64>, GreenersError>
    where
        T: Eq + Hash + Clone,
    {
        let n_rows = data.nrows();
        let n_cols = data.ncols();

        if n_rows != groups.len() {
            return Err(GreenersError::ShapeMismatch(
                "Data rows and Group IDs length mismatch".into(),
            ));
        }

        // 1. Calculate sums and counts per group
        let mut group_sums: HashMap<T, Array1<f64>> = HashMap::new();
        let mut group_counts: HashMap<T, usize> = HashMap::new();

        for (i, group_id) in groups.iter().enumerate() {
            let row = data.row(i).to_owned();

            group_sums
                .entry(group_id.clone())
                .and_modify(|sum| *sum = &*sum + &row)
                .or_insert(row);

            *group_counts.entry(group_id.clone()).or_insert(0) += 1;
        }

        // 2. Subtract group means from original data
        let mut transformed_data = Array2::zeros((n_rows, n_cols));

        for (i, group_id) in groups.iter().enumerate() {
            let sum = &group_sums[group_id];
            let count = group_counts[group_id] as f64;
            let mean = sum / count;

            let original_row = data.row(i);
            let demeaned_row = &original_row - &mean;

            transformed_data.row_mut(i).assign(&demeaned_row);
        }

        Ok(transformed_data)
    }

    /// Fits the Fixed Effects model using Within Estimation.
    ///
    /// # Arguments
    /// * `y` - Dependent variable.
    /// * `x` - Regressors (DO NOT include a constant/intercept column!).
    /// * `groups` - Vector of Entity IDs (Integers, Strings, etc.) corresponding to rows.
    pub fn fit<T>(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[T],
    ) -> Result<PanelResult, GreenersError>
    where
        T: Eq + Hash + Clone,
    {
        Self::fit_with_names(y, x, groups, None)
    }

    pub fn fit_with_names<T>(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &[T],
        variable_names: Option<Vec<String>>,
    ) -> Result<PanelResult, GreenersError>
    where
        T: Eq + Hash + Clone,
    {
        let n = x.nrows();

        // 1. Convert y to Array2 for the generic transform function
        let y_mat = y.view().insert_axis(Axis(1)).to_owned();

        // 2. Apply Within Transformation
        let y_demeaned_mat = Self::within_transform(&y_mat, groups)?;
        let x_demeaned = Self::within_transform(x, groups)?;

        // Flatten y back to Array1
        let y_demeaned = y_demeaned_mat.column(0).to_owned();

        // 3. Run OLS on demeaned data
        let ols_result = OLS::fit(&y_demeaned, &x_demeaned, CovarianceType::NonRobust)?;

        // 4. Degrees of Freedom Correction
        let mut unique_groups: HashMap<T, bool> = HashMap::new();
        for g in groups {
            unique_groups.insert(g.clone(), true);
        }
        let n_entities = unique_groups.len();

        let k = x.ncols();
        let df_resid_correct = n - k - (n_entities - 1); // FE correction

        if df_resid_correct == 0 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough degrees of freedom for Fixed Effects".into(),
            ));
        }

        // Recalculate Sigma and Standard Errors with correct DF
        let residuals = &y_demeaned - &x_demeaned.dot(&ols_result.params);
        let ssr = residuals.dot(&residuals);

        let sigma2 = ssr / (df_resid_correct as f64);
        let sigma = sigma2.sqrt();

        // Adjust Covariance Matrix: Multiply by (OLS_DF / FE_DF) adjustment
        let adjustment_factor = (ols_result.df_resid as f64) / (df_resid_correct as f64);

        let old_vars = ols_result.std_errors.mapv(|se| se.powi(2));
        let new_vars = old_vars * adjustment_factor;
        let std_errors = new_vars.mapv(f64::sqrt);

        let t_values = &ols_result.params / &std_errors;

        Ok(PanelResult {
            params: ols_result.params,
            std_errors,
            t_values,
            p_values: ols_result.p_values,
            r_squared: ols_result.r_squared,
            n_obs: n,
            n_entities,
            df_resid: df_resid_correct,
            sigma,
            variable_names,
        })
    }
}

// ===========================================================================
// RANDOM EFFECTS (SWAMY-ARORA GLS)
// ===========================================================================

#[derive(Debug)]
pub struct RandomEffectsResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared_overall: f64,
    pub sigma_u: f64, // Desvio padrão do erro idiossincrático
    pub sigma_e: f64, // Desvio padrão do efeito individual
    pub theta: f64,   // Peso da transformação GLS
}

impl fmt::Display for RandomEffectsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Random Effects (GLS) - Swamy-Arora ")?;
        writeln!(
            f,
            "{:<20} {:>15.4}",
            "R-squared (Over):", self.r_squared_overall
        )?;
        writeln!(f, "{:<20} {:>15.4}", "Theta:", self.theta)?;
        writeln!(f, "{:<20} {:>15.4}", "Sigma Alpha (Ind):", self.sigma_e)?;
        writeln!(f, "{:<20} {:>15.4}", "Sigma U (Idiosync):", self.sigma_u)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
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

pub struct RandomEffects;

impl RandomEffects {
    /// Estimates Random Effects model using a formula and DataFrame.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        entity_ids: &Array1<i64>,
    ) -> Result<RandomEffectsResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        Self::fit(&y, &x, entity_ids)
    }

    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &Array1<i64>, // IDs dos indivíduos/empresas
    ) -> Result<RandomEffectsResult, GreenersError> {
        let n_obs = y.len();
        let k = x.ncols();

        if entity_ids.len() != n_obs {
            return Err(GreenersError::ShapeMismatch(
                "Entity IDs length mismatch".into(),
            ));
        }

        // 1. Mapear Índices por Entidade
        let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for (idx, &id) in entity_ids.iter().enumerate() {
            // CORREÇÃO: or_insert em vez de or_insert_vec
            groups.entry(id).or_default().push(idx);
        }

        let n_entities = groups.len();
        // Assumindo painel balanceado para cálculo simplificado do Theta (T médio)
        let t_bar = (n_obs as f64) / (n_entities as f64);

        // --- PASSO 2: Estimar Variâncias (Swamy-Arora) ---

        // A. Variância Within (Fixed Effects) -> Sigma_u
        // Transformação Within manual: (x_it - x_i_bar)
        let mut y_within = y.clone();
        let mut x_within = x.clone();

        // B. Variância Between (Médias) -> Para Sigma_e
        let mut y_means = Vec::new();
        let mut x_means = Vec::new();

        for indices in groups.values() {
            let t_i = indices.len() as f64;

            // Calcular médias do grupo
            let mut y_sum = 0.0;
            let mut x_sum = Array1::<f64>::zeros(k);

            for &idx in indices {
                y_sum += y[idx];
                x_sum = &x_sum + &x.row(idx);
            }
            let y_mean = y_sum / t_i;
            let x_mean = x_sum / t_i;

            y_means.push(y_mean);
            // Flatten x_mean para o vetor de between
            for val in x_mean.iter() {
                x_means.push(*val);
            }

            // Subtrair média (Within Transformation)
            for &idx in indices {
                y_within[idx] -= y_mean;
                let mut row = x_within.row_mut(idx);
                row -= &x_mean;
            }
        }

        // --- CORREÇÃO DE SINGULARIDADE (CRUCIAL) ---
        // Ao subtrair a média, colunas constantes (como intercepto) viram zero.
        // Isso quebra a inversão de matriz do OLS. Precisamos filtrar essas colunas
        // apenas para o passo intermediário de calcular Sigma_u.
        let mut keep_indices = Vec::new();
        for j in 0..k {
            let col = x_within.column(j);
            let variance = col.var(0.0); // Variância populacional da coluna
            if variance > 1e-12 {
                // Se tem variação, mantemos
                keep_indices.push(j);
            }
        }

        // Selecionar apenas colunas que variam
        let x_within_clean = x_within.select(Axis(1), &keep_indices);

        // Rodar OLS nos dados filtrados (sem intercepto) para pegar Sigma_u
        let fe_model = OLS::fit(&y_within, &x_within_clean, CovarianceType::NonRobust)?;

        // Calcular resíduos usando os dados filtrados e betas obtidos
        let residuals_fe = &y_within - &x_within_clean.dot(&fe_model.params);
        let ssr_within = residuals_fe.mapv(|v| v.powi(2)).sum();

        // Graus de liberdade corrigidos (usando colunas efetivas)
        let k_eff = keep_indices.len();
        let df_resid_within = (n_obs as f64 - n_entities as f64 - k_eff as f64).max(1.0);
        let sigma_u_sq = ssr_within / df_resid_within;

        // Rodar OLS nos dados Between (Médias)
        let y_between_arr = Array1::from(y_means);
        // CORREÇÃO: Tratamento de erro no from_shape_vec
        let x_between_arr = Array2::from_shape_vec((n_entities, k), x_means)
            .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

        let be_model = OLS::fit(&y_between_arr, &x_between_arr, CovarianceType::NonRobust)?;

        let residuals_be = &y_between_arr - &x_between_arr.dot(&be_model.params);
        let ssr_between = residuals_be.mapv(|v| v.powi(2)).sum();

        // Variância composta do between = sigma_u^2 / T + sigma_e^2
        let df_resid_between = (n_entities as f64 - k as f64).max(1.0);
        let sigma_b_sq = ssr_between / df_resid_between;

        // Recuperar Sigma_e (Efeito Individual)
        // sigma_e^2 = sigma_b^2 - (sigma_u^2 / T)
        let sigma_e_sq = (sigma_b_sq - (sigma_u_sq / t_bar)).max(0.0); // max(0) para evitar variância negativa

        // --- PASSO 3: Transformação GLS (Theta) ---
        let theta = 1.0 - (sigma_u_sq / (sigma_u_sq + t_bar * sigma_e_sq)).sqrt();

        // --- PASSO 4: Transformar Dados Finais ---
        // y* = y_it - theta * y_i_bar
        let mut y_gls = y.clone();
        let mut x_gls = x.clone();

        for indices in groups.values() {
            let t_i = indices.len() as f64;

            // Recalcular médias (rápido)
            let mut y_sum = 0.0;
            let mut x_sum = Array1::<f64>::zeros(k);
            for &idx in indices {
                y_sum += y[idx];
                x_sum = &x_sum + &x.row(idx);
            }
            let y_mean = y_sum / t_i;
            let x_mean = x_sum / t_i;

            // Aplicar Quasi-Diferença
            for &idx in indices {
                y_gls[idx] -= theta * y_mean;
                let mut row = x_gls.row_mut(idx);
                row -= &(&x_mean * theta);
            }
        }

        // --- PASSO 5: OLS Final ---
        let final_model = OLS::fit(&y_gls, &x_gls, CovarianceType::NonRobust)?;

        // R2 Overall (Correlação entre Y predito e Y real original)
        let pred_original = x.dot(&final_model.params);
        let y_mean_total = y.mean().unwrap();
        let sst = (y - y_mean_total).mapv(|v| v.powi(2)).sum();
        let ssr = (y - &pred_original).mapv(|v| v.powi(2)).sum();
        let r2_overall = 1.0 - (ssr / sst);

        Ok(RandomEffectsResult {
            params: final_model.params,
            std_errors: final_model.std_errors,
            t_values: final_model.t_values,
            p_values: final_model.p_values,
            r_squared_overall: r2_overall,
            sigma_u: sigma_u_sq.sqrt(),
            sigma_e: sigma_e_sq.sqrt(),
            theta,
        })
    }
}

#[derive(Debug)]
pub struct BetweenResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub n_entities: usize,
}

impl fmt::Display for BetweenResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Between Estimator (Means) ")?;
        writeln!(f, "{:<20} {:>15.4}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>15}", "No. Entities:", self.n_entities)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
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

pub struct BetweenEstimator;

impl BetweenEstimator {
    /// Estimates Between model using a formula and DataFrame.
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        entity_ids: &Array1<i64>,
    ) -> Result<BetweenResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        Self::fit(&y, &x, entity_ids)
    }

    /// Estima a regressão nas médias temporais de cada indivíduo.
    /// y_bar_i = alpha + beta * x_bar_i + (alpha_i + u_bar_i)
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &Array1<i64>,
    ) -> Result<BetweenResult, GreenersError> {
        let n_obs = y.len();
        let k = x.ncols();

        if entity_ids.len() != n_obs {
            return Err(GreenersError::ShapeMismatch(
                "Entity IDs length mismatch".into(),
            ));
        }

        // 1. Agrupar por Entidade
        let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for (idx, &id) in entity_ids.iter().enumerate() {
            groups.entry(id).or_default().push(idx);
        }

        let n_entities = groups.len();

        // 2. Calcular Médias (Collapse)
        let mut y_means = Vec::with_capacity(n_entities);
        let mut x_means = Vec::with_capacity(n_entities * k);

        // Iterar sobre os grupos para criar o dataset reduzido (N x K)
        for indices in groups.values() {
            let t_i = indices.len() as f64;

            let mut y_sum = 0.0;
            let mut x_sum = Array1::<f64>::zeros(k);

            for &idx in indices {
                y_sum += y[idx];
                x_sum = &x_sum + &x.row(idx);
            }

            y_means.push(y_sum / t_i);

            let x_mean = x_sum / t_i;
            for val in x_mean.iter() {
                x_means.push(*val);
            }
        }

        let y_between = Array1::from(y_means);
        let x_between = Array2::from_shape_vec((n_entities, k), x_means)
            .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

        // 3. Rodar OLS no dataset colapsado
        let ols = OLS::fit(&y_between, &x_between, CovarianceType::NonRobust)?;

        Ok(BetweenResult {
            params: ols.params,
            std_errors: ols.std_errors,
            t_values: ols.t_values,
            p_values: ols.p_values,
            r_squared: ols.r_squared,
            n_entities,
        })
    }
}
