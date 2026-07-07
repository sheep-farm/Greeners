use crate::linalg::LinalgInverse as _;
use crate::{CovarianceType, DataFrame, Formula, GreenersError, InferenceType, OLS};
use ndarray::{Array1, Array2, Axis};
use indexmap::IndexMap;
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
    pub inference_type: InferenceType,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for PanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stat_label = match self.inference_type {
            InferenceType::StudentT => "t",
            InferenceType::Normal => "z",
        };

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
            "Variable",
            "coef",
            "std err",
            stat_label,
            format!("P>|{}|", stat_label)
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

impl PanelResult {
    /// Change inference type and recompute p-values
    ///
    /// Allows switching between Student's t-distribution and Normal distribution
    /// for hypothesis testing after model fitting.
    ///
    /// # Arguments
    /// * `inference_type` - New distribution type
    ///
    /// # Returns
    /// Modified PanelResult with updated p-values
    pub fn with_inference(mut self, inference_type: InferenceType) -> Result<Self, GreenersError> {
        // Reuse OLS compute_inference helper
        use crate::ols::OlsResult;

        let (p_values, _, _) = OlsResult::compute_inference(
            &self.t_values,
            &self.std_errors,
            &self.params,
            self.df_resid,
            &inference_type,
        )?;

        self.p_values = p_values;
        self.inference_type = inference_type;

        Ok(self)
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
        let var_names: Vec<String> = formula.independents.to_vec();

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
        let mut group_sums: IndexMap<T, Array1<f64>> = IndexMap::new();
        let mut group_counts: IndexMap<T, usize> = IndexMap::new();

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
        let mut unique_groups: IndexMap<T, bool> = IndexMap::new();
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

        // Extract inference type from OLS result
        let inference_type = ols_result.inference_type.clone();

        // Recalculate p-values with corrected df_resid
        use crate::ols::OlsResult;
        let (p_values, _, _) = OlsResult::compute_inference(
            &t_values,
            &std_errors,
            &ols_result.params,
            df_resid_correct,
            &inference_type,
        )?;

        Ok(PanelResult {
            params: ols_result.params,
            std_errors,
            t_values,
            p_values,
            r_squared: ols_result.r_squared,
            n_obs: n,
            n_entities,
            df_resid: df_resid_correct,
            sigma,
            inference_type,
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
    pub inference_type: InferenceType,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for RandomEffectsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stat_label = match self.inference_type {
            InferenceType::StudentT => "t",
            InferenceType::Normal => "z",
        };

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
            "Variable",
            "Coef",
            "Std Err",
            stat_label,
            format!("P>|{}|", stat_label)
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let label = self
                .variable_names
                .as_ref()
                .and_then(|v| v.get(i))
                .cloned()
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<10} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                label, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

impl RandomEffectsResult {
    /// Change inference type and recompute p-values
    pub fn with_inference(mut self, inference_type: InferenceType) -> Result<Self, GreenersError> {
        use crate::ols::OlsResult;

        // Random Effects doesn't have df_resid field, so we compute it
        // df = n - k where n is observations and k is parameters
        let df_resid = self.params.len(); // This is approximate; ideally should be stored

        let (p_values, _, _) = OlsResult::compute_inference(
            &self.t_values,
            &self.std_errors,
            &self.params,
            df_resid,
            &inference_type,
        )?;

        self.p_values = p_values;
        self.inference_type = inference_type;

        Ok(self)
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
        let mut result = Self::fit(&y, &x, entity_ids)?;
        // to_design_matrix coloca intercepto primeiro quando formula.intercept == true
        let mut var_names: Vec<String> = if formula.intercept {
            let mut v = vec!["const".to_string()];
            v.extend(formula.independents.iter().cloned());
            v
        } else {
            formula.independents.clone()
        };
        var_names.truncate(result.params.len());
        result.variable_names = Some(var_names);
        Ok(result)
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
        let mut groups: IndexMap<i64, Vec<usize>> = IndexMap::new();
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
            inference_type: final_model.inference_type,
            variable_names: None,
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
    pub inference_type: InferenceType,
}

impl fmt::Display for BetweenResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stat_label = match self.inference_type {
            InferenceType::StudentT => "t",
            InferenceType::Normal => "z",
        };

        writeln!(f, "\n{:=^78}", " Between Estimator (Means) ")?;
        writeln!(f, "{:<20} {:>15.4}", "R-squared:", self.r_squared)?;
        writeln!(f, "{:<20} {:>15}", "No. Entities:", self.n_entities)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable",
            "Coef",
            "Std Err",
            stat_label,
            format!("P>|{}|", stat_label)
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

impl BetweenResult {
    /// Change inference type and recompute p-values
    pub fn with_inference(mut self, inference_type: InferenceType) -> Result<Self, GreenersError> {
        use crate::ols::OlsResult;

        // Between estimator uses n_entities as sample size
        let df_resid = self.n_entities.saturating_sub(self.params.len());

        let (p_values, _, _) = OlsResult::compute_inference(
            &self.t_values,
            &self.std_errors,
            &self.params,
            df_resid,
            &inference_type,
        )?;

        self.p_values = p_values;
        self.inference_type = inference_type;

        Ok(self)
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
        let mut groups: IndexMap<i64, Vec<usize>> = IndexMap::new();
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
            inference_type: ols.inference_type,
        })
    }
}

// ===========================================================================
// FE-2SLS (xtivreg, fe) — Hausman (1978)
// ===========================================================================

#[derive(Debug)]
pub struct PanelIvResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub n_obs: usize,
    pub n_entities: usize,
    pub df_resid: usize,
    pub sigma: f64,
    pub inference_type: InferenceType,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for PanelIvResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let thick = "═".repeat(70);
        let thin = "─".repeat(70);
        writeln!(f, "\n{thick}")?;
        writeln!(f, " FE-2SLS (xtivreg, fe)  —  Hausman (1978)")?;
        writeln!(f, "{thick}")?;
        writeln!(
            f,
            " Obs: {:<8}  Entidades: {:<8}  df_resid: {}",
            self.n_obs, self.n_entities, self.df_resid
        )?;
        writeln!(
            f,
            " R² (within): {:.6}   σ: {:.6}",
            self.r_squared, self.sigma
        )?;
        writeln!(f, "{thin}")?;
        writeln!(
            f,
            " {:<18} {:>12}  {:>12}  {:>8}  {:>8}",
            "Variável", "coef", "SE", "t", "P>|t|"
        )?;
        writeln!(f, " {}", "─".repeat(64))?;
        for i in 0..self.params.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|v| v.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i + 1));
            writeln!(
                f,
                " {:<18} {:>12.4}  {:>12.4}  {:>8.3}  {:>8.4}",
                name, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{thick}")
    }
}

pub struct FE2SLS;

impl FE2SLS {
    /// Estima FE-2SLS: `feiv(y ~ x1+x2, ~ x1+z1+z2, df, id=col)`
    ///
    /// * `y`      — variável dependente (n)
    /// * `x`      — regressores estruturais sem constante (n × k); coluna endógena incluída
    /// * `z`      — matriz de instrumentos completa sem constante (n × l), l ≥ k
    ///   deve incluir os regressores exógenos + instrumentos excluídos
    /// * `groups` — IDs de entidade para a transformação within
    pub fn fit<T>(
        y: &Array1<f64>,
        x: &Array2<f64>,
        z: &Array2<f64>,
        groups: &[T],
        variable_names: Option<Vec<String>>,
    ) -> Result<PanelIvResult, GreenersError>
    where
        T: Eq + Hash + Clone,
    {
        let n = y.len();
        let k = x.ncols();
        let l = z.ncols();

        if x.nrows() != n || z.nrows() != n || groups.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "FE2SLS: dimensões de y, x, z e grupos divergem".into(),
            ));
        }
        if l < k {
            return Err(GreenersError::ShapeMismatch(format!(
                "FE2SLS: condição de ordem violada — Z tem {l} instrumentos, X tem {k} regressores"
            )));
        }
        if y.iter().any(|v| !v.is_finite())
            || x.iter().any(|v| !v.is_finite())
            || z.iter().any(|v| !v.is_finite())
        {
            return Err(GreenersError::InvalidOperation(
                "FE2SLS: dados contêm NaN ou Inf".into(),
            ));
        }

        // ── 1. Transformação within (demean) ──
        let y_mat = y.view().insert_axis(Axis(1)).to_owned();
        let y_dm = FixedEffects::within_transform(&y_mat, groups)?;
        let x_dm = FixedEffects::within_transform(x, groups)?;
        let z_dm = FixedEffects::within_transform(z, groups)?;

        let y_d: Array1<f64> = y_dm.column(0).to_owned();

        // ── 2. 2SLS: primeira etapa X̂ = P_Z X̃ ──
        let zt = z_dm.t();
        let zt_z = zt.dot(&z_dm);
        let zt_z_inv = zt_z.inv()?;
        let zt_x = zt.dot(&x_dm);
        let x_hat = z_dm.dot(&zt_z_inv.dot(&zt_x));

        // ── 3. Segunda etapa: β = (X̂'X̃)⁻¹ X̂'ỹ ──
        let xht = x_hat.t();
        let xht_xd = xht.dot(&x_dm);
        let xht_xd_inv = xht_xd.inv()?;
        let beta = xht_xd_inv.dot(&xht.dot(&y_d));

        // ── 4. Resíduos e graus de liberdade ──
        let fitted = x_dm.dot(&beta);
        let resid = &y_d - &fitted;
        let ssr = resid.dot(&resid);

        let n_entities = {
            let mut seen = IndexMap::new();
            for g in groups {
                seen.insert(g.clone(), ());
            }
            seen.len()
        };
        // FE consome N-1 graus de liberdade (efeitos individuais demeaned)
        let df_resid = n.saturating_sub(k).saturating_sub(n_entities - 1);
        if df_resid == 0 {
            return Err(GreenersError::ShapeMismatch(
                "FE2SLS: graus de liberdade insuficientes".into(),
            ));
        }

        let sigma2 = ssr / df_resid as f64;
        let sigma = sigma2.sqrt();

        // ── 5. V = σ² (X̂'X̃)⁻¹ ──
        let cov_mat = &xht_xd_inv * sigma2;
        let std_errors: Array1<f64> = (0..k)
            .map(|i| cov_mat[[i, i]].max(0.0).sqrt())
            .collect::<Vec<_>>()
            .into();

        let t_values = &beta / &std_errors;

        // ── 6. p-values (t com df_resid) ──
        use statrs::distribution::{ContinuousCDF, StudentsT};
        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values: Array1<f64> = t_values
            .iter()
            .map(|&t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
            .collect::<Vec<_>>()
            .into();

        // ── 7. R² within: corr²(ỹ, X̃β) ──
        let r_squared = {
            let ymean = y_d.mean().unwrap_or(0.0);
            let ss_tot: f64 = y_d.iter().map(|&v| (v - ymean).powi(2)).sum();
            let ss_res: f64 = ssr;
            if ss_tot > 1e-15 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            }
        };

        Ok(PanelIvResult {
            params: beta,
            std_errors,
            t_values,
            p_values,
            r_squared,
            n_obs: n,
            n_entities,
            df_resid,
            sigma,
            inference_type: InferenceType::StudentT,
            variable_names,
        })
    }
}

// ===========================================================================
// Helpers compartilhados — extração de painel balanceado
// ===========================================================================

type BalancedPanelResult = (Vec<i64>, Vec<Array1<f64>>, Vec<Array2<f64>>, usize);

/// Extrai submatrizes por entidade a partir de dados longos, ordenando por
/// (entity_id, time_id). Exige painel balanceado (T igual para todas as entidades).
/// Retorna (entidades_ordenadas, y_panels, x_panels, T).
fn extract_balanced_panels(
    y: &Array1<f64>,
    x: &Array2<f64>,
    entity_ids: &[i64],
    time_ids: &[i64],
) -> Result<BalancedPanelResult, GreenersError> {
    let n = y.len();
    if x.nrows() != n || entity_ids.len() != n || time_ids.len() != n {
        return Err(GreenersError::ShapeMismatch(
            "dimensões de y, x, entity_ids, time_ids divergem".into(),
        ));
    }

    // Ordena índices por (entity, time)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (entity_ids[i], time_ids[i]));

    // Coleta entidades únicas em ordem
    let mut entities: Vec<i64> = Vec::new();
    for &i in &order {
        let e = entity_ids[i];
        if entities.last() != Some(&e) {
            entities.push(e);
        }
    }

    // Conta T por entidade e verifica balance
    let mut counts: IndexMap<i64, usize> = IndexMap::new();
    for &i in &order {
        *counts.entry(entity_ids[i]).or_insert(0) += 1;
    }
    let t_vec: Vec<usize> = entities.iter().map(|e| counts[e]).collect();
    let t0 = t_vec[0];
    if t_vec.iter().any(|&t| t != t0) {
        return Err(GreenersError::InvalidOperation(
            "painel não balanceado: número de períodos difere entre entidades".into(),
        ));
    }

    let k = x.ncols();
    let mut y_panels: Vec<Array1<f64>> = Vec::with_capacity(entities.len());
    let mut x_panels: Vec<Array2<f64>> = Vec::with_capacity(entities.len());

    for &eid in &entities {
        let rows: Vec<usize> = order
            .iter()
            .filter(|&&i| entity_ids[i] == eid)
            .copied()
            .collect();
        let yi: Array1<f64> = rows.iter().map(|&i| y[i]).collect::<Vec<_>>().into();
        let mut xi = Array2::<f64>::zeros((t0, k));
        for (r, &src) in rows.iter().enumerate() {
            xi.row_mut(r).assign(&x.row(src));
        }
        y_panels.push(yi);
        x_panels.push(xi);
    }

    Ok((entities, y_panels, x_panels, t0))
}

fn t_pvalues(t_vals: &Array1<f64>, df: usize) -> Result<Array1<f64>, GreenersError> {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    let dist = StudentsT::new(0.0, 1.0, df as f64)
        .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
    Ok(t_vals.mapv(|t| 2.0 * (1.0 - dist.cdf(t.abs()))))
}

// ===========================================================================
// PCSE — Panel-Corrected Standard Errors (Beck & Katz 1995)
// Stata: xtpcse y x1 x2, id(firm) t(year)
// ===========================================================================

#[derive(Debug)]
pub struct PcseResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub n_obs: usize,
    pub n_entities: usize,
    pub t_periods: usize,
    pub df_resid: usize,
    pub sigma: f64,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for PcseResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let thick = "═".repeat(70);
        let thin = "─".repeat(70);
        writeln!(f, "\n{thick}")?;
        writeln!(
            f,
            " PCSE — Panel-Corrected Standard Errors  (Beck & Katz 1995)"
        )?;
        writeln!(f, "{thick}")?;
        writeln!(
            f,
            " Obs: {:<8}  Entidades: {:<6}  Períodos: {:<6}  df_resid: {}",
            self.n_obs, self.n_entities, self.t_periods, self.df_resid
        )?;
        writeln!(f, " R²: {:.6}   σ (OLS): {:.6}", self.r_squared, self.sigma)?;
        writeln!(f, "{thin}")?;
        writeln!(
            f,
            " {:<18} {:>12}  {:>12}  {:>8}  {:>8}",
            "Variável", "coef", "PCSE", "t", "P>|t|"
        )?;
        writeln!(f, " {}", "─".repeat(64))?;
        for i in 0..self.params.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|v| v.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i + 1));
            writeln!(
                f,
                " {:<18} {:>12.4}  {:>12.4}  {:>8.3}  {:>8.4}",
                name, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{thick}")
    }
}

pub struct PCSE;

impl PCSE {
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
        time_ids: &[i64],
        variable_names: Option<Vec<String>>,
    ) -> Result<PcseResult, GreenersError> {
        if y.iter().any(|v| !v.is_finite()) || x.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "PCSE: dados contêm NaN ou Inf".into(),
            ));
        }

        let (_, y_panels, x_panels, big_t) = extract_balanced_panels(y, x, entity_ids, time_ids)?;

        let n_entities = y_panels.len();
        let n_obs = n_entities * big_t;
        let k = x.ncols();
        let df_resid = n_obs.saturating_sub(k);

        // ── OLS β̂ = (X'X)⁻¹ X'y ──
        let xtx: Array2<f64> = x_panels
            .iter()
            .fold(Array2::zeros((k, k)), |acc, xi| acc + xi.t().dot(xi));
        let xty: Array1<f64> = x_panels
            .iter()
            .zip(y_panels.iter())
            .fold(Array1::zeros(k), |acc, (xi, yi)| acc + xi.t().dot(yi));
        let xtx_inv = xtx.inv()?;
        let beta = xtx_inv.dot(&xty);

        // ── Resíduos e σ ──
        let resid_panels: Vec<Array1<f64>> = y_panels
            .iter()
            .zip(x_panels.iter())
            .map(|(yi, xi)| yi - &xi.dot(&beta))
            .collect();
        let ssr: f64 = resid_panels.iter().map(|e| e.dot(e)).sum();
        let sigma = (ssr / df_resid as f64).sqrt();

        // ── Σ̂_ij = e_i'e_j / T ──
        let n = n_entities;
        let mut sigma_hat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let s = resid_panels[i].dot(&resid_panels[j]) / big_t as f64;
                sigma_hat[[i, j]] = s;
                sigma_hat[[j, i]] = s;
            }
        }

        // ── Meat = Σ_i Σ_j σ̂_ij X_i'X_j ──
        let mut meat = Array2::<f64>::zeros((k, k));
        for i in 0..n {
            for j in 0..n {
                meat = meat + x_panels[i].t().dot(&x_panels[j]) * sigma_hat[[i, j]];
            }
        }

        // ── V_PCSE = (X'X)⁻¹ Meat (X'X)⁻¹ ──
        let v = xtx_inv.dot(&meat).dot(&xtx_inv);
        let std_errors: Array1<f64> = (0..k)
            .map(|i| v[[i, i]].max(0.0).sqrt())
            .collect::<Vec<_>>()
            .into();
        let t_values = &beta / &std_errors;
        let p_values = t_pvalues(&t_values, df_resid)?;

        // ── R² ──
        let ymean = y.mean().unwrap_or(0.0);
        let ss_tot: f64 = y.iter().map(|&v| (v - ymean).powi(2)).sum();
        let r_squared = if ss_tot > 1e-15 {
            1.0 - ssr / ss_tot
        } else {
            0.0
        };

        Ok(PcseResult {
            params: beta,
            std_errors,
            t_values,
            p_values,
            r_squared,
            n_obs,
            n_entities,
            t_periods: big_t,
            df_resid,
            sigma,
            variable_names,
        })
    }
}

// ===========================================================================
// Panel GLS — Parks (1967) / Stata xtgls
// panels=hetero : σ²_i por entidade (diagonal Σ)
// panels=corr   : Σ completa entre entidades (Parks clássico)
// ===========================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum GlsPanels {
    Hetero,
    Correlated,
}

#[derive(Debug)]
pub struct PanelGlsResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub n_obs: usize,
    pub n_entities: usize,
    pub t_periods: usize,
    pub df_resid: usize,
    pub sigma: f64,
    pub panels: GlsPanels,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for PanelGlsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let method = match self.panels {
            GlsPanels::Hetero => "heteroscedastic (diagonal Σ)",
            GlsPanels::Correlated => "correlated (Parks, Σ completa)",
        };
        let thick = "═".repeat(70);
        let thin = "─".repeat(70);
        writeln!(f, "\n{thick}")?;
        writeln!(f, " Panel GLS  —  panels({})", method)?;
        writeln!(f, "{thick}")?;
        writeln!(
            f,
            " Obs: {:<8}  Entidades: {:<6}  Períodos: {:<6}  df_resid: {}",
            self.n_obs, self.n_entities, self.t_periods, self.df_resid
        )?;
        writeln!(f, " R²: {:.6}   σ (GLS): {:.6}", self.r_squared, self.sigma)?;
        writeln!(f, "{thin}")?;
        writeln!(
            f,
            " {:<18} {:>12}  {:>12}  {:>8}  {:>8}",
            "Variável", "coef", "SE", "z", "P>|z|"
        )?;
        writeln!(f, " {}", "─".repeat(64))?;
        for i in 0..self.params.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|v| v.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i + 1));
            writeln!(
                f,
                " {:<18} {:>12.4}  {:>12.4}  {:>8.3}  {:>8.4}",
                name, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{thick}")
    }
}

pub struct PanelGLS;

impl PanelGLS {
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
        time_ids: &[i64],
        panels: GlsPanels,
        variable_names: Option<Vec<String>>,
    ) -> Result<PanelGlsResult, GreenersError> {
        if y.iter().any(|v| !v.is_finite()) || x.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "PanelGLS: dados contêm NaN ou Inf".into(),
            ));
        }

        let (_, y_panels, x_panels, big_t) = extract_balanced_panels(y, x, entity_ids, time_ids)?;

        let n_entities = y_panels.len();
        let n_obs = n_entities * big_t;
        let k = x.ncols();
        let df_resid = n_obs.saturating_sub(k);

        // ── Passo 1: OLS para resíduos iniciais ──
        let xtx0: Array2<f64> = x_panels
            .iter()
            .fold(Array2::zeros((k, k)), |acc, xi| acc + xi.t().dot(xi));
        let xty0: Array1<f64> = x_panels
            .iter()
            .zip(y_panels.iter())
            .fold(Array1::zeros(k), |acc, (xi, yi)| acc + xi.t().dot(yi));
        let beta0 = xtx0.inv()?.dot(&xty0);
        let resid0: Vec<Array1<f64>> = y_panels
            .iter()
            .zip(x_panels.iter())
            .map(|(yi, xi)| yi - &xi.dot(&beta0))
            .collect();

        // ── Passo 2: estimar Σ̂ ──
        let n = n_entities;
        let (xtox, xtoy) = match panels {
            GlsPanels::Hetero => {
                // Diagonal: σ̂²_i = e_i'e_i / T
                let mut xtox = Array2::<f64>::zeros((k, k));
                let mut xtoy = Array1::<f64>::zeros(k);
                for i in 0..n {
                    let sigma2_i = resid0[i].dot(&resid0[i]) / big_t as f64;
                    if sigma2_i < 1e-15 {
                        return Err(GreenersError::InvalidOperation(format!(
                            "PanelGLS: σ²_i ≈ 0 para entidade {i} — resíduos perfeitamente ajustados?"
                        )));
                    }
                    let w = 1.0 / sigma2_i;
                    xtox = xtox + x_panels[i].t().dot(&x_panels[i]) * w;
                    xtoy = xtoy + x_panels[i].t().dot(&y_panels[i]) * w;
                }
                (xtox, xtoy)
            }
            GlsPanels::Correlated => {
                // Σ̂ completa: σ̂_ij = e_i'e_j / T, depois inverte
                let mut sigma_hat = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    for j in i..n {
                        let s = resid0[i].dot(&resid0[j]) / big_t as f64;
                        sigma_hat[[i, j]] = s;
                        sigma_hat[[j, i]] = s;
                    }
                }
                let sigma_inv = sigma_hat.inv()?;

                let mut xtox = Array2::<f64>::zeros((k, k));
                let mut xtoy = Array1::<f64>::zeros(k);
                for i in 0..n {
                    for j in 0..n {
                        let w = sigma_inv[[i, j]];
                        xtox = xtox + x_panels[i].t().dot(&x_panels[j]) * w;
                        xtoy = xtoy + x_panels[i].t().dot(&y_panels[j]) * w;
                    }
                }
                (xtox, xtoy)
            }
        };

        // ── Passo 3: β̂_GLS = (X'Ω⁻¹X)⁻¹ X'Ω⁻¹y ──
        let xtox_inv = xtox.inv()?;
        let beta = xtox_inv.dot(&xtoy);

        // ── Resíduos GLS e σ ──
        let resid_gls: Vec<Array1<f64>> = y_panels
            .iter()
            .zip(x_panels.iter())
            .map(|(yi, xi)| yi - &xi.dot(&beta))
            .collect();
        let ssr_gls: f64 = resid_gls.iter().map(|e| e.dot(e)).sum();
        let sigma = (ssr_gls / df_resid as f64).sqrt();

        // ── V_GLS = (X'Ω⁻¹X)⁻¹  (SE assintótica, usa Normal) ──
        let std_errors: Array1<f64> = (0..k)
            .map(|i| xtox_inv[[i, i]].max(0.0).sqrt())
            .collect::<Vec<_>>()
            .into();
        let t_values = &beta / &std_errors;
        // Parks usa distribuição Normal (z), não t
        use statrs::distribution::{ContinuousCDF, Normal};
        let norm =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values: Array1<f64> = t_values.mapv(|z| 2.0 * (1.0 - norm.cdf(z.abs())));

        // ── R² ──
        let ymean = y.mean().unwrap_or(0.0);
        let ss_tot: f64 = y.iter().map(|&v| (v - ymean).powi(2)).sum();
        let r_squared = if ss_tot > 1e-15 {
            1.0 - ssr_gls / ss_tot
        } else {
            0.0
        };

        Ok(PanelGlsResult {
            params: beta,
            std_errors,
            t_values,
            p_values,
            r_squared,
            n_obs,
            n_entities,
            t_periods: big_t,
            df_resid,
            sigma,
            panels,
            variable_names,
        })
    }
}
