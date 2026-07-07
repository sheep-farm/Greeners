use ndarray::{Array1, Array2};

type ModelStats = (f64, f64, f64, f64, f64, f64, f64, usize);
type MundlakResult = (f64, f64, usize, Vec<f64>, Vec<f64>);

/// Model selection and comparison utilities
pub struct ModelSelection;

impl ModelSelection {
    /// Compare multiple models by information criteria
    ///
    /// # Arguments
    /// * `models` - Vector of (model_name, log_likelihood, n_params, n_obs) tuples
    ///
    /// # Returns
    /// Vector of (model_name, AIC, BIC, rank_AIC, rank_BIC) sorted by AIC
    ///
    /// # Example
    /// ```no_run
    /// use greeners::ModelSelection;
    ///
    /// let models = vec![
    ///     ("Model 1", -100.0, 3, 100),
    ///     ("Model 2", -95.0, 5, 100),
    ///     ("Model 3", -98.0, 4, 100),
    /// ];
    ///
    /// let comparison = ModelSelection::compare_models(models);
    /// // Returns models sorted by AIC with rankings
    /// ```
    pub fn compare_models(
        models: Vec<(&str, f64, usize, usize)>,
    ) -> Vec<(String, f64, f64, usize, usize)> {
        let mut results: Vec<(String, f64, f64)> = models
            .iter()
            .map(|(name, loglik, k, n)| {
                let aic = -2.0 * loglik + 2.0 * (*k as f64);
                let bic = -2.0 * loglik + (*k as f64) * (*n as f64).ln();
                (name.to_string(), aic, bic)
            })
            .collect();

        // Sort by AIC
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Create sorted BIC for ranking
        let mut bic_sorted = results.clone();
        bic_sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Assign rankings
        results
            .iter()
            .map(|(name, aic, bic)| {
                let rank_aic = results.iter().position(|x| &x.0 == name).unwrap() + 1;
                let rank_bic = bic_sorted.iter().position(|x| &x.0 == name).unwrap() + 1;
                (name.clone(), *aic, *bic, rank_aic, rank_bic)
            })
            .collect()
    }

    /// Calculate delta AIC and Akaike weights for model averaging
    ///
    /// # Arguments
    /// * `aic_values` - Vector of AIC values from different models
    ///
    /// # Returns
    /// Tuple of (delta_aic, akaike_weights)
    ///
    /// # Interpretation
    /// - Δ_AIC < 2: Substantial support
    /// - 4 < Δ_AIC < 7: Considerably less support
    /// - Δ_AIC > 10: Essentially no support
    pub fn akaike_weights(aic_values: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let min_aic = aic_values.iter().cloned().fold(f64::INFINITY, f64::min);

        let delta_aic: Vec<f64> = aic_values.iter().map(|aic| aic - min_aic).collect();

        // Calculate relative likelihoods: exp(-Δ_AIC/2)
        let rel_likelihood: Vec<f64> = delta_aic.iter().map(|d| (-d / 2.0).exp()).collect();

        // Sum of relative likelihoods
        let sum_rel: f64 = rel_likelihood.iter().sum();

        // Akaike weights: normalized relative likelihoods
        let weights: Vec<f64> = rel_likelihood.iter().map(|r| r / sum_rel).collect();

        (delta_aic, weights)
    }

    /// Pretty print model comparison table
    ///
    /// # Arguments
    /// * `comparison` - Output from compare_models()
    pub fn print_comparison(comparison: &[(String, f64, f64, usize, usize)]) {
        println!("\n{:=^80}", " Model Comparison ");
        println!("{:-^80}", "");
        println!(
            "{:<20} | {:>12} | {:>12} | {:>8} | {:>8}",
            "Model", "AIC", "BIC", "Rank(AIC)", "Rank(BIC)"
        );
        println!("{:-^80}", "");

        for (name, aic, bic, rank_aic, rank_bic) in comparison {
            println!(
                "{:<20} | {:>12.2} | {:>12.2} | {:>8} | {:>8}",
                name, aic, bic, rank_aic, rank_bic
            );
        }
        println!("{:=^80}", "");
    }
}

/// Panel data diagnostic tests
pub struct PanelDiagnostics;

impl PanelDiagnostics {
    /// Breusch-Pagan LM test for random effects
    ///
    /// Tests H₀: σ²_u = 0 (no panel effect, pooled OLS adequate)
    /// against H₁: σ²_u > 0 (random effects model needed)
    ///
    /// # Arguments
    /// * `residuals_pooled` - Residuals from pooled OLS
    /// * `entity_ids` - Entity identifiers for each observation
    ///
    /// # Returns
    /// Tuple of (LM_statistic, p_value)
    ///
    /// # Interpretation
    /// - If p < 0.05: Reject H₀, use RE or FE instead of pooled OLS
    /// - If p > 0.05: Pooled OLS is adequate
    pub fn breusch_pagan_lm(
        residuals_pooled: &Array1<f64>,
        entity_ids: &[usize],
    ) -> Result<(f64, f64), String> {
        use indexmap::IndexMap;
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let n = residuals_pooled.len();

        if entity_ids.len() != n {
            return Err("Entity IDs length must match residuals length".to_string());
        }

        // Group residuals by entity
        let mut entity_residuals: IndexMap<usize, Vec<f64>> = IndexMap::new();
        for (i, &entity_id) in entity_ids.iter().enumerate() {
            entity_residuals
                .entry(entity_id)
                .or_default()
                .push(residuals_pooled[i]);
        }

        let n_entities = entity_residuals.len();
        let t_bar = n as f64 / n_entities as f64; // Average T per entity

        // Calculate entity-specific mean residuals
        let mut sum_squared_means = 0.0;
        let mut sum_squared_residuals = 0.0;

        for residuals in entity_residuals.values() {
            let mean: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
            let t = residuals.len() as f64;
            sum_squared_means += t * mean.powi(2);

            for &r in residuals {
                sum_squared_residuals += r.powi(2);
            }
        }

        // LM statistic
        let lm_stat = (n as f64 / 2.0)
            * ((sum_squared_means / sum_squared_residuals) - 1.0).powi(2)
            / (t_bar - 1.0);

        // Under H₀, LM ~ χ²(1)
        let chi2_dist = ChiSquared::new(1.0).map_err(|e| e.to_string())?;
        let p_value = 1.0 - chi2_dist.cdf(lm_stat);

        Ok((lm_stat, p_value))
    }

    /// F-test for fixed effects (vs pooled OLS)
    ///
    /// Tests H₀: All entity effects are zero (pooled OLS adequate)
    /// against H₁: Entity effects exist (use fixed effects)
    ///
    /// # Arguments
    /// * `ssr_pooled` - Sum of squared residuals from pooled OLS
    /// * `ssr_fe` - Sum of squared residuals from fixed effects model
    /// * `n` - Total number of observations
    /// * `n_entities` - Number of entities
    /// * `k` - Number of slope parameters (excluding entity dummies)
    ///
    /// # Returns
    /// Tuple of (F_statistic, p_value)
    ///
    /// # Interpretation
    /// - If p < 0.05: Reject H₀, use FE instead of pooled OLS
    /// - If p > 0.05: Pooled OLS is adequate
    pub fn f_test_fixed_effects(
        ssr_pooled: f64,
        ssr_fe: f64,
        n: usize,
        n_entities: usize,
        k: usize,
    ) -> Result<(f64, f64), String> {
        use statrs::distribution::{ContinuousCDF, FisherSnedecor};

        // Check for sufficient degrees of freedom before calculating
        if n <= n_entities + k {
            return Err("Insufficient degrees of freedom".to_string());
        }

        // Degrees of freedom
        let df_num = n_entities - 1; // Entity dummies
        let df_denom = n - n_entities - k;

        // F-statistic
        let f_stat = ((ssr_pooled - ssr_fe) / df_num as f64) / (ssr_fe / df_denom as f64);

        // p-value
        let f_dist =
            FisherSnedecor::new(df_num as f64, df_denom as f64).map_err(|e| e.to_string())?;
        let p_value = 1.0 - f_dist.cdf(f_stat);

        Ok((f_stat, p_value))
    }

    /// Arellano-Bond (1991) test for serial correlation in first-differenced residuals.
    ///
    /// Computa as estatísticas m1 e m2 para testar autocorrelação serial de ordem 1 e 2
    /// nos resíduos da equação em primeira diferença.
    ///
    /// Interpretação:
    ///   m1 DEVE rejeitar H₀ (FD induz AR(1) por construção — confirma o modelo)
    ///   m2 NÃO deve rejeitar H₀ (valida instrumentos y_{i,t-2} do GMM)
    ///
    /// Estatística: m_p = C_p / √V̂_p  ~ N(0,1)
    ///   C_p  = Σ_{i,t} Δê_it · Δê_{i,t-p}
    ///   V̂_p = Σ_i (Σ_t Δê_it · Δê_{i,t-p})²   (sandwich sob H₀)
    ///
    /// Returns (m1_stat, m1_pval, m2_stat, m2_pval)
    pub fn arellano_bond_test(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
        time_vals: &[f64],
    ) -> Result<(f64, f64, f64, f64), String> {
        use crate::{CovarianceType, OLS};
        use indexmap::IndexMap;
        use statrs::distribution::{ContinuousCDF, Normal};

        let n = y.len();
        let k = x.ncols();

        if entity_ids.len() != n || time_vals.len() != n {
            return Err("entity_ids e time_vals devem ter o mesmo comprimento que y".to_string());
        }

        // 1. Group indices by entity, sorted by time
        let mut entity_idx: IndexMap<i64, Vec<usize>> = IndexMap::new();
        for (i, &eid) in entity_ids.iter().enumerate() {
            entity_idx.entry(eid).or_default().push(i);
        }
        for indices in entity_idx.values_mut() {
            indices.sort_by(|&a, &b| {
                time_vals[a]
                    .partial_cmp(&time_vals[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut sorted_entities: Vec<i64> = entity_idx.keys().copied().collect();
        sorted_entities.sort();

        // 2. First-difference y and X within each entity
        let mut dy_vec: Vec<f64> = Vec::new();
        let mut dx_rows: Vec<Vec<f64>> = Vec::new();

        for eid in &sorted_entities {
            let indices = &entity_idx[eid];
            let t = indices.len();
            if t < 2 {
                continue;
            }
            for s in 1..t {
                let curr = indices[s];
                let prev = indices[s - 1];
                dy_vec.push(y[curr] - y[prev]);
                dx_rows.push((0..k).map(|c| x[[curr, c]] - x[[prev, c]]).collect());
            }
        }

        let n_fd = dy_vec.len();
        if n_fd == 0 {
            return Err("Nenhuma observação após primeira diferença".to_string());
        }

        let dy = Array1::from_vec(dy_vec);
        let mut dx = Array2::<f64>::zeros((n_fd, k));
        for (i, row) in dx_rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                dx[[i, j]] = v;
            }
        }

        // Drop zero columns (constants become zero after FD)
        let active_cols: Vec<usize> = (0..k)
            .filter(|&c| dx.column(c).iter().any(|&v| v.abs() > 1e-12))
            .collect();

        let dx_active = {
            let mut m = Array2::<f64>::zeros((n_fd, active_cols.len()));
            for (nc, &oc) in active_cols.iter().enumerate() {
                m.column_mut(nc).assign(&dx.column(oc));
            }
            m
        };

        // 3. OLS on FD model → residuals Δê
        let fd_ols = OLS::fit(&dy, &dx_active, CovarianceType::NonRobust)
            .map_err(|e| format!("OLS na primeira diferença: {e}"))?;
        let fd_resid = fd_ols.residuals(&dy, &dx_active);

        // 4. Map FD residuals back to entity groups
        //    fd_resid[row_ptr..row_ptr+fd_count] = entity eid's FD residuals in time order
        let mut entity_fd_resid: IndexMap<i64, Vec<f64>> = IndexMap::new();
        let mut row_ptr = 0usize;
        for eid in &sorted_entities {
            let t = entity_idx[eid].len();
            if t < 2 {
                continue;
            }
            let fd_count = t - 1;
            let resids: Vec<f64> = (row_ptr..row_ptr + fd_count).map(|i| fd_resid[i]).collect();
            entity_fd_resid.insert(*eid, resids);
            row_ptr += fd_count;
        }

        // 5. Compute m_p for p = 1 and p = 2
        let m_stat = |p: usize| -> Option<(f64, f64)> {
            let mut c_p = 0.0f64;
            let mut v_p = 0.0f64;

            for resids in entity_fd_resid.values() {
                let m = resids.len();
                if m <= p {
                    continue;
                }
                // Entity-level cross-product sum: Σ_t Δê_t * Δê_{t-p}
                let entity_sum: f64 = (p..m).map(|t| resids[t] * resids[t - p]).sum();
                c_p += entity_sum;
                v_p += entity_sum * entity_sum; // squared for sandwich variance
            }

            if v_p < 1e-20 {
                return None; // Not enough data or degenerate
            }

            let stat = c_p / v_p.sqrt();
            let normal = Normal::new(0.0, 1.0).ok()?;
            let pval = 2.0 * (1.0 - normal.cdf(stat.abs()));
            Some((stat, pval))
        };

        let (m1, p1) = m_stat(1)
            .ok_or("Dados insuficientes para m1 (precisa T ≥ 3 por entidade)".to_string())?;
        let (m2, p2) = m_stat(2)
            .ok_or("Dados insuficientes para m2 (precisa T ≥ 4 por entidade)".to_string())?;

        Ok((m1, p1, m2, p2))
    }

    /// Chamberlain (1982) test for correlation between regressors and individual effects.
    ///
    /// Generalização do Mundlak: em vez de usar apenas a média individual X̄_i, inclui
    /// os valores de X em TODOS os períodos — testando a forma mais geral de correlação
    /// entre efeitos individuais e regressores.
    ///
    /// H₀: Π_1 = Π_2 = ... = Π_T = 0 (RE consistente)
    /// H₁: pelo menos um Π_s ≠ 0 (efeitos correlacionados com X — use FE)
    ///
    /// Requer painel balanceado (mesmos períodos para todas as entidades).
    ///
    /// Procedure:
    ///   1. Para cada regressor não-constante j e período s, cria coluna contendo
    ///      o valor x_{i,j,s} para cada observação da entidade i (constante dentro da entidade)
    ///   2. Augmenta o modelo com essas k×T colunas (descartando zero-variância)
    ///   3. F-test H₀: todos os coeficientes das colunas augmentadas = 0
    ///
    /// Returns (f_stat, p_value, k_active, df_denom, n_entities, t_count)
    pub fn chamberlain(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
        time_vals: &[f64],
    ) -> Result<(f64, f64, usize, usize, usize, usize), String> {
        use crate::{CovarianceType, OLS};
        use indexmap::IndexMap;
        use statrs::distribution::{ContinuousCDF, FisherSnedecor};

        let n = y.len();
        let k_full = x.ncols();

        if entity_ids.len() != n || time_vals.len() != n {
            return Err("entity_ids e time_vals devem ter o mesmo comprimento que y".to_string());
        }

        // Non-constant regressors
        let non_const_cols: Vec<usize> = (0..k_full)
            .filter(|&c| {
                let mean = x.column(c).sum() / n as f64;
                x.column(c).iter().any(|&v| (v - mean).abs() > 1e-10)
            })
            .collect();
        let k = non_const_cols.len();
        if k == 0 {
            return Err("Nenhum regressor variante no tempo encontrado".to_string());
        }

        // Unique sorted time periods (via bit-based dedup to handle floats correctly)
        let mut seen_bits: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut unique_times: Vec<f64> = Vec::new();
        for &t in time_vals {
            if seen_bits.insert(t.to_bits()) {
                unique_times.push(t);
            }
        }
        unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let t_count = unique_times.len();

        if t_count < 2 {
            return Err("O teste de Chamberlain requer pelo menos 2 períodos".to_string());
        }

        // Time value → sorted index
        let time_to_idx: IndexMap<u64, usize> = unique_times
            .iter()
            .enumerate()
            .map(|(i, &t)| (t.to_bits(), i))
            .collect();

        // For each entity, collect non-const regressor values at each period
        let mut entity_period: IndexMap<i64, IndexMap<usize, Vec<f64>>> = IndexMap::new();
        for (obs, &eid) in entity_ids.iter().enumerate() {
            let t_idx = *time_to_idx
                .get(&time_vals[obs].to_bits())
                .ok_or("Período não encontrado no índice")?;
            let vals: Vec<f64> = non_const_cols.iter().map(|&c| x[[obs, c]]).collect();
            entity_period.entry(eid).or_default().insert(t_idx, vals);
        }

        let n_entities = entity_period.len();

        // Require balanced panel
        for (&eid, periods) in &entity_period {
            if periods.len() != t_count {
                return Err(format!(
                    "Painel desbalanceado: entidade {} tem {} períodos (esperado {}). \
                     Filtre o dataset para incluir apenas entidades com todos os {} períodos.",
                    eid,
                    periods.len(),
                    t_count,
                    t_count
                ));
            }
        }

        // Build augmented design matrix: [X | Chamberlain cols]
        // For each (j=regressor, s=period): column value for obs (i,t) = x_{i,j,s}
        let k_chamber = k * t_count;
        let k_aug_total = k_full + k_chamber;
        let mut x_aug = Array2::<f64>::zeros((n, k_aug_total));

        for i in 0..n {
            for c in 0..k_full {
                x_aug[[i, c]] = x[[i, c]];
            }
            let eid = entity_ids[i];
            let ep = entity_period
                .get(&eid)
                .ok_or("ID de entidade não encontrado")?;
            for (j, _) in non_const_cols.iter().enumerate() {
                for s in 0..t_count {
                    if let Some(vals) = ep.get(&s) {
                        x_aug[[i, k_full + j * t_count + s]] = vals[j];
                    }
                }
            }
        }

        // Drop Chamberlain columns with zero variance (e.g., time-invariant regressors
        // or periods with identical values across all entities)
        let active_aug: Vec<usize> = (k_full..k_aug_total)
            .filter(|&c| {
                let mean = x_aug.column(c).sum() / n as f64;
                x_aug.column(c).iter().any(|&v| (v - mean).abs() > 1e-10)
            })
            .collect();

        let k_active = active_aug.len();
        if k_active == 0 {
            return Err(
                "Nenhuma coluna de augmentação com variância — regressores constantes?".to_string(),
            );
        }

        // Build final augmented matrix: [original X | active Chamberlain cols]
        let k_final = k_full + k_active;
        if n <= k_final {
            return Err(format!(
                "Graus de liberdade insuficientes: n={} ≤ colunas do modelo augmentado={}. \
                 T muito grande relativo ao número de observações.",
                n, k_final
            ));
        }

        let mut x_final = Array2::<f64>::zeros((n, k_final));
        for i in 0..n {
            for c in 0..k_full {
                x_final[[i, c]] = x[[i, c]];
            }
            for (new_c, &old_c) in active_aug.iter().enumerate() {
                x_final[[i, k_full + new_c]] = x_aug[[i, old_c]];
            }
        }

        // Restricted OLS: y ~ X
        let ols_r =
            OLS::fit(y, x, CovarianceType::NonRobust).map_err(|e| format!("OLS restrito: {e}"))?;

        // Unrestricted OLS: y ~ X + Chamberlain cols
        let ols_u = OLS::fit(y, &x_final, CovarianceType::NonRobust)
            .map_err(|e| format!("OLS não-restrito: {e}"))?;

        let ssr_r = ols_r.sigma.powi(2) * ols_r.df_resid as f64;
        let ssr_u = ols_u.sigma.powi(2) * ols_u.df_resid as f64;
        let df_u = ols_u.df_resid;

        if df_u == 0 || ssr_u < 1e-15 {
            return Err("Graus de liberdade insuficientes no modelo não-restrito".to_string());
        }

        let f_stat = ((ssr_r - ssr_u) / k_active as f64) / (ssr_u / df_u as f64);

        let f_dist = FisherSnedecor::new(k_active as f64, df_u as f64)
            .map_err(|e| format!("F-distribuição: {e}"))?;
        let p_value = 1.0 - f_dist.cdf(f_stat.max(0.0));

        Ok((f_stat, p_value, k_active, df_u, n_entities, t_count))
    }

    /// Mundlak (1978) test for correlation between regressors and individual effects.
    ///
    /// H₀: γ = 0 — médias individuais não correlacionadas com os regressores (RE consistente)
    /// H₁: γ ≠ 0 — efeitos individuais correlacionados com X (use FE)
    ///
    /// Procedure:
    ///   1. Compute entity means X̄_i for each non-constant column of X
    ///   2. Run OLS restricted: y ~ X
    ///   3. Run OLS unrestricted: y ~ X + X̄
    ///   4. F-test H₀: all γ = 0   F(k, n - 2k - 1) where k = non-constant regressors
    ///
    /// Returns (f_stat, p_value, k, gamma_hat, gamma_se)
    pub fn mundlak(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
    ) -> Result<MundlakResult, String> {
        use crate::{CovarianceType, OLS};
        use indexmap::IndexMap;
        use statrs::distribution::{ContinuousCDF, FisherSnedecor};

        let n = y.len();
        let k_full = x.ncols();

        if entity_ids.len() != n {
            return Err("entity_ids deve ter o mesmo comprimento que y".to_string());
        }

        // Identify non-constant columns (positive variance)
        let non_const_cols: Vec<usize> = (0..k_full)
            .filter(|&c| {
                let col: Vec<f64> = x.column(c).to_vec();
                let mean = col.iter().sum::<f64>() / col.len() as f64;
                col.iter().any(|&v| (v - mean).abs() > 1e-10)
            })
            .collect();

        let k = non_const_cols.len();
        if k == 0 {
            return Err("Nenhum regressor variante no tempo encontrado".to_string());
        }

        // Compute entity means for non-constant columns
        let mut entity_sums: IndexMap<i64, (Vec<f64>, usize)> = IndexMap::new();
        for (i, &eid) in entity_ids.iter().enumerate() {
            let entry = entity_sums.entry(eid).or_insert_with(|| (vec![0.0; k], 0));
            for (j, &c) in non_const_cols.iter().enumerate() {
                entry.0[j] += x[[i, c]];
            }
            entry.1 += 1;
        }
        let entity_means: IndexMap<i64, Vec<f64>> = entity_sums
            .into_iter()
            .map(|(eid, (sums, cnt))| (eid, sums.into_iter().map(|s| s / cnt as f64).collect()))
            .collect();

        // Build augmented design matrix [X | X̄]
        let mut x_aug = Array2::<f64>::zeros((n, k_full + k));
        for i in 0..n {
            for c in 0..k_full {
                x_aug[[i, c]] = x[[i, c]];
            }
            let means = &entity_means[&entity_ids[i]];
            for (j, &mean) in means.iter().enumerate() {
                x_aug[[i, k_full + j]] = mean;
            }
        }

        // Restricted OLS: y ~ X
        let ols_r = OLS::fit(y, x, CovarianceType::NonRobust)
            .map_err(|e| format!("OLS restrito falhou: {e}"))?;

        // Unrestricted OLS: y ~ X + X̄
        let ols_u = OLS::fit(y, &x_aug, CovarianceType::NonRobust)
            .map_err(|e| format!("OLS não-restrito falhou: {e}"))?;

        let ssr_r = ols_r.sigma.powi(2) * ols_r.df_resid as f64;
        let ssr_u = ols_u.sigma.powi(2) * ols_u.df_resid as f64;
        let df_u = ols_u.df_resid;

        if df_u == 0 || ssr_u < 1e-15 {
            return Err("Graus de liberdade insuficientes no modelo não-restrito".to_string());
        }

        let f_stat = ((ssr_r - ssr_u) / k as f64) / (ssr_u / df_u as f64);

        let f_dist = FisherSnedecor::new(k as f64, df_u as f64)
            .map_err(|e| format!("F-distribuição: {e}"))?;
        let p_value = 1.0 - f_dist.cdf(f_stat.max(0.0));

        // Extract γ̂ and SE from the unrestricted model (last k parameters)
        let n_params = ols_u.params.len();
        let gamma_hat: Vec<f64> = (n_params - k..n_params).map(|i| ols_u.params[i]).collect();
        let gamma_se: Vec<f64> = (n_params - k..n_params)
            .map(|i| ols_u.std_errors[i])
            .collect();

        Ok((f_stat, p_value, k, gamma_hat, gamma_se))
    }

    /// Wooldridge (2002) test for serial correlation in panel data.
    ///
    /// H₀: no first-order serial correlation in idiosyncratic errors (ρ = -0.5)
    /// H₁: serial correlation exists
    ///
    /// Procedure:
    ///   1. Sort by time within entity, first-difference y and X
    ///   2. Run OLS on first-differenced model to get residuals ê
    ///   3. Regress ê_it on ê_{i,t-1} (no intercept, pooled across entities)
    ///   4. Test H₀: ρ̂ = -0.5 using t(N-1) distribution
    ///
    /// Requires T ≥ 3 for at least some entities (to build residual lag pairs).
    ///
    /// Returns (rho_hat, t_stat, p_value, n_pairs)
    pub fn wooldridge_serial(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
        time_vals: &[f64],
    ) -> Result<(f64, f64, f64, usize), String> {
        use crate::{CovarianceType, OLS};
        use indexmap::IndexMap;
        use statrs::distribution::{ContinuousCDF, StudentsT};

        let n = y.len();
        if entity_ids.len() != n || time_vals.len() != n {
            return Err("entity_ids e time_vals devem ter o mesmo comprimento que y".to_string());
        }
        let k = x.ncols();
        if n < 4 {
            return Err("Observações insuficientes para o teste de Wooldridge".to_string());
        }

        // 1. Group indices by entity, sorted by time
        let mut entity_idx: IndexMap<i64, Vec<usize>> = IndexMap::new();
        for (i, &eid) in entity_ids.iter().enumerate() {
            entity_idx.entry(eid).or_default().push(i);
        }
        for indices in entity_idx.values_mut() {
            indices.sort_by(|&a, &b| {
                time_vals[a]
                    .partial_cmp(&time_vals[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let n_entities = entity_idx.len();
        let mut sorted_entities: Vec<i64> = entity_idx.keys().copied().collect();
        sorted_entities.sort();

        // 2. First-difference y and X within each entity
        let mut dy_vec: Vec<f64> = Vec::new();
        let mut dx_rows: Vec<Vec<f64>> = Vec::new();

        for eid in &sorted_entities {
            let indices = &entity_idx[eid];
            let t = indices.len();
            if t < 2 {
                continue;
            }
            for s in 1..t {
                let i_curr = indices[s];
                let i_prev = indices[s - 1];
                dy_vec.push(y[i_curr] - y[i_prev]);
                let row: Vec<f64> = (0..k).map(|c| x[[i_curr, c]] - x[[i_prev, c]]).collect();
                dx_rows.push(row);
            }
        }

        let n_fd = dy_vec.len();
        if n_fd == 0 {
            return Err("Nenhuma observação após primeira diferença".to_string());
        }

        let dy = Array1::from_vec(dy_vec);
        let mut dx = Array2::<f64>::zeros((n_fd, k));
        for (i, row) in dx_rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                dx[[i, j]] = v;
            }
        }

        // Drop zero columns (constants become zero after differencing)
        let active_cols: Vec<usize> = (0..k)
            .filter(|&c| dx.column(c).iter().any(|&v| v.abs() > 1e-12))
            .collect();

        if active_cols.is_empty() {
            return Err("Todos os regressores tornaram-se zero após diferenciação".to_string());
        }

        let dx_active = {
            let mut m = Array2::<f64>::zeros((n_fd, active_cols.len()));
            for (new_c, &old_c) in active_cols.iter().enumerate() {
                m.column_mut(new_c).assign(&dx.column(old_c));
            }
            m
        };

        // 3. OLS on first-differenced model
        let fd_ols = OLS::fit(&dy, &dx_active, CovarianceType::NonRobust)
            .map_err(|e| format!("OLS na primeira diferença falhou: {e}"))?;
        let fd_resid = fd_ols.residuals(&dy, &dx_active);

        // 4. Build residual lag pairs (ê_it, ê_{i,t-1}) within each entity
        //    fd_resid rows correspond to entities in sorted order, consecutive FD periods
        let mut aux_curr: Vec<f64> = Vec::new();
        let mut aux_lag: Vec<f64> = Vec::new();
        let mut row_ptr = 0usize;

        for eid in &sorted_entities {
            let t = entity_idx[eid].len();
            if t < 2 {
                continue;
            }
            let fd_count = t - 1;
            // Need at least 2 FD residuals per entity (T >= 3)
            if fd_count >= 2 {
                for s in 1..fd_count {
                    aux_curr.push(fd_resid[row_ptr + s]);
                    aux_lag.push(fd_resid[row_ptr + s - 1]);
                }
            }
            row_ptr += fd_count;
        }

        let n_pairs = aux_curr.len();
        if n_pairs < 2 {
            return Err(
                "Poucas observações para o teste (necessário T ≥ 3 em pelo menos uma entidade)"
                    .to_string(),
            );
        }

        // 5. Estimate ρ: ê_it = ρ * ê_{i,t-1} + v  (no intercept)
        let sum_xx: f64 = aux_lag.iter().map(|&v| v * v).sum();
        let sum_xy: f64 = aux_lag
            .iter()
            .zip(aux_curr.iter())
            .map(|(&xl, &xc)| xl * xc)
            .sum();

        if sum_xx < 1e-15 {
            return Err("Matriz singular na regressão auxiliar".to_string());
        }

        let rho_hat = sum_xy / sum_xx;

        let ssr_aux: f64 = aux_lag
            .iter()
            .zip(aux_curr.iter())
            .map(|(&xl, &xc)| (xc - rho_hat * xl).powi(2))
            .sum();

        let df_aux = (n_pairs - 1) as f64;
        let se_rho = (ssr_aux / df_aux / sum_xx).sqrt();

        if se_rho < 1e-15 {
            return Err("Erro padrão de ρ̂ próximo de zero".to_string());
        }

        // 6. t-statistic: H₀: ρ = -0.5, df = N - 1 (Wooldridge, 2002, p.283)
        let t_stat = (rho_hat - (-0.5)) / se_rho;
        let df_t = (n_entities - 1) as f64;

        let t_dist = StudentsT::new(0.0, 1.0, df_t).map_err(|e| format!("t-distribuição: {e}"))?;
        let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));

        Ok((rho_hat, t_stat, p_value, n_pairs))
    }

    /// Pesaran (2004) CD test for cross-sectional dependence in panel data.
    ///
    /// H₀: residuals are cross-sectionally independent (ρ_ij = 0 ∀ i≠j)
    /// H₁: cross-sectional dependence exists
    ///
    /// CD = sqrt(2/(N(N-1))) * Σ_{i<j} sqrt(T_ij) * ρ̂_ij  ~ N(0,1) under H₀
    ///
    /// Arguments: residuals from a panel model (pooled OLS or FE) and entity IDs
    /// in the same order as the observations.
    pub fn pesaran_cd(residuals: &Array1<f64>, entity_ids: &[usize]) -> Result<(f64, f64), String> {
        use indexmap::IndexMap;
        use statrs::distribution::{ContinuousCDF, Normal};

        let n_obs = residuals.len();
        if entity_ids.len() != n_obs {
            return Err("entity_ids length must match residuals length".to_string());
        }

        // Group residuals by entity, preserving observation order (= time order)
        let mut groups: IndexMap<usize, Vec<f64>> = IndexMap::new();
        for (&id, &r) in entity_ids.iter().zip(residuals.iter()) {
            groups.entry(id).or_default().push(r);
        }

        let mut entity_list: Vec<usize> = groups.keys().copied().collect();
        entity_list.sort();
        let n_entities = entity_list.len();

        if n_entities < 2 {
            return Err("Need at least 2 entities for the Pesaran CD test".to_string());
        }

        let residuals_by_entity: Vec<&Vec<f64>> =
            entity_list.iter().map(|id| &groups[id]).collect();

        let mut cd_sum = 0.0;
        let mut n_pairs = 0usize;

        for (i, &ei) in residuals_by_entity.iter().enumerate() {
            for &ej in residuals_by_entity.iter().skip(i + 1) {
                let t_ij = ei.len().min(ej.len());
                if t_ij < 2 {
                    continue;
                }

                let mean_i = ei[..t_ij].iter().sum::<f64>() / t_ij as f64;
                let mean_j = ej[..t_ij].iter().sum::<f64>() / t_ij as f64;

                let cov: f64 = ei[..t_ij]
                    .iter()
                    .zip(ej[..t_ij].iter())
                    .map(|(&a, &b)| (a - mean_i) * (b - mean_j))
                    .sum();
                let var_i: f64 = ei[..t_ij].iter().map(|&a| (a - mean_i).powi(2)).sum();
                let var_j: f64 = ej[..t_ij].iter().map(|&b| (b - mean_j).powi(2)).sum();

                let denom = (var_i * var_j).sqrt();
                if denom < 1e-15 {
                    continue;
                }

                cd_sum += (t_ij as f64).sqrt() * (cov / denom);
                n_pairs += 1;
            }
        }

        if n_pairs == 0 {
            return Err("No valid entity pairs found for CD test".to_string());
        }

        let cd = (2.0 / (n_entities * (n_entities - 1)) as f64).sqrt() * cd_sum;

        let normal = Normal::new(0.0, 1.0).map_err(|e| e.to_string())?;
        let p_value = 2.0 * (1.0 - normal.cdf(cd.abs()));

        Ok((cd, p_value))
    }
}

/// Summary statistics helper
pub struct SummaryStats;

impl SummaryStats {
    /// Calculate comprehensive descriptive statistics
    ///
    /// # Arguments
    /// * `data` - Data vector
    ///
    /// # Returns
    /// Tuple of (mean, std, min, q25, median, q75, max, n_obs)
    pub fn describe(data: &Array1<f64>) -> (f64, f64, f64, f64, f64, f64, f64, usize) {
        let n = data.len();
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = sorted[n - 1];

        let q25 = Self::percentile(&sorted, 25.0);
        let median = Self::percentile(&sorted, 50.0);
        let q75 = Self::percentile(&sorted, 75.0);

        (mean, std, min, q25, median, q75, max, n)
    }

    /// Calculate percentile from sorted data
    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        let n = sorted_data.len();
        let idx = (p / 100.0) * (n - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;
        let weight = idx - lower as f64;

        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }

    /// Pretty print summary statistics table
    ///
    /// # Arguments
    /// * `stats` - Vector of (variable_name, stats_tuple) pairs
    // pub fn print_summary(stats: &[(&str, (f64, f64, f64, f64, f64, f64, f64, usize))]) {
    pub fn print_summary(stats: &[(&str, ModelStats)]) {
        println!("\n{:=^90}", " Descriptive Statistics ");
        println!("{:-^90}", "");
        println!(
            "{:<12} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}",
            "Variable", "Mean", "Std", "Min", "Q25", "Median", "Q75", "Max"
        );
        println!("{:-^90}", "");

        for (name, (mean, std, min, q25, median, q75, max, _n)) in stats {
            println!(
                "{:<12} | {:>8.2} | {:>8.2} | {:>8.2} | {:>8.2} | {:>8.2} | {:>8.2} | {:>8.2}",
                name, mean, std, min, q25, median, q75, max
            );
        }
        println!("{:=^90}", "");
    }
}
