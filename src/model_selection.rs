use ndarray::Array1;

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
        use statrs::distribution::{ChiSquared, ContinuousCDF};
        use std::collections::HashMap;

        let n = residuals_pooled.len();

        if entity_ids.len() != n {
            return Err("Entity IDs length must match residuals length".to_string());
        }

        // Group residuals by entity
        let mut entity_residuals: HashMap<usize, Vec<f64>> = HashMap::new();
        for (i, &entity_id) in entity_ids.iter().enumerate() {
            entity_residuals
                .entry(entity_id)
                .or_insert_with(Vec::new)
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
        let lm_stat = (n as f64 / 2.0) * ((sum_squared_means / sum_squared_residuals) - 1.0).powi(2)
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
    pub fn print_summary(stats: &[(&str, (f64, f64, f64, f64, f64, f64, f64, usize))]) {
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
