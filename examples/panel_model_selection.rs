use greeners::{
    CovarianceType, DataFrame, Formula, ModelSelection, PanelDiagnostics, SummaryStats, OLS,
};
use indexmap::IndexMap;
use ndarray::Array1;
use rand::{thread_rng, Rng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("   Greeners v0.9.0 - Panel Diagnostics & Model Selection");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // CREATE PANEL DATASET
    // ═══════════════════════════════════════════════════════════════════════════

    println!("Dataset: Firm Investment Panel Data (20 firms × 10 years = 200 obs)");
    println!("Variables:");
    println!("  • investment: Capital investment");
    println!("  • profit: Firm profit");
    println!("  • cash_flow: Operating cash flow");
    println!("  • size: Firm size (log assets)\n");

    let n_firms = 20;
    let n_periods = 10;
    let n_obs = n_firms * n_periods;

    let mut investment_data = Vec::with_capacity(n_obs);
    let mut profit_data = Vec::with_capacity(n_obs);
    let mut cash_flow_data = Vec::with_capacity(n_obs);
    let mut size_data = Vec::with_capacity(n_obs);
    let mut firm_ids = Vec::with_capacity(n_obs);
    let mut time_ids = Vec::with_capacity(n_obs);

    // Generate realistic panel data with firm fixed effects and random noise
    let mut rng = thread_rng();

    for firm in 0..n_firms {
        let firm_effect = (firm as f64 - 10.0) * 0.8; // Firm-specific effect

        for period in 0..n_periods {
            let t = period as f64;

            // Generate independent random variables with firm effects and trend
            let profit = 10.0 + firm_effect * 0.7 + t * 0.4 + rng.gen_range(-2.0..2.0);
            let cash_flow = 8.0 + firm_effect * 0.5 + t * 0.3 + rng.gen_range(-1.5..1.5);
            let size = 5.0 + firm_effect * 0.4 + t * 0.15 + rng.gen_range(-0.8..0.8);

            // Investment depends on other variables + firm effect + noise
            let investment = 2.0
                + firm_effect * 1.0
                + profit * 0.35
                + cash_flow * 0.25
                + size * 0.15
                + t * 0.2
                + rng.gen_range(-2.5..2.5);

            investment_data.push(investment);
            profit_data.push(profit);
            cash_flow_data.push(cash_flow);
            size_data.push(size);
            firm_ids.push(firm);
            time_ids.push(period);
        }
    }

    let mut data = IndexMap::new();
    data.insert("investment".to_string(), Array1::from(investment_data));
    data.insert("profit".to_string(), Array1::from(profit_data));
    data.insert("cash_flow".to_string(), Array1::from(cash_flow_data));
    data.insert("size".to_string(), Array1::from(size_data));

    let df = DataFrame::new(data.clone())?;

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. DESCRIPTIVE STATISTICS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("1. DESCRIPTIVE STATISTICS");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let inv_stats = SummaryStats::describe(&data["investment"]);
    let prof_stats = SummaryStats::describe(&data["profit"]);
    let cf_stats = SummaryStats::describe(&data["cash_flow"]);
    let size_stats = SummaryStats::describe(&data["size"]);

    let summary_data = vec![
        ("investment", inv_stats),
        ("profit", prof_stats),
        ("cash_flow", cf_stats),
        ("size", size_stats),
    ];

    SummaryStats::print_summary(&summary_data);

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. ESTIMATE COMPETING MODELS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("2. ESTIMATE COMPETING MODELS");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Model 1: Pooled OLS (ignores panel structure)
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Model 1: Pooled OLS (investment ~ profit + cash_flow + size)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    let formula_full = Formula::parse("investment ~ profit + cash_flow + size")?;
    let model1_pooled = OLS::from_formula(&formula_full, &df, CovarianceType::NonRobust)?;
    println!("{}", model1_pooled);

    // Model 2: Pooled OLS without size
    println!("\n─────────────────────────────────────────────────────────────────────────────");
    println!("Model 2: Pooled OLS (investment ~ profit + cash_flow)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    let formula_restricted = Formula::parse("investment ~ profit + cash_flow")?;
    let model2_pooled = OLS::from_formula(&formula_restricted, &df, CovarianceType::NonRobust)?;
    println!("{}", model2_pooled);

    // Model 3: Only profit
    println!("\n─────────────────────────────────────────────────────────────────────────────");
    println!("Model 3: Pooled OLS (investment ~ profit)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    let formula_simple = Formula::parse("investment ~ profit")?;
    let model3_simple = OLS::from_formula(&formula_simple, &df, CovarianceType::NonRobust)?;
    println!("{}", model3_simple);

    // Note: Fixed Effects model excluded due to numerical instability issues
    // with current implementation. This will be addressed in a future update.

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. MODEL COMPARISON BY AIC/BIC
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("3. MODEL COMPARISON (Information Criteria)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let models_to_compare = vec![
        (
            "Pooled (Full)",
            model1_pooled.log_likelihood,
            4, // intercept + 3 slopes
            n_obs,
        ),
        ("Pooled (No size)", model2_pooled.log_likelihood, 3, n_obs),
        (
            "Pooled (Profit only)",
            model3_simple.log_likelihood,
            2,
            n_obs,
        ),
    ];

    let comparison = ModelSelection::compare_models(models_to_compare);
    ModelSelection::print_comparison(&comparison);

    // Calculate Akaike weights
    let aic_values: Vec<f64> = comparison.iter().map(|(_, aic, _, _, _)| *aic).collect();
    let (delta_aic, weights) = ModelSelection::akaike_weights(&aic_values);

    println!("\n📊 AKAIKE WEIGHTS (Model Averaging):");
    println!("{:-^80}", "");
    println!(
        "{:<20} | {:>12} | {:>12} | {:>20}",
        "Model", "Δ_AIC", "Weight", "Interpretation"
    );
    println!("{:-^80}", "");

    for (i, (name, _, _, _, _)) in comparison.iter().enumerate() {
        let support = if delta_aic[i] < 2.0 {
            "Substantial support"
        } else if delta_aic[i] < 4.0 {
            "Moderate support"
        } else if delta_aic[i] < 7.0 {
            "Less support"
        } else {
            "No support"
        };

        println!(
            "{:<20} | {:>12.2} | {:>12.3} | {:>20}",
            name, delta_aic[i], weights[i], support
        );
    }
    println!("{:-^80}", "");

    println!("\n✨ BEST MODEL: {}", comparison[0].0);
    println!("   (Lowest AIC, highest Akaike weight)");

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. PANEL DIAGNOSTICS - Breusch-Pagan LM Test
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("4. BREUSCH-PAGAN LM TEST for Random Effects");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("\nH₀: σ²_u = 0 (no panel effect, pooled OLS adequate)");
    println!("H₁: σ²_u > 0 (random effects needed)\n");

    let (y_pooled, x_pooled) = df.to_design_matrix(&formula_full)?;
    let residuals_pooled = model1_pooled.residuals(&y_pooled, &x_pooled);

    let (lm_stat, lm_p) = PanelDiagnostics::breusch_pagan_lm(&residuals_pooled, &firm_ids)?;

    println!("LM Statistic: {:.4}", lm_stat);
    println!("P-value: {:.6}", lm_p);

    if lm_p < 0.05 {
        println!("\n✅ REJECT H₀ at 5% level");
        println!("   → Panel effects exist");
        println!("   → Use Random Effects or Fixed Effects instead of Pooled OLS");
    } else {
        println!("\n❌ FAIL TO REJECT H₀ at 5% level");
        println!("   → No evidence of panel effects");
        println!("   → Pooled OLS is adequate");
    }

    // Note: F-test for Fixed Effects excluded due to FE model numerical issues
    // See PanelDiagnostics::f_test_fixed_effects() for the implementation

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. DECISION TREE
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("6. PANEL DATA MODEL SELECTION DECISION TREE");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("\n┌─ Start: Panel Data");
    println!("│");
    println!("├─ Step 1: Breusch-Pagan LM Test");
    println!("│  ├─ H₀ rejected? → Panel effects exist");
    println!("│  └─ H₀ not rejected? → Use Pooled OLS");
    println!("│");
    println!("├─ Step 2: F-test for Fixed Effects");
    println!("│  ├─ H₀ rejected? → Firm effects significant");
    println!("│  └─ H₀ not rejected? → Use Pooled OLS");
    println!("│");
    println!("├─ Step 3: Hausman Test (if both reject)");
    println!("│  ├─ H₀ rejected? → Use Fixed Effects (RE inconsistent)");
    println!("│  └─ H₀ not rejected? → Use Random Effects (more efficient)");
    println!("│");
    println!("└─ Step 4: Compare AIC/BIC for final decision");

    println!("\n🎯 RECOMMENDATION FOR THIS DATA:");
    if lm_p < 0.05 {
        println!("   → LM test rejects H₀");
        println!("   → Panel effects are significant");
        println!("   → Consider Random Effects or Fixed Effects model");
        println!("   → Run F-test and Hausman test for further model selection");
    } else {
        println!("   → LM test does not reject H₀");
        println!("   → No evidence of panel effects");
        println!("   → Pooled OLS is adequate");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("✨ SUMMARY OF v0.9.0 FEATURES");
    println!("══════════════════════════════════════════════════════════════════════════════");

    println!("\n1. MODEL SELECTION:");
    println!("   • Compare multiple models by AIC/BIC");
    println!("   • Automatic ranking and sorting");
    println!("   • Akaike weights for model averaging");
    println!("   • Δ_AIC interpretation guidelines");

    println!("\n2. PANEL DIAGNOSTICS:");
    println!("   • Breusch-Pagan LM test for random effects");
    println!("   • F-test for fixed effects vs pooled OLS");
    println!("   • Decision tree for model selection");

    println!("\n3. SUMMARY STATISTICS:");
    println!("   • Comprehensive descriptive stats (mean, std, quantiles)");
    println!("   • Pretty-printed tables");
    println!("   • Easy variable comparison");

    println!("\n4. WHEN TO USE:");
    println!("   • Model Selection: Comparing non-nested models");
    println!("   • BP LM Test: Testing for random effects");
    println!("   • F-test FE: Testing for fixed effects");
    println!("   • Summary Stats: Initial data exploration");

    println!("\n5. STATA/R/PYTHON EQUIVALENTS:");
    println!("   • Stata: xttest0 (BP LM), testparm (F-test), estat ic (AIC/BIC)");
    println!("   • R: plm::plmtest(), pFtest(), AIC()");
    println!("   • Python: linearmodels.panel diagnostics");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}
