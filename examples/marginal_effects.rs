use greeners::{Logit, Probit, DataFrame, Formula};
use ndarray::Array1;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("   Greeners v0.5.0 - Marginal Effects for Binary Choice Models");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // CREATE DATASET: College Admission Example
    // ═══════════════════════════════════════════════════════════════════════════

    println!("Dataset: College Admission (N = 20)");
    println!("Variables:");
    println!("  • admitted: 1 = admitted, 0 = rejected");
    println!("  • gpa: Grade Point Average (0-4 scale)");
    println!("  • sat: SAT score (standardized: mean 0, sd 1)");
    println!("  • legacy: 1 = legacy student, 0 = non-legacy\n");

    let mut data = HashMap::new();

    // Create realistic data with noise to avoid perfect separation
    // Pattern: Higher GPA + Higher SAT → Higher admission chance (but not perfect)
    data.insert("admitted".to_string(), Array1::from(vec![
        0.0, 0.0, 0.0, 1.0, 0.0,  // Low scores: mostly rejected, some luck
        0.0, 1.0, 0.0, 1.0, 0.0,  // Medium-low: mixed
        1.0, 0.0, 1.0, 1.0, 0.0,  // Medium-high: mixed (some unlucky rejections)
        1.0, 1.0, 0.0, 1.0, 1.0,  // High: mostly admitted, occasional rejection
    ]));

    // GPA (0-4 scale) - with variation within groups
    data.insert("gpa".to_string(), Array1::from(vec![
        2.5, 2.7, 2.6, 2.9, 2.8,  // Low-medium
        3.1, 3.0, 2.9, 3.2, 3.0,  // Medium
        3.3, 3.2, 3.4, 3.5, 3.3,  // Medium-high
        3.6, 3.7, 3.5, 3.8, 3.7,  // High
    ]));

    // SAT (standardized) - with variation
    data.insert("sat".to_string(), Array1::from(vec![
        -1.0, -0.8, -1.2, -0.5, -0.9,  // Below average
        -0.3, -0.1, -0.4, 0.1, -0.2,   // Slightly below average
        0.3, 0.0, 0.5, 0.6, 0.2,       // Above average
        0.8, 1.0, 0.7, 1.2, 0.9,       // Well above average
    ]));

    // Legacy status (binary) - mixed throughout
    data.insert("legacy".to_string(), Array1::from(vec![
        0.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
    ]));

    let df = DataFrame::new(data)?;

    // ═══════════════════════════════════════════════════════════════════════════
    // LOGIT MODEL
    // ═══════════════════════════════════════════════════════════════════════════

    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("LOGIT MODEL: admitted ~ gpa + sat + legacy");
    println!("─────────────────────────────────────────────────────────────────────────────\n");

    let formula = Formula::parse("admitted ~ gpa + sat + legacy")?;
    let logit_result = Logit::from_formula(&formula, &df)?;
    println!("{}", logit_result);

    // Get design matrix for marginal effects
    let (_, x) = df.to_design_matrix(&formula)?;

    // Calculate marginal effects
    let ame_logit = logit_result.average_marginal_effects(&x)?;
    let mem_logit = logit_result.marginal_effects_at_means(&x)?;

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("LOGIT MARGINAL EFFECTS");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Average Marginal Effects (AME) - RECOMMENDED");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("\nAME = (1/n) Σᵢ [∂P(yᵢ=1|xᵢ)/∂xⱼ]\n");
    println!("Interpretation: Average effect of 1-unit increase in X on Pr(admitted=1)\n");
    println!("{:-^78}", "");
    println!("{:<15} | {:>12} | {:>12} | {:>15}", "Variable", "Coefficient", "AME", "Interpretation");
    println!("{:-^78}", "");
    println!("{:<15} | {:>12.4} | {:>12} | {:>15}", "Intercept", logit_result.params[0], "-", "Baseline");
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>15}",
        "gpa", logit_result.params[1], ame_logit[1], "+1 GPA point");
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>15}",
        "sat", logit_result.params[2], ame_logit[2], "+1 SD in SAT");
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>15}",
        "legacy", logit_result.params[3], ame_logit[3], "Legacy effect");
    println!("{:-^78}", "");

    println!("\n📊 INTERPRETATION GUIDE:");
    println!("   • Coefficients show LOG-ODDS (hard to interpret directly)");
    println!("   • AME shows PROBABILITY change (easy to interpret!)");
    println!("\n   Example: If AME[gpa] = {:.4}:", ame_logit[1]);
    println!("     → A 1-point increase in GPA increases admission probability by {:.1}%",
        ame_logit[1] * 100.0);
    println!("     → For a student with 50% admission chance, this raises it to {:.1}%",
        (0.5 + ame_logit[1]) * 100.0);

    println!("\n─────────────────────────────────────────────────────────────────────────────");
    println!("Marginal Effects at Means (MEM)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("\nMEM = ∂P(y=1|x̄)/∂xⱼ evaluated at sample means\n");
    println!("{:-^78}", "");
    println!("{:<15} | {:>12} | {:>12} | {:>20}", "Variable", "Coefficient", "MEM", "Difference vs AME");
    println!("{:-^78}", "");
    println!("{:<15} | {:>12.4} | {:>12} | {:>20}", "Intercept", logit_result.params[0], "-", "-");
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>20.4}",
        "gpa", logit_result.params[1], mem_logit[1], mem_logit[1] - ame_logit[1]);
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>20.4}",
        "sat", logit_result.params[2], mem_logit[2], mem_logit[2] - ame_logit[2]);
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>20.4}",
        "legacy", logit_result.params[3], mem_logit[3], mem_logit[3] - ame_logit[3]);
    println!("{:-^78}", "");

    println!("\n⚠️  AME vs MEM:");
    println!("   • AME: Averages marginal effects across ALL observations");
    println!("   • MEM: Evaluates marginal effect at AVERAGE observation");
    println!("   • AME is RECOMMENDED because:");
    println!("     - Accounts for heterogeneity in the sample");
    println!("     - More robust to non-linearities");
    println!("     - MEM can evaluate at impossible values (e.g., average of dummies)");

    // ═══════════════════════════════════════════════════════════════════════════
    // PROBIT MODEL
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n─────────────────────────────────────────────────────────────────────────────");
    println!("PROBIT MODEL: admitted ~ gpa + sat + legacy");
    println!("─────────────────────────────────────────────────────────────────────────────\n");

    let probit_result = Probit::from_formula(&formula, &df)?;
    println!("{}", probit_result);

    let ame_probit = probit_result.average_marginal_effects(&x)?;
    let _mem_probit = probit_result.marginal_effects_at_means(&x)?;

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("PROBIT MARGINAL EFFECTS");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Average Marginal Effects (AME)");
    println!("─────────────────────────────────────────────────────────────────────────────\n");
    println!("{:-^78}", "");
    println!("{:<15} | {:>12} | {:>12} | {:>15}", "Variable", "Coefficient", "AME", "Interpretation");
    println!("{:-^78}", "");
    println!("{:<15} | {:>12.4} | {:>12} | {:>15}", "Intercept", probit_result.params[0], "-", "Baseline");
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>15}",
        "gpa", probit_result.params[1], ame_probit[1], "+1 GPA point");
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>15}",
        "sat", probit_result.params[2], ame_probit[2], "+1 SD in SAT");
    println!("{:<15} | {:>12.4} | {:>12.4} | {:>15}",
        "legacy", probit_result.params[3], ame_probit[3], "Legacy effect");
    println!("{:-^78}", "");

    // ═══════════════════════════════════════════════════════════════════════════
    // LOGIT vs PROBIT COMPARISON
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("LOGIT vs PROBIT COMPARISON");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Model Fit Comparison");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("{:<25} | {:>20} | {:>20}", "Metric", "Logit", "Probit");
    println!("{:-^78}", "");
    println!("{:<25} | {:>20.4} | {:>20.4}", "Log-Likelihood",
        logit_result.log_likelihood, probit_result.log_likelihood);
    println!("{:<25} | {:>20.4} | {:>20.4}", "Pseudo R²",
        logit_result.pseudo_r2, probit_result.pseudo_r2);
    println!("{:<25} | {:>20} | {:>20}", "Iterations",
        logit_result.iterations, probit_result.iterations);
    println!("{:-^78}", "");

    println!("\n─────────────────────────────────────────────────────────────────────────────");
    println!("Marginal Effects Comparison (AME)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("{:<15} | {:>15} | {:>15} | {:>15}", "Variable", "Logit AME", "Probit AME", "Difference");
    println!("{:-^78}", "");
    println!("{:<15} | {:>15.4} | {:>15.4} | {:>15.4}",
        "gpa", ame_logit[1], ame_probit[1], ame_logit[1] - ame_probit[1]);
    println!("{:<15} | {:>15.4} | {:>15.4} | {:>15.4}",
        "sat", ame_logit[2], ame_probit[2], ame_logit[2] - ame_probit[2]);
    println!("{:<15} | {:>15.4} | {:>15.4} | {:>15.4}",
        "legacy", ame_logit[3], ame_probit[3], ame_logit[3] - ame_probit[3]);
    println!("{:-^78}", "");

    println!("\n📊 KEY INSIGHTS:");
    println!("   • Logit and Probit typically give VERY SIMILAR marginal effects");
    println!("   • Main difference: Probit assumes normal errors, Logit assumes logistic");
    println!("   • Logit has fatter tails (more weight to extreme probabilities)");
    println!("   • In practice: Choice between Logit/Probit rarely matters for AME");
    println!("   • Both are better than Linear Probability Model (LPM) for binary outcomes");

    // ═══════════════════════════════════════════════════════════════════════════
    // PREDICTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("PREDICTIONS");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    println!("Let's predict admission probability for 3 hypothetical students:\n");

    // Create prediction dataset manually (with intercept)
    use ndarray::Array2;

    // Student 1: Low GPA, Low SAT, Non-legacy
    // Student 2: High GPA, High SAT, Non-legacy
    // Student 3: Medium GPA, Medium SAT, Legacy
    let x_new = Array2::from_shape_vec((3, 4), vec![
        1.0, 2.5, -1.5, 0.0,  // Intercept, GPA, SAT, Legacy
        1.0, 3.9,  1.5, 0.0,
        1.0, 3.2,  0.0, 1.0,
    ])?;

    let probs_logit = logit_result.predict_proba(&x_new);
    let probs_probit = probit_result.predict_proba(&x_new);

    println!("{:-^78}", "");
    println!("{:<10} | {:>8} | {:>8} | {:>8} | {:>15} | {:>15}",
        "Student", "GPA", "SAT", "Legacy", "Logit P(admit)", "Probit P(admit)");
    println!("{:-^78}", "");
    println!("{:<10} | {:>8.1} | {:>8.1} | {:>8.0} | {:>15.1}% | {:>15.1}%",
        "Low", 2.5, -1.5, 0.0, probs_logit[0] * 100.0, probs_probit[0] * 100.0);
    println!("{:<10} | {:>8.1} | {:>8.1} | {:>8.0} | {:>15.1}% | {:>15.1}%",
        "High", 3.9, 1.5, 0.0, probs_logit[1] * 100.0, probs_probit[1] * 100.0);
    println!("{:<10} | {:>8.1} | {:>8.1} | {:>8.0} | {:>15.1}% | {:>15.1}%",
        "Legacy", 3.2, 0.0, 1.0, probs_logit[2] * 100.0, probs_probit[2] * 100.0);
    println!("{:-^78}", "");

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("✨ SUMMARY OF MARGINAL EFFECTS");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("\n1. WHEN TO USE:");
    println!("   • ALWAYS report marginal effects for Logit/Probit models");
    println!("   • Coefficients alone are NOT interpretable (log-odds scale)");
    println!("   • Marginal effects give PROBABILITY changes (easy to understand)");

    println!("\n2. AME vs MEM:");
    println!("   • AME (Average Marginal Effects) - RECOMMENDED");
    println!("     - Averages effects across all observations");
    println!("     - Accounts for sample heterogeneity");
    println!("     - More robust");
    println!("   • MEM (Marginal Effects at Means)");
    println!("     - Evaluates at average observation");
    println!("     - Can give misleading results with dummies");
    println!("     - Use only when specifically required");

    println!("\n3. INTERPRETATION:");
    println!("   • AME[x] = {:.4} means:", ame_logit[1]);
    println!("     \"A 1-unit increase in x increases P(y=1) by {:.1} percentage points\"",
        ame_logit[1] * 100.0);
    println!("   • This holds ON AVERAGE across the sample");

    println!("\n4. REPORTING:");
    println!("   • Always report BOTH coefficients AND marginal effects");
    println!("   • Coefficients for statistical significance (z-test, p-values)");
    println!("   • Marginal effects for economic/substantive significance");

    println!("\n5. STATA/R/PYTHON EQUIVALENTS:");
    println!("   • Stata: margins, dydx(*) atmeans  [MEM]");
    println!("   •         margins, dydx(*)          [AME]");
    println!("   • R: mfx::logitmfx() or margins::margins()");
    println!("   • Python: statsmodels get_margeff()");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}
