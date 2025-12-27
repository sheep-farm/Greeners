use greeners::{OLS, DataFrame, Formula, CovarianceType};
use ndarray::Array1;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("                    Greeners v0.3.0 - New Features Demo");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // Create sample data: wage determination model
    let mut data = HashMap::new();

    // Wage (dependent variable)
    data.insert("wage".to_string(), Array1::from(vec![
        15.5, 18.2, 22.1, 25.3, 19.8, 28.5, 32.1, 24.9, 21.3, 26.7,
        17.2, 20.8, 23.5, 27.2, 30.1, 19.5, 22.8, 26.4, 29.7, 24.2,
    ]));

    // Education (years)
    data.insert("education".to_string(), Array1::from(vec![
        12.0, 14.0, 16.0, 18.0, 14.0, 16.0, 18.0, 16.0, 14.0, 16.0,
        12.0, 14.0, 16.0, 18.0, 20.0, 13.0, 15.0, 17.0, 19.0, 15.0,
    ]));

    // Experience (years)
    data.insert("experience".to_string(), Array1::from(vec![
        5.0, 7.0, 8.0, 10.0, 6.0, 12.0, 15.0, 9.0, 7.0, 11.0,
        4.0, 6.0, 8.0, 10.0, 12.0, 5.0, 7.0, 9.0, 11.0, 8.0,
    ]));

    // Female indicator (0 = male, 1 = female)
    data.insert("female".to_string(), Array1::from(vec![
        0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    ]));

    let df = DataFrame::new(data)?;

    println!("Dataset: Wage Determination (N = 20)");
    println!("Variables: wage, education, experience, female\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // FEATURE 1: INTERACTION TERMS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("FEATURE 1: INTERACTION TERMS");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // Model 1: Without interaction (baseline)
    println!("Model 1: wage ~ education + experience + female (no interaction)");
    let formula1 = Formula::parse("wage ~ education + experience + female")?;
    let result1 = OLS::from_formula(&formula1, &df, CovarianceType::HC3)?;
    println!("{}", result1);

    // Model 2: With full interaction (education * female)
    println!("\nModel 2: wage ~ education * female + experience");
    println!("Full interaction expands to: education + female + education:female + experience\n");
    let formula2 = Formula::parse("wage ~ education * female + experience")?;
    let result2 = OLS::from_formula(&formula2, &df, CovarianceType::HC3)?;
    println!("{}", result2);

    println!("\n📊 INTERPRETATION:");
    println!("   • education:female coefficient shows if education returns differ by gender");
    println!("   • Positive interaction → education has higher returns for females");
    println!("   • Negative interaction → education has lower returns for females");
    println!("   • Compare Model 1 R² vs Model 2 R² to assess interaction importance\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // FEATURE 2: ROBUST STANDARD ERROR COMPARISON (HC1, HC2, HC3)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("FEATURE 2: HETEROSCEDASTICITY-ROBUST SE COMPARISON");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    let formula3 = Formula::parse("wage ~ education + experience")?;

    println!("Same model estimated with different robust SE estimators:\n");

    let result_hc1 = OLS::from_formula(&formula3, &df, CovarianceType::HC1)?;
    let result_hc2 = OLS::from_formula(&formula3, &df, CovarianceType::HC2)?;
    let result_hc3 = OLS::from_formula(&formula3, &df, CovarianceType::HC3)?;

    println!("{:<20} | {:>12} | {:>12} | {:>12}", "Variable", "HC1", "HC2", "HC3 (Recommended)");
    println!("{:-<20}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "");

    for i in 0..result_hc1.params.len() {
        let var_name = match i {
            0 => "Intercept",
            1 => "Education",
            2 => "Experience",
            _ => "Unknown",
        };

        println!(
            "{:<20} | {:>12.4} | {:>12.4} | {:>12.4}",
            var_name,
            result_hc1.std_errors[i],
            result_hc2.std_errors[i],
            result_hc3.std_errors[i]
        );
    }

    println!("\n📊 WHICH TO USE?");
    println!("   • HC1 (White, 1980): Most common, uses small-sample correction n/(n-k)");
    println!("   • HC2 (Leverage-adjusted): More efficient with small samples");
    println!("   • HC3 (Jackknife): Most robust for small samples - RECOMMENDED DEFAULT");
    println!("   • HC3 standard errors are typically LARGER (more conservative)\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // FEATURE 3: POST-ESTIMATION PREDICTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("FEATURE 3: POST-ESTIMATION PREDICTIONS");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // Use formula with interaction for prediction example
    let formula4 = Formula::parse("wage ~ education + experience")?;
    let result4 = OLS::from_formula(&formula4, &df, CovarianceType::HC3)?;

    println!("Fitted Model: wage = β₀ + β₁·education + β₂·experience\n");
    println!("Estimated Coefficients:");
    println!("   β₀ (Intercept) = {:.4}", result4.params[0]);
    println!("   β₁ (Education) = {:.4}", result4.params[1]);
    println!("   β₂ (Experience) = {:.4}", result4.params[2]);

    // Create new data for prediction
    println!("\n🔮 OUT-OF-SAMPLE PREDICTIONS:");
    println!("{:-<78}", "");
    println!("{:<15} | {:<15} | {:>15} | {:>15}", "Education", "Experience", "Predicted Wage", "Description");
    println!("{:-<78}", "");

    // Scenario 1: High school graduate, entry level
    let scenario1 = ndarray::Array2::from_shape_vec((1, 3), vec![1.0, 12.0, 0.0])?;
    let pred1 = result4.predict(&scenario1);
    println!("{:<15} | {:<15} | {:>15.2} | {:>15}", "12 years", "0 years", pred1[0], "Entry-level");

    // Scenario 2: Bachelor's degree, 5 years exp
    let scenario2 = ndarray::Array2::from_shape_vec((1, 3), vec![1.0, 16.0, 5.0])?;
    let pred2 = result4.predict(&scenario2);
    println!("{:<15} | {:<15} | {:>15.2} | {:>15}", "16 years", "5 years", pred2[0], "Mid-career");

    // Scenario 3: Master's degree, 10 years exp
    let scenario3 = ndarray::Array2::from_shape_vec((1, 3), vec![1.0, 18.0, 10.0])?;
    let pred3 = result4.predict(&scenario3);
    println!("{:<15} | {:<15} | {:>15.2} | {:>15}", "18 years", "10 years", pred3[0], "Senior");

    // Scenario 4: PhD, 15 years exp
    let scenario4 = ndarray::Array2::from_shape_vec((1, 3), vec![1.0, 20.0, 15.0])?;
    let pred4 = result4.predict(&scenario4);
    println!("{:<15} | {:<15} | {:>15.2} | {:>15}", "20 years", "15 years", pred4[0], "Expert");

    println!("\n📊 INTERPRETATION:");
    println!("   • Each additional year of education increases wage by ${:.2}/hr", result4.params[1]);
    println!("   • Each additional year of experience increases wage by ${:.2}/hr", result4.params[2]);
    println!("   • Use predictions for policy analysis, salary benchmarking, etc.\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("✨ SUMMARY OF v0.3.0 FEATURES");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("\n1. INTERACTION TERMS:");
    println!("   • Syntax: y ~ x1 * x2 (full interaction: x1 + x2 + x1:x2)");
    println!("   • Syntax: y ~ x1 : x2 (interaction only)");
    println!("   • Essential for testing differential effects");

    println!("\n2. ADDITIONAL ROBUST SE ESTIMATORS:");
    println!("   • HC2: Leverage-adjusted (more efficient, small samples)");
    println!("   • HC3: Jackknife (most robust - RECOMMENDED)");
    println!("   • Complements existing HC1 and Newey-West");

    println!("\n3. POST-ESTIMATION METHODS:");
    println!("   • result.predict(&x_new) - Generate predictions");
    println!("   • result.fitted_values(&x) - In-sample fitted values");
    println!("   • result.residuals(&y, &x) - Calculate residuals");

    println!("\n✅ All features work with formula API and all estimators!");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}
