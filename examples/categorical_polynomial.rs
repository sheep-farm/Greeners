use greeners::{CovarianceType, DataFrame, Formula, OLS};
use ndarray::Array1;
use indexmap::IndexMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("   Greeners v0.4.0 - Categorical Variables & Polynomial Terms");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // FEATURE 1: CATEGORICAL VARIABLES
    // ═══════════════════════════════════════════════════════════════════════════

    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("FEATURE 1: CATEGORICAL VARIABLES C(var)");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // Create dataset with categorical variable (region: 0=North, 1=South, 2=East, 3=West)
    let mut data = IndexMap::new();

    data.insert(
        "sales".to_string(),
        Array1::from(vec![
            120.0, 135.0, 145.0, 110.0, 150.0, 165.0, 140.0, 125.0, 180.0, 195.0, 175.0, 160.0,
            200.0, 210.0, 190.0, 185.0,
        ]),
    );

    data.insert(
        "advertising".to_string(),
        Array1::from(vec![
            10.0, 12.0, 15.0, 8.0, 16.0, 18.0, 14.0, 11.0, 20.0, 22.0, 19.0, 17.0, 24.0, 25.0,
            23.0, 21.0,
        ]),
    );

    // Region: 0=North, 1=South, 2=East, 3=West
    data.insert(
        "region".to_string(),
        Array1::from(vec![
            0.0, 0.0, 0.0, 0.0, // North
            1.0, 1.0, 1.0, 1.0, // South
            2.0, 2.0, 2.0, 2.0, // East
            3.0, 3.0, 3.0, 3.0, // West
        ]),
    );

    let df = DataFrame::new(data)?;

    println!("Dataset: Sales across 4 regions (N = 16)");
    println!("Variables:");
    println!("  • sales: Total sales (continuous)");
    println!("  • advertising: Ad spending (continuous)");
    println!("  • region: 0=North, 1=South, 2=East, 3=West (categorical)\n");

    // Model WITHOUT categorical encoding
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Model 1: sales ~ advertising (WRONG - treats region as continuous)");
    println!("─────────────────────────────────────────────────────────────────────────────\n");

    let formula1 = Formula::parse("sales ~ advertising + region")?;
    let result1 = OLS::from_formula(&formula1, &df, CovarianceType::HC3)?;
    println!("{}", result1);

    println!("\n⚠️  PROBLEM: Treating region as continuous (0, 1, 2, 3) assumes:");
    println!("     • Linear relationship: South is '1 unit' more than North");
    println!("     • Equal spacing: North→South same as South→East");
    println!("     • This is WRONG for categorical data!\n");

    // Model WITH categorical encoding
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Model 2: sales ~ advertising + C(region) (CORRECT)");
    println!("─────────────────────────────────────────────────────────────────────────────\n");

    let formula2 = Formula::parse("sales ~ advertising + C(region)")?;
    let result2 = OLS::from_formula(&formula2, &df, CovarianceType::HC3)?;
    println!("{}", result2);

    println!("\n✅ CORRECT: C(region) creates dummy variables:");
    println!("     • Reference category: North (region=0) - DROPPED");
    println!("     • Dummy 1: South (region=1)");
    println!("     • Dummy 2: East (region=2)");
    println!("     • Dummy 3: West (region=3)");
    println!("\n📊 INTERPRETATION:");
    println!("     • Intercept: Baseline sales for North with zero advertising");
    println!("     • advertising: Effect of $1 ad spending (same across regions)");
    println!("     • Region dummies: Sales difference vs North (holding advertising constant)");
    println!("     • If 'South' coef = 15.0 → South has $15k higher sales than North\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // FEATURE 2: POLYNOMIAL TERMS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("FEATURE 2: POLYNOMIAL TERMS I(x^n)");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // Create data with non-linear relationship (diminishing returns)
    let mut data2 = IndexMap::new();

    data2.insert(
        "output".to_string(),
        Array1::from(vec![
            10.0, 18.0, 24.0, 28.0, 30.0, 31.0, 31.5, 31.8, 32.0, 32.1,
        ]),
    );

    data2.insert(
        "input".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    );

    let df2 = DataFrame::new(data2)?;

    println!("Dataset: Production function with diminishing returns (N = 10)");
    println!("Variables:");
    println!("  • output: Production output");
    println!("  • input: Input quantity (shows diminishing returns)\n");

    // Linear model (misspecified)
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Model 3: output ~ input (LINEAR - misspecified)");
    println!("─────────────────────────────────────────────────────────────────────────────\n");

    let formula3 = Formula::parse("output ~ input")?;
    let result3 = OLS::from_formula(&formula3, &df2, CovarianceType::HC3)?;
    println!("{}", result3);
    println!(
        "\n⚠️  PROBLEM: R² = {:.4} - poor fit due to non-linear relationship\n",
        result3.r_squared
    );

    // Quadratic model
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Model 4: output ~ input + I(input^2) (QUADRATIC)");
    println!("─────────────────────────────────────────────────────────────────────────────\n");

    let formula4 = Formula::parse("output ~ input + I(input^2)")?;
    let result4 = OLS::from_formula(&formula4, &df2, CovarianceType::HC3)?;
    println!("{}", result4);
    println!(
        "\n✅ MUCH BETTER: R² = {:.4} - captures curvature!",
        result4.r_squared
    );
    println!("\n📊 INTERPRETATION:");
    println!("     • output = β₀ + β₁·input + β₂·input²");
    println!(
        "     • β₁ (input) = {:.4} - linear effect",
        result4.params[1]
    );
    println!(
        "     • β₂ (input²) = {:.4} - quadratic effect",
        result4.params[2]
    );
    println!("     • If β₂ < 0: Diminishing returns (output increases at decreasing rate)");
    println!("     • If β₂ > 0: Increasing returns (output increases at increasing rate)\n");

    // Cubic model
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Model 5: output ~ input + I(input^2) + I(input^3) (CUBIC)");
    println!("─────────────────────────────────────────────────────────────────────────────\n");

    let formula5 = Formula::parse("output ~ input + I(input^2) + I(input^3)")?;
    let result5 = OLS::from_formula(&formula5, &df2, CovarianceType::HC3)?;
    println!("{}", result5);
    println!("\n📊 COMPARING POLYNOMIAL MODELS:");
    println!("{:-<78}", "");
    println!(
        "{:<25} | {:>15} | {:>15} | {:>15}",
        "Model", "R²", "Adj. R²", "AIC"
    );
    println!("{:-<78}", "");
    println!(
        "{:<25} | {:>15.4} | {:>15.4} | {:>15.2}",
        "Linear", result3.r_squared, result3.adj_r_squared, result3.aic
    );
    println!(
        "{:<25} | {:>15.4} | {:>15.4} | {:>15.2}",
        "Quadratic", result4.r_squared, result4.adj_r_squared, result4.aic
    );
    println!(
        "{:<25} | {:>15.4} | {:>15.4} | {:>15.2}",
        "Cubic", result5.r_squared, result5.adj_r_squared, result5.aic
    );
    println!("\n✅ Choose model with:");
    println!("     • Highest Adj. R² (penalizes overfitting)");
    println!("     • Lowest AIC (balances fit and complexity)");
    println!("     • Significant polynomial terms\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // COMBINING FEATURES
    // ═══════════════════════════════════════════════════════════════════════════

    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("COMBINING CATEGORICAL + POLYNOMIAL");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    println!("You can combine both features:\n");
    println!("Example: sales ~ C(region) + advertising + I(advertising^2)");
    println!("  • C(region): Categorical dummies for region effects");
    println!("  • advertising: Linear advertising effect");
    println!("  • I(advertising^2): Diminishing returns to advertising\n");

    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("✨ SUMMARY OF v0.4.0 CATEGORICAL & POLYNOMIAL FEATURES");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("\n1. CATEGORICAL VARIABLES C(var):");
    println!("   • Automatic dummy variable creation");
    println!("   • Drops first category (reference level)");
    println!("   • Essential for: regions, industries, treatment groups, etc.");
    println!("   • Python equivalent: pd.get_dummies(drop_first=True)");
    println!("   • R equivalent: factor() with contr.treatment");

    println!("\n2. POLYNOMIAL TERMS I(expr):");
    println!("   • Supports I(x^2), I(x^3), ..., I(x^n)");
    println!("   • Also accepts I(x**2) syntax");
    println!("   • Captures non-linear relationships");
    println!("   • Use for: production functions, wage curves, growth models");

    println!("\n3. COMBINED WITH:");
    println!("   • Interactions (x1 * x2)");
    println!("   • Robust SE (HC1, HC2, HC3)");
    println!("   • All estimators (OLS, IV, Panel, etc.)");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}
