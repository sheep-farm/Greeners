// Practical example: Using auto-detected Bool variables directly in regression
// Demonstrates the bug fix that allows Bool, Int, and Categorical in formulas

use greeners::{CovarianceType, DataFrame, Formula, OLS};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║  Bool Variables in Formulas - Practical Example      ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Load real dataset with binary variables
    let df = DataFrame::from_csv("examples/data/cattaneo2.csv")?;

    println!("📊 Dataset: cattaneo2.csv ({} rows)\n", df.n_rows());

    // Show detected types for key variables
    println!("Auto-detected column types:");
    let bool_vars = vec!["mmarried", "mbsmoke", "fbaby", "alcohol", "mhisp", "fhisp"];
    for var in &bool_vars {
        if let Ok(column) = df.get_column(var) {
            println!("  {} -> {:?}", var, column.dtype());
        }
    }
    println!();

    // Example 1: Simple regression with Bool variables
    println!("═══════════════════════════════════════════════════════");
    println!("Example 1: Birth weight ~ smoking + married");
    println!("═══════════════════════════════════════════════════════\n");

    let formula1 = Formula::parse("bweight ~ mbsmoke + mmarried")?;
    let result1 = OLS::from_formula(&formula1, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!("  Intercept: {:.2} grams", result1.params[0]);
    println!("  mbsmoke (mother smokes): {:.2} grams", result1.params[1]);
    println!(
        "  mmarried (mother married): {:.2} grams",
        result1.params[2]
    );
    println!("\nInterpretation:");
    println!(
        "  • Maternal smoking associated with {:.0}g lower birth weight",
        -result1.params[1]
    );
    println!(
        "  • Being married associated with {:.0}g higher birth weight",
        result1.params[2]
    );
    println!("  • R² = {:.4}\n", result1.r_squared);

    // Example 2: Multiple Bool controls
    println!("═══════════════════════════════════════════════════════");
    println!("Example 2: Adding more binary controls");
    println!("═══════════════════════════════════════════════════════\n");

    let formula2 = Formula::parse("bweight ~ mbsmoke + mmarried + fbaby + mhisp")?;
    let result2 = OLS::from_formula(&formula2, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!("  Intercept: {:.2}", result2.params[0]);
    println!("  mbsmoke: {:.2}", result2.params[1]);
    println!("  mmarried: {:.2}", result2.params[2]);
    println!("  fbaby (first baby): {:.2}", result2.params[3]);
    println!("  mhisp (hispanic): {:.2}", result2.params[4]);
    println!("  R² = {:.4}\n", result2.r_squared);

    // Example 3: Interaction with Bool
    println!("═══════════════════════════════════════════════════════");
    println!("Example 3: Interaction - smoking × married");
    println!("═══════════════════════════════════════════════════════\n");

    let formula3 = Formula::parse("bweight ~ mbsmoke * mmarried")?;
    let result3 = OLS::from_formula(&formula3, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!("  Intercept: {:.2}", result3.params[0]);
    println!("  mbsmoke: {:.2}", result3.params[1]);
    println!("  mmarried: {:.2}", result3.params[2]);
    println!("  mbsmoke:mmarried: {:.2}", result3.params[3]);
    println!("\nInterpretation:");
    if result3.params[3].abs() > 0.0 {
        println!("  • Smoking effect differs by marital status");
        println!(
            "  • Interaction coefficient: {:.2} grams",
            result3.params[3]
        );
    }
    println!("  • R² = {:.4}\n", result3.r_squared);

    // Example 4: Mix of Bool and Int
    println!("═══════════════════════════════════════════════════════");
    println!("Example 4: Bool + Int variables");
    println!("═══════════════════════════════════════════════════════\n");

    let formula4 = Formula::parse("bweight ~ mbsmoke + mmarried + mage + nprenatal")?;
    let result4 = OLS::from_formula(&formula4, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!("  Intercept: {:.2}", result4.params[0]);
    println!("  mbsmoke (Bool): {:.2}", result4.params[1]);
    println!("  mmarried (Bool): {:.2}", result4.params[2]);
    println!("  mage (Int): {:.2}", result4.params[3]);
    println!("  nprenatal (Int): {:.2}", result4.params[4]);
    println!("  R² = {:.4}\n", result4.r_squared);

    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║  KEY TAKEAWAY                                         ║");
    println!("╠═══════════════════════════════════════════════════════╣");
    println!("║  ✅ Bool variables work directly in formulas          ║");
    println!("║  ✅ No manual conversion needed (0/1 coding)          ║");
    println!("║  ✅ Interactions with Bool variables work             ║");
    println!("║  ✅ Mix Bool + Int + Float seamlessly                 ║");
    println!("║  ✅ Binary detection in ANY language!                 ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    Ok(())
}
