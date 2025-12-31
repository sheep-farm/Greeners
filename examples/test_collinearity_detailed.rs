// Detailed test showing automatic collinearity detection in action
// This example prints full regression tables to demonstrate omitted variables

use greeners::{CovarianceType, DataFrame, Formula, OLS};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("Testing Automatic Collinearity Detection");
    println!("═══════════════════════════════════════════════════════\n");

    // Test 1: Perfect collinearity - x3 = x1 + x2
    println!("Test 1: Perfect collinearity (x3 = x1 + x2)");
    println!("───────────────────────────────────────────────────────\n");

    let mut file = File::create("collinear_data.csv")?;
    writeln!(file, "y,x1,x2,x3")?;
    writeln!(file, "10,1,2,3")?; // x3 = x1 + x2
    writeln!(file, "15,2,3,5")?; // x3 = x1 + x2
    writeln!(file, "20,3,4,7")?; // x3 = x1 + x2
    writeln!(file, "25,4,5,9")?; // x3 = x1 + x2
    writeln!(file, "30,5,6,11")?; // x3 = x1 + x2
    drop(file);

    let df = DataFrame::from_csv("collinear_data.csv")?;

    println!("Data: x3 = x1 + x2 (perfect linear dependence)");
    println!("Formula: y ~ x1 + x2 + x3\n");

    let formula1 = Formula::parse("y ~ x1 + x2 + x3")?;
    let result1 = OLS::from_formula(&formula1, &df, CovarianceType::HC3)?;

    println!("{}", result1);
    println!();

    // Test 2: Dummy variable trap
    println!("───────────────────────────────────────────────────────");
    println!("Test 2: Dummy variable trap (with intercept)");
    println!("───────────────────────────────────────────────────────\n");

    let mut file2 = File::create("dummy_trap.csv")?;
    writeln!(file2, "y,male,female")?;
    writeln!(file2, "100,1,0")?; // male + female = 1 always!
    writeln!(file2, "110,0,1")?;
    writeln!(file2, "105,1,0")?;
    writeln!(file2, "115,0,1")?;
    writeln!(file2, "108,1,0")?;
    drop(file2);

    let df2 = DataFrame::from_csv("dummy_trap.csv")?;

    println!("Data: male + female = 1 (collinear with intercept)");
    println!("Formula: y ~ male + female\n");

    let formula2 = Formula::parse("y ~ male + female")?;
    let result2 = OLS::from_formula(&formula2, &df2, CovarianceType::HC3)?;

    println!("{}", result2);
    println!();

    // Test 3: Without intercept (should work fine)
    println!("───────────────────────────────────────────────────────");
    println!("Test 3: Without intercept (no collinearity)");
    println!("───────────────────────────────────────────────────────\n");

    println!("Formula: y ~ male + female - 1\n");

    let formula3 = Formula::parse("y ~ male + female - 1")?;
    let result3 = OLS::from_formula(&formula3, &df2, CovarianceType::HC3)?;

    println!("{}", result3);
    println!();

    // Clean up
    std::fs::remove_file("collinear_data.csv").ok();
    std::fs::remove_file("dummy_trap.csv").ok();

    // Summary
    println!("═══════════════════════════════════════════════════════");
    println!("SUMMARY: Automatic Collinearity Detection");
    println!("═══════════════════════════════════════════════════════\n");

    println!("✅ Greeners now automatically:");
    println!("  • Detects perfect collinearity using QR decomposition");
    println!("  • Drops redundant variables before estimation");
    println!("  • Reports omitted variables with 'o.varname' notation");
    println!("  • Proceeds with estimation using non-collinear subset\n");

    println!("✅ Behavior matches Stata:");
    println!("  • No singular matrix errors");
    println!("  • Transparent reporting of omitted variables");
    println!("  • Keeps first occurrence, drops later ones");
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}
