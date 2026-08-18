// Test what happens with perfect collinearity in Greeners
// Compare with Stata behavior

use greeners::{CovarianceType, DataFrame, Formula, OLS};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("Testing Perfect Collinearity Detection");
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

    println!("Data loaded: {} rows", df.n_rows());
    println!("Variables: y, x1, x2, x3 (where x3 = x1 + x2)\n");

    println!("Attempting: y ~ x1 + x2 + x3");
    match Formula::parse("y ~ x1 + x2 + x3") {
        Ok(formula) => match OLS::from_formula(&formula, &df, CovarianceType::HC3) {
            Ok(result) => {
                println!("✓ Regression succeeded (no error detected)");
                println!("  Coefficients: {:?}", result.params);
                println!("  ⚠️  This might indicate numerical instability!");
            }
            Err(e) => {
                println!("✗ Regression failed with error:");
                println!("  Error: {}", e);
                println!("  ✓ Expected behavior: Greeners detects singular matrix");
            }
        },
        Err(e) => {
            println!("✗ Formula parse failed: {}", e);
        }
    }

    // Test 2: Dummy variable trap
    println!("\n───────────────────────────────────────────────────────");
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

    println!("Data: male + female = 1 (perfect collinearity with intercept)");
    println!("Attempting: y ~ male + female (WITH intercept)\n");

    match Formula::parse("y ~ male + female") {
        Ok(formula) => match OLS::from_formula(&formula, &df2, CovarianceType::HC3) {
            Ok(result) => {
                println!("✓ Regression succeeded");
                println!("  Coefficients: {:?}", result.params);
                println!("  ⚠️  Should have dropped one dummy!");
            }
            Err(e) => {
                println!("✗ Regression failed:");
                println!("  Error: {}", e);
                println!("  ✓ Greeners detects the problem");
            }
        },
        Err(e) => {
            println!("✗ Formula error: {}", e);
        }
    }

    // Test 3: Without intercept (should work)
    println!("\n───────────────────────────────────────────────────────");
    println!("Test 3: No dummy trap (without intercept)");
    println!("───────────────────────────────────────────────────────\n");

    println!("Attempting: y ~ male + female - 1 (NO intercept)\n");

    match Formula::parse("y ~ male + female - 1") {
        Ok(formula) => match OLS::from_formula(&formula, &df2, CovarianceType::HC3) {
            Ok(result) => {
                println!("✓ Regression succeeded!");
                println!("  Coefficients: {:?}", result.params);
                println!("  ✓ No collinearity without intercept");
            }
            Err(e) => {
                println!("✗ Unexpected error: {}", e);
            }
        },
        Err(e) => {
            println!("✗ Formula error: {}", e);
        }
    }

    // Clean up
    std::fs::remove_file("collinear_data.csv").ok();
    std::fs::remove_file("dummy_trap.csv").ok();

    // Summary
    println!("\n═══════════════════════════════════════════════════════");
    println!("SUMMARY: Collinearity Handling in Greeners");
    println!("═══════════════════════════════════════════════════════\n");

    println!("Current behavior:");
    println!("  ✗ No automatic detection of collinear variables");
    println!("  ✗ No automatic dropping of redundant variables");
    println!("  ✓ Returns SingularMatrix error when detected");
    println!("  ✓ User must manually remove collinear variables\n");

    println!("Stata behavior (for comparison):");
    println!("  ✓ Automatically detects perfect collinearity");
    println!("  ✓ Drops redundant variables (marks as 'omitted')");
    println!("  ✓ Shows 'o.varname' in output");
    println!("  ✓ Estimation proceeds with non-collinear subset\n");

    println!("Recommendation:");
    println!("  → Implement automatic collinearity detection");
    println!("  → Drop redundant variables before estimation");
    println!("  → Report which variables were omitted");
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}
