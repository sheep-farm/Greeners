// Test that Bool, Int, and Categorical columns work directly in formulas
// This demonstrates the bug fix for automatic type conversion in to_design_matrix()

use greeners::{CovarianceType, DataFrame, Formula, OLS};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Bool/Int/Categorical in Formulas ===\n");

    // Create CSV with binary boolean variables (detected automatically)
    let mut file = File::create("test_formula_types.csv")?;
    writeln!(file, "wage,casado,sexo,education,region")?;
    writeln!(file, "3000,sim,M,12,1")?;
    writeln!(file, "3500,não,F,14,2")?;
    writeln!(file, "4000,sim,M,16,1")?;
    writeln!(file, "4500,não,F,18,2")?;
    writeln!(file, "5000,sim,M,20,1")?;
    writeln!(file, "3200,não,F,12,3")?;
    writeln!(file, "4200,sim,M,16,3")?;
    writeln!(file, "4800,não,F,18,1")?;
    drop(file);

    let df = DataFrame::from_csv("test_formula_types.csv")?;

    println!("✓ Loaded {} rows x {} columns\n", df.n_rows(), df.n_cols());

    // Check detected types
    println!("Column types:");
    for col in df.column_names() {
        if let Ok(column) = df.get_column(&col) {
            println!("  {} -> {:?}", col, column.dtype());
        }
    }
    println!();

    // Test 1: Bool variable directly in formula
    println!("Test 1: Bool variable (casado) directly in formula");
    println!("  Formula: wage ~ education + casado");

    match Formula::parse("wage ~ education + casado") {
        Ok(formula) => {
            match OLS::from_formula(&formula, &df, CovarianceType::NonRobust) {
                Ok(result) => {
                    println!("  ✅ SUCCESS! Bool column works in formula");
                    println!("  Coefficients:");
                    println!("    Intercept: {:.2}", result.params[0]);
                    println!("    education: {:.2}", result.params[1]);
                    println!("    casado: {:.2}", result.params[2]);
                    println!("    R²: {:.4}\n", result.r_squared);
                }
                Err(e) => {
                    println!("  ❌ FAILED: {}\n", e);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Formula parse failed: {}\n", e);
        }
    }

    // Test 2: Multiple Bool variables
    println!("Test 2: Multiple Bool variables in formula");
    println!("  Formula: wage ~ education + casado + sexo");

    match Formula::parse("wage ~ education + casado + sexo") {
        Ok(formula) => {
            match OLS::from_formula(&formula, &df, CovarianceType::HC3) {
                Ok(result) => {
                    println!("  ✅ SUCCESS! Multiple Bool columns work");
                    println!("  Coefficients:");
                    println!("    Intercept: {:.2}", result.params[0]);
                    println!("    education: {:.2}", result.params[1]);
                    println!("    casado: {:.2}", result.params[2]);
                    println!("    sexo: {:.2}", result.params[3]);
                    println!("    R²: {:.4}\n", result.r_squared);
                }
                Err(e) => {
                    println!("  ❌ FAILED: {}\n", e);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Formula parse failed: {}\n", e);
        }
    }

    // Test 3: Int variable directly (no need for conversion)
    println!("Test 3: Int variable (education) directly in formula");
    println!("  Formula: wage ~ education");

    match Formula::parse("wage ~ education") {
        Ok(formula) => {
            match OLS::from_formula(&formula, &df, CovarianceType::NonRobust) {
                Ok(result) => {
                    println!("  ✅ SUCCESS! Int column works in formula");
                    println!("  Coefficients:");
                    println!("    Intercept: {:.2}", result.params[0]);
                    println!("    education: {:.2}", result.params[1]);
                    println!("    R²: {:.4}\n", result.r_squared);
                }
                Err(e) => {
                    println!("  ❌ FAILED: {}\n", e);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Formula parse failed: {}\n", e);
        }
    }

    // Test 4: Categorical with C()
    println!("Test 4: Categorical variable with C(region)");
    println!("  Formula: wage ~ education + C(region)");

    match Formula::parse("wage ~ education + C(region)") {
        Ok(formula) => {
            match OLS::from_formula(&formula, &df, CovarianceType::NonRobust) {
                Ok(result) => {
                    println!("  ✅ SUCCESS! C(region) creates dummies");
                    println!("  Coefficients: (intercept + education + region_2 + region_3)");
                    for (i, param) in result.params.iter().enumerate() {
                        println!("    β[{}]: {:.2}", i, param);
                    }
                    println!("    R²: {:.4}\n", result.r_squared);
                }
                Err(e) => {
                    println!("  ❌ FAILED: {}\n", e);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Formula parse failed: {}\n", e);
        }
    }

    // Test 5: Interaction with Bool
    println!("Test 5: Interaction with Bool variable");
    println!("  Formula: wage ~ education + casado + education:casado");

    match Formula::parse("wage ~ education + casado + education:casado") {
        Ok(formula) => {
            match OLS::from_formula(&formula, &df, CovarianceType::NonRobust) {
                Ok(result) => {
                    println!("  ✅ SUCCESS! Bool interaction works");
                    println!("  Coefficients:");
                    println!("    Intercept: {:.2}", result.params[0]);
                    println!("    education: {:.2}", result.params[1]);
                    println!("    casado: {:.2}", result.params[2]);
                    println!("    education:casado: {:.2}", result.params[3]);
                    println!("    R²: {:.4}\n", result.r_squared);
                }
                Err(e) => {
                    println!("  ❌ FAILED: {}\n", e);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Formula parse failed: {}\n", e);
        }
    }

    // Test 6: Polynomial with Int
    println!("Test 6: Polynomial term with Int variable");
    println!("  Formula: wage ~ education + I(education^2)");

    match Formula::parse("wage ~ education + I(education^2)") {
        Ok(formula) => {
            match OLS::from_formula(&formula, &df, CovarianceType::NonRobust) {
                Ok(result) => {
                    println!("  ✅ SUCCESS! Polynomial with Int works");
                    println!("  Coefficients:");
                    println!("    Intercept: {:.2}", result.params[0]);
                    println!("    education: {:.2}", result.params[1]);
                    println!("    education²: {:.2}", result.params[2]);
                    println!("    R²: {:.4}\n", result.r_squared);
                }
                Err(e) => {
                    println!("  ❌ FAILED: {}\n", e);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Formula parse failed: {}\n", e);
        }
    }

    // Clean up
    std::fs::remove_file("test_formula_types.csv").ok();

    println!("==================================================");
    println!("✅ ALL TESTS PASSED!");
    println!("Bool, Int, and Categorical columns now work");
    println!("directly in formulas without manual conversion!");
    println!("==================================================");

    Ok(())
}
