// Complete guide to dummy variables in Greeners
// Shows both automatic Bool detection AND C(var) categorical encoding

use greeners::{CovarianceType, DataFrame, Formula, OLS};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Complete Guide to Dummy Variables in Greeners               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Create a comprehensive dataset
    create_dataset()?;

    let df = DataFrame::from_csv("wage_data.csv")?;
    println!("📊 Dataset: Wage data (N = {})\n", df.n_rows());

    // Show auto-detected types
    println!("═══════════════════════════════════════════════════════════════");
    println!("Auto-detected Column Types:");
    println!("═══════════════════════════════════════════════════════════════");
    for col in df.column_names() {
        if let Ok(column) = df.get_column(&col) {
            println!("  {:<15} -> {:?}", col, column.dtype());
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // PART 1: AUTOMATIC BINARY DUMMIES (Bool Detection)
    // ═══════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  PART 1: Automatic Binary Dummies (UNIQUE to Greeners!)      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Greeners automatically detects binary variables as Bool:");
    println!("  • female: ['Masculino', 'Feminino'] → Bool (0/1)");
    println!("  • married: ['Solteiro', 'Casado'] → Bool (0/1)");
    println!("  • union: ['Não', 'Sim'] → Bool (0/1)\n");

    println!("You can use them DIRECTLY in formulas - no conversion needed!\n");

    // Example 1: Single binary dummy
    println!("───────────────────────────────────────────────────────────────");
    println!("Example 1: Gender wage gap");
    println!("Formula: wage ~ education + experience + female");
    println!("───────────────────────────────────────────────────────────────\n");

    let formula1 = Formula::parse("wage ~ education + experience + female")?;
    let result1 = OLS::from_formula(&formula1, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!(
        "  β₀ (Intercept):    {:>8.2} (baseline wage for males)",
        result1.params[0]
    );
    println!(
        "  β₁ (education):    {:>8.2} (wage increase per year of education)",
        result1.params[1]
    );
    println!(
        "  β₂ (experience):   {:>8.2} (wage increase per year of experience)",
        result1.params[2]
    );
    println!(
        "  β₃ (female):       {:>8.2} (wage difference: female - male)",
        result1.params[3]
    );

    println!("\n📊 INTERPRETATION:");
    if result1.params[3] < 0.0 {
        println!(
            "  Women earn R$ {:.2} LESS than men (controlling for education & experience)",
            -result1.params[3]
        );
        println!(
            "  This is the \"gender wage gap\" after controlling for observable characteristics"
        );
    } else {
        println!("  Women earn R$ {:.2} MORE than men", result1.params[3]);
    }
    println!("  R² = {:.4}\n", result1.r_squared);

    // Example 2: Multiple binary dummies
    println!("───────────────────────────────────────────────────────────────");
    println!("Example 2: Multiple binary variables");
    println!("Formula: wage ~ education + female + married + union");
    println!("───────────────────────────────────────────────────────────────\n");

    let formula2 = Formula::parse("wage ~ education + female + married + union")?;
    let result2 = OLS::from_formula(&formula2, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!("  Intercept:   {:>8.2}", result2.params[0]);
    println!(
        "  education:   {:>8.2} (each additional year of schooling)",
        result2.params[1]
    );
    println!(
        "  female:      {:>8.2} (being female vs male)",
        result2.params[2]
    );
    println!(
        "  married:     {:>8.2} (being married vs single)",
        result2.params[3]
    );
    println!(
        "  union:       {:>8.2} (union member vs non-member)",
        result2.params[4]
    );

    println!("\n📊 INTERPRETATION:");
    println!("  Reference group: Single, non-union, male worker");
    if result2.params[3] > 0.0 {
        println!(
            "  Marriage premium: R$ {:.2} higher wage",
            result2.params[3]
        );
    }
    if result2.params[4] > 0.0 {
        println!("  Union premium: R$ {:.2} higher wage", result2.params[4]);
    }
    println!("  R² = {:.4}\n", result2.r_squared);

    // Example 3: Interaction with binary dummy
    println!("───────────────────────────────────────────────────────────────");
    println!("Example 3: Returns to education by gender");
    println!("Formula: wage ~ education * female");
    println!("───────────────────────────────────────────────────────────────\n");

    let formula3 = Formula::parse("wage ~ education * female")?;
    let result3 = OLS::from_formula(&formula3, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!(
        "  Intercept:           {:>8.2} (baseline for males, 0 education)",
        result3.params[0]
    );
    println!(
        "  education:           {:>8.2} (return to education for MALES)",
        result3.params[1]
    );
    println!(
        "  female:              {:>8.2} (base difference for females)",
        result3.params[2]
    );
    println!(
        "  education:female:    {:>8.2} (ADDITIONAL return to education for females)",
        result3.params[3]
    );

    println!("\n📊 INTERPRETATION:");
    println!(
        "  Return to education for males:   R$ {:.2} per year",
        result3.params[1]
    );
    println!(
        "  Return to education for females: R$ {:.2} per year",
        result3.params[1] + result3.params[3]
    );
    if result3.params[3].abs() > 1.0 {
        if result3.params[3] > 0.0 {
            println!("  ✓ Women have HIGHER returns to education than men!");
        } else {
            println!("  ✗ Women have LOWER returns to education than men");
        }
    }
    println!("  R² = {:.4}\n", result3.r_squared);

    // ═══════════════════════════════════════════════════════════════════
    // PART 2: CATEGORICAL DUMMIES (C(var) for 3+ categories)
    // ═══════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  PART 2: Categorical Dummies C(var) - 3+ Categories          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("For categorical variables with 3+ categories, use C(var):");
    println!("  • region: [1, 2, 3, 4] = [Norte, Sul, Leste, Oeste]");
    println!("  • C(region) creates 3 dummies (drops first as reference)\n");

    // Example 4: Categorical variable
    println!("───────────────────────────────────────────────────────────────");
    println!("Example 4: Regional wage differences");
    println!("Formula: wage ~ education + C(region)");
    println!("───────────────────────────────────────────────────────────────\n");

    let formula4 = Formula::parse("wage ~ education + C(region)")?;
    let result4 = OLS::from_formula(&formula4, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!(
        "  Intercept:     {:>8.2} (baseline: Norte, 0 education)",
        result4.params[0]
    );
    println!(
        "  education:     {:>8.2} (return to education)",
        result4.params[1]
    );
    println!(
        "  region_2 (Sul):   {:>8.2} (Sul vs Norte)",
        result4.params[2]
    );
    println!(
        "  region_3 (Leste): {:>8.2} (Leste vs Norte)",
        result4.params[3]
    );
    println!(
        "  region_4 (Oeste): {:>8.2} (Oeste vs Norte)",
        result4.params[4]
    );

    println!("\n📊 INTERPRETATION:");
    println!("  Reference region: Norte (region=1) - DROPPED as baseline");
    println!("  All coefficients are relative to Norte:");
    if result4.params[2] > 0.0 {
        println!(
            "  • Sul workers earn R$ {:.2} MORE than Norte",
            result4.params[2]
        );
    } else {
        println!(
            "  • Sul workers earn R$ {:.2} LESS than Norte",
            -result4.params[2]
        );
    }
    if result4.params[3] > 0.0 {
        println!(
            "  • Leste workers earn R$ {:.2} MORE than Norte",
            result4.params[3]
        );
    } else {
        println!(
            "  • Leste workers earn R$ {:.2} LESS than Norte",
            -result4.params[3]
        );
    }
    println!("  R² = {:.4}\n", result4.r_squared);

    // Example 5: Combining both types
    println!("───────────────────────────────────────────────────────────────");
    println!("Example 5: Combining binary AND categorical dummies");
    println!("Formula: wage ~ education + female + married + C(region)");
    println!("───────────────────────────────────────────────────────────────\n");

    let formula5 = Formula::parse("wage ~ education + female + married + C(region)")?;
    let result5 = OLS::from_formula(&formula5, &df, CovarianceType::HC3)?;

    println!("Coefficients:");
    println!("  Intercept:        {:>8.2}", result5.params[0]);
    println!("  education:        {:>8.2}", result5.params[1]);
    println!("  female (Bool):    {:>8.2}", result5.params[2]);
    println!("  married (Bool):   {:>8.2}", result5.params[3]);
    println!("  region_2:         {:>8.2}", result5.params[4]);
    println!("  region_3:         {:>8.2}", result5.params[5]);
    println!("  region_4:         {:>8.2}", result5.params[6]);

    println!("\n📊 INTERPRETATION:");
    println!("  Reference: Single, male, Norte worker");
    println!("  Model controls for:");
    println!("    • Education (continuous)");
    println!("    • Gender (binary dummy - auto-detected)");
    println!("    • Marital status (binary dummy - auto-detected)");
    println!("    • Region (3 categorical dummies via C(region))");
    println!("  R² = {:.4}\n", result5.r_squared);

    // Clean up
    std::fs::remove_file("wage_data.csv").ok();

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY: Dummy Variables in Greeners                        ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  1. BINARY DUMMIES (2 categories):                           ║");
    println!("║     ✅ Automatic detection: ['M','F'] → Bool                  ║");
    println!("║     ✅ Works in ANY language: ['sim','não'], etc.             ║");
    println!("║     ✅ Use directly: wage ~ female + married                  ║");
    println!("║     ✅ UNIQUE to Greeners - no other tool does this!         ║");
    println!("║                                                               ║");
    println!("║  2. CATEGORICAL DUMMIES (3+ categories):                     ║");
    println!("║     ✅ Use C(var): wage ~ C(region)                           ║");
    println!("║     ✅ Creates K-1 dummies (drops first as reference)        ║");
    println!("║     ✅ Standard R/Python syntax                               ║");
    println!("║                                                               ║");
    println!("║  3. INTERACTIONS:                                            ║");
    println!("║     ✅ Binary × continuous: wage ~ education * female        ║");
    println!("║     ✅ Binary × binary: wage ~ female * married              ║");
    println!("║     ✅ Categorical × continuous: wage ~ C(region)*education  ║");
    println!("║                                                               ║");
    println!("║  4. INTERPRETATION:                                          ║");
    println!("║     • Intercept = baseline (all dummies = 0)                 ║");
    println!("║     • Dummy coef = difference vs baseline                    ║");
    println!("║     • Interaction = differential effect by group             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

fn create_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create("wage_data.csv")?;

    // Header
    writeln!(
        file,
        "wage,education,experience,female,married,union,region"
    )?;

    // Data: Mix of values to create realistic patterns
    // Region: 1=Norte, 2=Sul, 3=Leste, 4=Oeste
    let data = vec![
        (3200, 12, 5, "Masculino", "Solteiro", "Não", 1),
        (3800, 16, 3, "Feminino", "Casado", "Sim", 2),
        (4200, 18, 7, "Masculino", "Casado", "Sim", 3),
        (2900, 11, 2, "Feminino", "Solteiro", "Não", 1),
        (5100, 20, 10, "Masculino", "Casado", "Sim", 4),
        (3500, 14, 4, "Feminino", "Solteiro", "Sim", 2),
        (4800, 18, 8, "Masculino", "Casado", "Não", 3),
        (3100, 12, 3, "Feminino", "Solteiro", "Não", 1),
        (4500, 16, 6, "Masculino", "Casado", "Sim", 4),
        (3300, 13, 4, "Feminino", "Solteiro", "Não", 2),
        (5500, 22, 12, "Masculino", "Casado", "Sim", 3),
        (3600, 14, 5, "Feminino", "Casado", "Não", 1),
        (4100, 16, 7, "Masculino", "Solteiro", "Sim", 4),
        (2800, 11, 1, "Feminino", "Solteiro", "Não", 2),
        (4900, 18, 9, "Masculino", "Casado", "Sim", 3),
        (3400, 13, 4, "Feminino", "Solteiro", "Não", 1),
        (5200, 20, 11, "Masculino", "Casado", "Sim", 4),
        (3700, 15, 5, "Feminino", "Casado", "Sim", 2),
        (4400, 17, 8, "Masculino", "Solteiro", "Não", 3),
        (3000, 12, 2, "Feminino", "Solteiro", "Não", 1),
    ];

    for (wage, edu, exp, gender, married, union, region) in data {
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            wage, edu, exp, gender, married, union, region
        )?;
    }

    Ok(())
}
