use greeners::{CovarianceType, DataFrame, Formula, OLS};

/// Example demonstrating CSV file reading with Formula API
///
/// This shows how to:
/// 1. Load data directly from a CSV file with headers
/// 2. Use formula syntax to specify models
/// 3. Run regressions with robust standard errors
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=^78}", " CSV + Formula API Example ");
    println!("\nDemonstrating how to load CSV data and run regressions using formulas.\n");

    // Load data from CSV file (with headers)
    let df = DataFrame::from_csv("examples/data/sample_data.csv")?;

    println!("✅ CSV file loaded successfully!");
    println!("   Rows: {}", df.n_rows());
    println!("   Columns: {}", df.n_cols());
    println!("   Variables: {:?}\n", df.column_names());

    // Example 1: Simple OLS with formula
    println!("{:=^78}", " Example 1: OLS with Formula ");
    let formula = Formula::parse("y ~ x1 + x2")?;
    println!(
        "Formula: {} ~ {}\n",
        formula.dependent,
        formula.independents.join(" + ")
    );

    let result = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;
    println!("{}", result);

    // Example 2: Different model specification
    println!("\n{:=^78}", " Example 2: Different Variables ");
    let formula2 = Formula::parse("y ~ x1 + x3")?;
    println!(
        "Formula: {} ~ {}\n",
        formula2.dependent,
        formula2.independents.join(" + ")
    );

    let result2 = OLS::from_formula(&formula2, &df, CovarianceType::HC1)?;
    println!("{}", result2);

    // Example 3: All variables
    println!("\n{:=^78}", " Example 3: All Variables ");
    let formula3 = Formula::parse("y ~ x1 + x2 + x3")?;
    println!(
        "Formula: {} ~ {}\n",
        formula3.dependent,
        formula3.independents.join(" + ")
    );

    let result3 = OLS::from_formula(&formula3, &df, CovarianceType::HC1)?;
    println!("{}", result3);

    println!("\n{:=^78}", "");
    println!("\n✨ Summary:");
    println!("  - Data loaded from CSV with headers");
    println!("  - Multiple models estimated using formula syntax");
    println!("  - Just like Python/R, but with Rust's speed!\n");

    println!("Python equivalent:");
    println!("  import pandas as pd");
    println!("  import statsmodels.formula.api as smf");
    println!("  df = pd.read_csv('sample_data.csv')");
    println!("  model = smf.ols('y ~ x1 + x2', data=df).fit(cov_type='HC1')");

    println!("\nR equivalent:");
    println!("  library(sandwich)");
    println!("  df <- read.csv('sample_data.csv')");
    println!("  model <- lm(y ~ x1 + x2, data=df)");
    println!("  coeftest(model, vcov=vcovHC(model, type='HC1'))\n");

    println!("{:=^78}", "");

    Ok(())
}
