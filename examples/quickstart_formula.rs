use greeners::{CovarianceType, DataFrame, Formula, OLS};
use ndarray::Array1;
use std::collections::HashMap;

/// Quick start example demonstrating the Formula API
/// This is the example shown in the README
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data (like a pandas DataFrame)
    let mut data = HashMap::new();
    data.insert(
        "y".to_string(),
        Array1::from(vec![1.0, 2.1, 3.2, 3.9, 5.1, 6.0, 7.2, 8.1, 9.0, 10.1]),
    );
    data.insert(
        "x1".to_string(),
        Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    );
    data.insert(
        "x2".to_string(),
        Array1::from(vec![1.5, 2.8, 2.9, 3.6, 3.8, 4.2, 5.1, 5.3, 6.2, 6.4]),
    );

    let df = DataFrame::new(data)?;

    println!("\n{:=^78}", " Greeners Quick Start: Formula API ");
    println!("\nThis example demonstrates how easy it is to run regressions");
    println!("using R/Python-style formula syntax in Rust!\n");

    // Specify model using formula (just like Python/R!)
    let formula = Formula::parse("y ~ x1 + x2")?;

    println!(
        "Model formula: {} ~ {}",
        formula.dependent,
        formula.independents.join(" + ")
    );

    // Estimate with robust standard errors
    let result = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;

    println!("{}", result);

    println!("\n{:=^78}", "");
    println!("\n✅ That's it! Just like in Python/R, but with Rust's speed and safety.");
    println!("\nPython equivalent:");
    println!("  import statsmodels.formula.api as smf");
    println!("  model = smf.ols('y ~ x1 + x2', data=df).fit(cov_type='HC1')");
    println!("\nR equivalent:");
    println!("  library(sandwich)");
    println!("  model <- lm(y ~ x1 + x2, data=df)");
    println!("  coeftest(model, vcov=vcovHC(model, type='HC1'))");
    println!("\n{:=^78}", "");

    Ok(())
}
