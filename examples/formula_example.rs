use greeners::{CovarianceType, DataFrame, Formula, OLS, FGLS};
use ndarray::Array1;
use rand::prelude::*;
use statrs::distribution::Normal;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=^78}", " Formula API Example ");
    println!("Demonstrating R/Python-style formula parsing for econometrics\n");

    // Create sample data with some randomness (similar to a pandas DataFrame or R data.frame)
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 2.0).unwrap();

    let n = 100;
    let mut data = HashMap::new();

    // Generate independent variables
    let tratado: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
    let t: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
    let effect: Vec<f64> = (0..n).map(|i| (i as f64 / 10.0) + normal.sample(&mut rng)).collect();

    // Generate dependent variable with a linear relationship + noise
    let fte: Vec<f64> = (0..n)
        .map(|i| {
            10.0 + 5.0 * tratado[i] + 2.0 * t[i] + 1.5 * effect[i] + normal.sample(&mut rng)
        })
        .collect();

    data.insert("fte".to_string(), Array1::from(fte));
    data.insert("tratado".to_string(), Array1::from(tratado));
    data.insert("t".to_string(), Array1::from(t));
    data.insert("effect".to_string(), Array1::from(effect));

    let df = DataFrame::new(data)?;

    println!("Data loaded successfully!");
    println!("Number of observations: {}", df.n_rows());
    println!("Number of variables: {}", df.n_cols());
    println!("Variables: {:?}\n", df.column_names());

    // Example 1: OLS with formula (like Python's statsmodels)
    // Python equivalent: smf.ols('fte ~ tratado + t + effect', data=df).fit()
    println!("{:=^78}", " Example 1: OLS with Formula ");
    let formula = Formula::parse("fte ~ tratado + t + effect")?;
    println!("Formula: {} ~ {}", formula.dependent, formula.independents.join(" + "));
    println!("Intercept: {}\n", formula.intercept);

    let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust)?;
    println!("{}", result);

    // Example 2: OLS with Robust Standard Errors (HC1)
    // Python equivalent: smf.ols('fte ~ tratado + t + effect', data=df).fit(cov_type='HC1')
    println!("\n{:=^78}", " Example 2: OLS with Robust SE (HC1) ");
    let result_robust = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;
    println!("{}", result_robust);

    // Example 3: OLS without Intercept
    // Python equivalent: smf.ols('fte ~ tratado + t + effect - 1', data=df).fit()
    println!("\n{:=^78}", " Example 3: OLS without Intercept ");
    let formula_no_int = Formula::parse("fte ~ tratado + t + effect - 1")?;
    println!("Formula: {} ~ {} (no intercept)\n",
        formula_no_int.dependent,
        formula_no_int.independents.join(" + "));

    let result_no_int = OLS::from_formula(&formula_no_int, &df, CovarianceType::HC1)?;
    println!("{}", result_no_int);

    // Example 4: WLS (Weighted Least Squares) with formula
    // Python equivalent: smf.wls('fte ~ tratado + t + effect', data=df, weights=weights).fit()
    println!("\n{:=^78}", " Example 4: WLS with Formula ");

    // Create some weights (in practice, these would be based on variance estimates)
    let weights = Array1::from(vec![1.0; n]);

    let result_wls = FGLS::wls_from_formula(&formula, &df, &weights)?;
    println!("{}", result_wls);

    // Example 5: Cochrane-Orcutt for autocorrelation
    println!("\n{:=^78}", " Example 5: Cochrane-Orcutt (AR1) ");
    let result_co = FGLS::cochrane_orcutt_from_formula(&formula, &df)?;
    println!("{}", result_co);

    println!("\n{:=^78}", " Summary ");
    println!("Formula API successfully demonstrated!");
    println!("\nKey features:");
    println!("  - R/Python-style formula syntax: 'y ~ x1 + x2 + x3'");
    println!("  - Support for intercept control: '- 1' or '0 +'");
    println!("  - DataFrame structure for tabular data");
    println!("  - Integration with OLS, WLS, and other estimators");
    println!("  - Robust standard errors (HC1, Newey-West)");
    println!("\n{:=^78}", "");

    Ok(())
}
