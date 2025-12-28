use greeners::{CovarianceType, DataFrame, Formula, OLS};
use ndarray::Array1;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Clustered Standard Errors Example ===\n");
    println!("Demonstrating the importance of clustering in panel data\n");

    // Simulated panel data: 5 firms, 10 time periods each = 50 observations
    // We expect errors to be correlated within firms

    let mut data = HashMap::new();

    // Dependent variable (e.g., profits)
    let y_data = vec![
        // Firm 1 (10 obs)
        12.5, 13.2, 14.1, 13.8, 14.5, 15.2, 14.9, 15.5, 16.1, 16.8, // Firm 2 (10 obs)
        10.2, 10.8, 11.5, 11.2, 12.1, 12.8, 12.5, 13.2, 13.9, 14.5, // Firm 3 (10 obs)
        15.5, 16.2, 16.9, 16.5, 17.2, 17.9, 17.5, 18.2, 18.9, 19.5, // Firm 4 (10 obs)
        8.5, 9.1, 9.8, 9.4, 10.1, 10.8, 10.4, 11.1, 11.8, 12.4, // Firm 5 (10 obs)
        11.2, 11.9, 12.6, 12.2, 12.9, 13.6, 13.2, 13.9, 14.6, 15.2,
    ];

    // Independent variable 1 (e.g., advertising expenditure)
    let x1_data = vec![
        // Firm 1
        2.0, 2.1, 2.3, 2.2, 2.4, 2.6, 2.5, 2.7, 2.9, 3.0, // Firm 2
        1.5, 1.6, 1.8, 1.7, 1.9, 2.1, 2.0, 2.2, 2.4, 2.5, // Firm 3
        2.5, 2.6, 2.8, 2.7, 2.9, 3.1, 3.0, 3.2, 3.4, 3.5, // Firm 4
        1.0, 1.1, 1.3, 1.2, 1.4, 1.6, 1.5, 1.7, 1.9, 2.0, // Firm 5
        1.8, 1.9, 2.1, 2.0, 2.2, 2.4, 2.3, 2.5, 2.7, 2.8,
    ];

    // Independent variable 2 (e.g., R&D spending)
    let x2_data = vec![
        // Firm 1
        1.5, 1.6, 1.7, 1.6, 1.8, 1.9, 1.8, 2.0, 2.1, 2.2, // Firm 2
        1.2, 1.3, 1.4, 1.3, 1.5, 1.6, 1.5, 1.7, 1.8, 1.9, // Firm 3
        2.0, 2.1, 2.2, 2.1, 2.3, 2.4, 2.3, 2.5, 2.6, 2.7, // Firm 4
        0.8, 0.9, 1.0, 0.9, 1.1, 1.2, 1.1, 1.3, 1.4, 1.5, // Firm 5
        1.4, 1.5, 1.6, 1.5, 1.7, 1.8, 1.7, 1.9, 2.0, 2.1,
    ];

    data.insert("profit".to_string(), Array1::from(y_data));
    data.insert("advertising".to_string(), Array1::from(x1_data));
    data.insert("rd_spending".to_string(), Array1::from(x2_data));

    let df = DataFrame::new(data)?;

    // Define cluster IDs (firm identifiers)
    // 0,0,0,...,0 (10 times), 1,1,1,...,1 (10 times), etc.
    let cluster_ids: Vec<usize> = (0..5)
        .flat_map(|firm_id| std::iter::repeat_n(firm_id, 10))
        .collect();

    // Specify the model
    let formula = Formula::parse("profit ~ advertising + rd_spending")?;

    println!("Model: profit ~ advertising + rd_spending");
    println!("Panel structure: 5 firms × 10 time periods = 50 observations\n");

    // 1. Standard OLS (incorrect for panel data!)
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("1. STANDARD OLS (INCORRECT - assumes independence)");
    println!("══════════════════════════════════════════════════════════════════════════════");
    let ols_standard = OLS::from_formula(&formula, &df, CovarianceType::NonRobust)?;
    println!("{}", ols_standard);

    // 2. Robust Standard Errors (HC1) - only corrects for heteroskedasticity
    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("2. ROBUST SE (HC1) - Only fixes heteroskedasticity");
    println!("══════════════════════════════════════════════════════════════════════════════");
    let ols_robust = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;
    println!("{}", ols_robust);

    // 3. Clustered Standard Errors (CORRECT for panel data!)
    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("3. CLUSTERED SE (CORRECT - accounts for within-firm correlation)");
    println!("══════════════════════════════════════════════════════════════════════════════");
    let ols_clustered = OLS::from_formula(&formula, &df, CovarianceType::Clustered(cluster_ids))?;
    println!("{}", ols_clustered);

    // 4. Comparison Table
    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("COMPARISON: How Standard Errors Change with Different Estimators");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!(
        "{:<15} | {:>12} | {:>12} | {:>12}",
        "Variable", "Non-Robust", "HC1", "Clustered"
    );
    println!("{:-<15}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "");

    for i in 0..3 {
        let var_name = match i {
            0 => "Intercept",
            1 => "Advertising",
            2 => "R&D Spending",
            _ => "",
        };

        println!(
            "{:<15} | {:>12.6} | {:>12.6} | {:>12.6}",
            var_name,
            ols_standard.std_errors[i],
            ols_robust.std_errors[i],
            ols_clustered.std_errors[i]
        );
    }

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("KEY INSIGHTS:");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("✓ Coefficients are IDENTICAL across all methods (β is consistent)");
    println!("✓ Standard errors DIFFER - clustered SE are typically LARGER");
    println!("✓ Clustering corrects for within-group correlation (e.g., firm-specific shocks)");
    println!("✓ Ignoring clustering → overstated significance → false discoveries!");
    println!("\n⚠️  CRITICAL: Always use clustered SE when:");
    println!("   - Panel data (repeated observations per entity)");
    println!("   - Hierarchical data (students in schools, patients in hospitals)");
    println!("   - Experimental data with treatment clusters");
    println!("   - Geographic clustering (observations in regions/countries)");

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("When to use each covariance estimator:");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("• NonRobust:  Classical OLS (homoskedastic errors assumed)");
    println!("• HC1:        Heteroskedasticity only (White's robust SE)");
    println!("• NeweyWest:  Heteroskedasticity + Autocorrelation (time series)");
    println!("• Clustered:  Within-cluster correlation (panel/grouped data)");

    Ok(())
}
