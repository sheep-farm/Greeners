use greeners::{CovarianceType, Diagnostics, OLS};
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Regression Diagnostics Example ===\n");

    // Create sample data with some issues:
    // - Multicollinearity (x2 highly correlated with x1)
    // - One influential outlier
    let y = Array1::from(vec![
        2.5, 3.2, 4.1, 4.8, 5.5, 6.2, 6.9, 7.5, 8.2, 15.0, // Last point is outlier
    ]);

    let x = Array2::from_shape_vec(
        (10, 3),
        vec![
            // Intercept, x1, x2
            1.0, 1.0, 1.1, // x2 ≈ x1 (multicollinearity)
            1.0, 2.0, 2.2, 1.0, 3.0, 3.1, 1.0, 4.0, 4.2, 1.0, 5.0, 5.1, 1.0, 6.0, 6.3, 1.0, 7.0,
            7.2, 1.0, 8.0, 8.1, 1.0, 9.0, 9.2, 1.0, 10.0, 10.5, // Outlier with high leverage
        ],
    )?;

    // Fit OLS model
    let result = OLS::fit(&y, &x, CovarianceType::HC1)?;

    println!("=== REGRESSION RESULTS ===");
    println!("{}", result);

    // Get residuals
    let y_hat = x.dot(&result.params);
    let residuals = &y - &y_hat;

    // Calculate MSE
    let mse = result.sigma.powi(2);

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("1. MULTICOLLINEARITY DIAGNOSTICS");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // Variance Inflation Factor (VIF)
    let vif_values = Diagnostics::vif(&x)?;
    println!("Variance Inflation Factors (VIF):");
    println!("{:-<60}", "");
    println!("{:<20} | {:>15} | {:<20}", "Variable", "VIF", "Assessment");
    println!("{:-<60}", "");

    let var_names = vec!["Intercept", "x1", "x2"];
    for (i, &vif) in vif_values.iter().enumerate() {
        let assessment = if vif.is_nan() {
            "Undefined (constant)"
        } else if vif < 5.0 {
            "✓ Good"
        } else if vif < 10.0 {
            "⚠ Moderate"
        } else {
            "✗ High (problematic)"
        };

        println!("{:<20} | {:>15.2} | {:<20}", var_names[i], vif, assessment);
    }

    // Condition Number
    let cond_num = Diagnostics::condition_number(&x)?;
    println!("\nCondition Number: {:.2}", cond_num);
    println!(
        "Assessment: {}",
        if cond_num < 10.0 {
            "✓ No multicollinearity"
        } else if cond_num < 30.0 {
            "⚠ Moderate multicollinearity"
        } else if cond_num < 100.0 {
            "✗ Strong multicollinearity"
        } else {
            "✗✗ Severe multicollinearity (critical!)"
        }
    );

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("2. INFLUENTIAL OBSERVATIONS DIAGNOSTICS");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // Leverage
    let leverage_values = Diagnostics::leverage(&x)?;
    let n = y.len();
    let k = x.ncols();
    let avg_leverage = (k as f64) / (n as f64);
    let high_leverage_threshold = 2.0 * avg_leverage;

    println!("Leverage Statistics:");
    println!("{:-<80}", "");
    println!(
        "{:<8} | {:>12} | {:>12} | {:>12} | {:<20}",
        "Obs", "Residual", "Leverage", "Cook's D", "Assessment"
    );
    println!("{:-<80}", "");

    // Cook's Distance
    let cooks_d = Diagnostics::cooks_distance(&residuals, &x, mse)?;

    for i in 0..n {
        let mut flags = Vec::new();

        if leverage_values[i] > high_leverage_threshold {
            flags.push("High Leverage");
        }
        if cooks_d[i] > 1.0 {
            flags.push("✗ Very Influential");
        } else if cooks_d[i] > 4.0 / (n as f64) {
            flags.push("⚠ Influential");
        }
        if residuals[i].abs() > 2.0 * result.sigma {
            flags.push("Large Residual");
        }

        let assessment = if flags.is_empty() {
            "✓ Normal".to_string()
        } else {
            flags.join(", ")
        };

        println!(
            "{:<8} | {:>12.4} | {:>12.4} | {:>12.4} | {:<20}",
            i + 1,
            residuals[i],
            leverage_values[i],
            cooks_d[i],
            assessment
        );
    }

    println!("\nThresholds:");
    println!("  • Average Leverage (k/n):     {:.4}", avg_leverage);
    println!(
        "  • High Leverage (2k/n):       {:.4}",
        high_leverage_threshold
    );
    println!("  • Cook's D Influential (4/n): {:.4}", 4.0 / (n as f64));
    println!("  • Cook's D Critical:          1.0");

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("3. RESIDUAL DIAGNOSTICS");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // Jarque-Bera Test for Normality
    let (jb_stat, jb_pvalue) = Diagnostics::jarque_bera(&residuals)?;
    println!("Jarque-Bera Test for Normality:");
    println!("  Statistic: {:.4}", jb_stat);
    println!("  P-value:   {:.4}", jb_pvalue);
    println!(
        "  Result:    {}",
        if jb_pvalue > 0.05 {
            "✓ Residuals appear normally distributed (p > 0.05)"
        } else {
            "✗ Residuals deviate from normality (p < 0.05)"
        }
    );

    // Breusch-Pagan Test for Heteroskedasticity
    let (bp_stat, bp_pvalue) = Diagnostics::breusch_pagan(&residuals, &x)?;
    println!("\nBreusch-Pagan Test for Heteroskedasticity:");
    println!("  LM Statistic: {:.4}", bp_stat);
    println!("  P-value:      {:.4}", bp_pvalue);
    println!(
        "  Result:       {}",
        if bp_pvalue > 0.05 {
            "✓ Homoskedasticity (constant variance, p > 0.05)"
        } else {
            "✗ Heteroskedasticity detected (p < 0.05) - use robust SE!"
        }
    );

    // Durbin-Watson Test for Autocorrelation
    let dw_stat = Diagnostics::durbin_watson(&residuals);
    println!("\nDurbin-Watson Test for Autocorrelation:");
    println!("  Statistic: {:.4}", dw_stat);
    println!(
        "  Result:    {}",
        if (dw_stat - 2.0).abs() < 0.5 {
            "✓ No significant autocorrelation (≈ 2.0)"
        } else if dw_stat < 2.0 {
            "⚠ Positive autocorrelation detected (< 2.0)"
        } else {
            "⚠ Negative autocorrelation detected (> 2.0)"
        }
    );

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("KEY RECOMMENDATIONS:");
    println!("══════════════════════════════════════════════════════════════════════════════");

    let mut recommendations = Vec::new();

    if cond_num > 30.0 || vif_values.iter().any(|&v| !v.is_nan() && v > 10.0) {
        recommendations.push("✗ MULTICOLLINEARITY: Remove or combine highly correlated predictors");
    }

    if cooks_d.iter().any(|&d| d > 1.0) {
        recommendations.push("✗ INFLUENTIAL POINTS: Investigate observations with Cook's D > 1.0");
    }

    if bp_pvalue < 0.05 {
        recommendations.push("⚠ HETEROSKEDASTICITY: Use robust standard errors (HC1, HC2, etc.)");
    }

    if jb_pvalue < 0.05 {
        recommendations
            .push("⚠ NON-NORMAL RESIDUALS: Check for outliers or model misspecification");
    }

    if recommendations.is_empty() {
        println!("✓ Model passes all diagnostic checks!");
    } else {
        for rec in recommendations {
            println!("{}", rec);
        }
    }

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("DIAGNOSTIC WORKFLOW:");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("1. Check multicollinearity (VIF, Condition Number)");
    println!("2. Identify influential observations (Leverage, Cook's D)");
    println!("3. Test assumptions (Normality, Homoskedasticity, No autocorrelation)");
    println!("4. Use appropriate standard errors (Robust, Clustered, HAC)");
    println!("5. Consider model modifications if diagnostics fail");

    Ok(())
}
