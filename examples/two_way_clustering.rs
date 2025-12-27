use greeners::{OLS, DataFrame, Formula, CovarianceType};
use ndarray::Array1;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("   Greeners v0.5.0 - Two-Way Clustered Standard Errors");
    println!("   Cameron-Gelbach-Miller (2011)");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // PANEL DATA: Firm Performance over Time
    // ═══════════════════════════════════════════════════════════════════════════

    println!("Dataset: Firm Performance (4 firms × 6 time periods = 24 observations)");
    println!("Variables:");
    println!("  • profit: Firm profit (dependent variable)");
    println!("  • investment: Capital investment");
    println!("  • market_share: Market share percentage");
    println!("  • firm_id: Firm identifier (0-3)");
    println!("  • time_id: Time period (0-5)\n");

    // Create panel dataset: 4 firms, 6 time periods
    let n_firms = 4;
    let n_periods = 6;
    let n_obs = n_firms * n_periods;

    let mut profit_data = Vec::with_capacity(n_obs);
    let mut investment_data = Vec::with_capacity(n_obs);
    let mut market_share_data = Vec::with_capacity(n_obs);
    let mut firm_ids = Vec::with_capacity(n_obs);
    let mut time_ids = Vec::with_capacity(n_obs);

    // Generate data: profit depends on investment and market share
    for firm in 0..n_firms {
        for period in 0..n_periods {
            let inv = 10.0 + (firm * period) as f64 * 2.0 + (period as f64 * 1.5);
            let mkt = 15.0 + (firm as f64 * 3.0) + (period as f64 * 0.5);
            let prof = 50.0 + inv * 1.5 + mkt * 2.0 + ((firm + period) as f64 * 0.8);

            profit_data.push(prof);
            investment_data.push(inv);
            market_share_data.push(mkt);
            firm_ids.push(firm);
            time_ids.push(period);
        }
    }

    let mut data = HashMap::new();
    data.insert("profit".to_string(), Array1::from(profit_data));
    data.insert("investment".to_string(), Array1::from(investment_data));
    data.insert("market_share".to_string(), Array1::from(market_share_data));

    let df = DataFrame::new(data)?;
    let formula = Formula::parse("profit ~ investment + market_share")?;

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Model: profit ~ investment + market_share");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. NON-ROBUST (WRONG for panel data!)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("1. NON-ROBUST Standard Errors (WRONG for panel data!)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    let result_nonrobust = OLS::from_formula(&formula, &df, CovarianceType::NonRobust)?;
    println!("{}", result_nonrobust);
    println!("\n⚠️  PROBLEM: Assumes no correlation within firms or over time");
    println!("    → Standard errors are DOWNWARD BIASED");
    println!("    → t-statistics are TOO LARGE");
    println!("    → p-values are TOO SMALL (Type I error!)");

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. ONE-WAY CLUSTERING BY FIRM
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n─────────────────────────────────────────────────────────────────────────────");
    println!("2. ONE-WAY Clustered SE (by firm)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    let result_firm = OLS::from_formula(&formula, &df, CovarianceType::Clustered(firm_ids.clone()))?;
    println!("{}", result_firm);
    println!("\n✅ ACCOUNTS FOR: Correlation within firms over time");
    println!("⚠️  IGNORES: Correlation across firms within time periods");

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. ONE-WAY CLUSTERING BY TIME
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n─────────────────────────────────────────────────────────────────────────────");
    println!("3. ONE-WAY Clustered SE (by time)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    let result_time = OLS::from_formula(&formula, &df, CovarianceType::Clustered(time_ids.clone()))?;
    println!("{}", result_time);
    println!("\n✅ ACCOUNTS FOR: Correlation across firms within time periods");
    println!("⚠️  IGNORES: Correlation within firms over time");

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. TWO-WAY CLUSTERING (RECOMMENDED)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n─────────────────────────────────────────────────────────────────────────────");
    println!("4. TWO-WAY Clustered SE (Cameron-Gelbach-Miller, 2011) - RECOMMENDED");
    println!("─────────────────────────────────────────────────────────────────────────────");
    let result_twoway = OLS::from_formula(
        &formula,
        &df,
        CovarianceType::ClusteredTwoWay(firm_ids.clone(), time_ids.clone())
    )?;
    println!("{}", result_twoway);
    println!("\n✅ ACCOUNTS FOR: BOTH within-firm AND within-time correlation");
    println!("✅ Formula: V = V_firm + V_time - V_intersection");
    println!("✅ Most robust for panel data with two-way dependence");

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPARISON TABLE
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("COMPARISON: Standard Errors Across Methods");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    println!("{:-^78}", "");
    println!("{:<20} | {:>12} | {:>12} | {:>12}", "Variable", "Coef", "Std Err", "t-stat");
    println!("{:-^78}", "");

    println!("\nNon-Robust (WRONG):");
    for i in 0..result_nonrobust.params.len() {
        println!("{:<20} | {:>12.4} | {:>12.4} | {:>12.3}",
            format!("x{}", i),
            result_nonrobust.params[i],
            result_nonrobust.std_errors[i],
            result_nonrobust.t_values[i]);
    }

    println!("\nOne-Way: Firm Clusters:");
    for i in 0..result_firm.params.len() {
        println!("{:<20} | {:>12.4} | {:>12.4} | {:>12.3}",
            format!("x{}", i),
            result_firm.params[i],
            result_firm.std_errors[i],
            result_firm.t_values[i]);
    }

    println!("\nOne-Way: Time Clusters:");
    for i in 0..result_time.params.len() {
        println!("{:<20} | {:>12.4} | {:>12.4} | {:>12.3}",
            format!("x{}", i),
            result_time.params[i],
            result_time.std_errors[i],
            result_time.t_values[i]);
    }

    println!("\nTwo-Way: Firm × Time (BEST):");
    for i in 0..result_twoway.params.len() {
        println!("{:<20} | {:>12.4} | {:>12.4} | {:>12.3}",
            format!("x{}", i),
            result_twoway.params[i],
            result_twoway.std_errors[i],
            result_twoway.t_values[i]);
    }
    println!("{:-^78}", "");

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("✨ WHEN TO USE TWO-WAY CLUSTERING");
    println!("══════════════════════════════════════════════════════════════════════════════");

    println!("\n1. PANEL DATA with BOTH dimensions:");
    println!("   • Firms/individuals over time");
    println!("   • Countries across years");
    println!("   • Schools with repeated measurements");

    println!("\n2. CORRELATION PATTERNS:");
    println!("   • Within-entity correlation (e.g., same firm over time)");
    println!("   • Within-time correlation (e.g., all firms in same period)");
    println!("   • Both simultaneously!");

    println!("\n3. FORMULA (Cameron-Gelbach-Miller, 2011):");
    println!("   V_2way = V₁ + V₂ - V₁₂");
    println!("   Where:");
    println!("     V₁ = one-way clustering by dimension 1 (e.g., firm)");
    println!("     V₂ = one-way clustering by dimension 2 (e.g., time)");
    println!("     V₁₂ = clustering by intersection (firm × time pairs)");

    println!("\n4. WHY IT MATTERS:");
    println!("   ⚠️  One-way clustering ONLY corrects for one source of correlation");
    println!("   ✅ Two-way clustering corrects for BOTH sources");
    println!("   ⚠️  Ignoring one dimension → standard errors TOO SMALL");
    println!("   ✅ Two-way clustering → conservative, robust inference");

    println!("\n5. STATA/R/PYTHON EQUIVALENTS:");
    println!("   • Stata: reghdfe y x, absorb(firm_id time_id) vce(cluster firm_id time_id)");
    println!("   • R: felm(y ~ x | firm_id + time_id | 0 | firm_id + time_id)");
    println!("   • Python: Not directly available in statsmodels (requires manual implementation)");

    println!("\n6. RULE OF THUMB:");
    println!("   • If panel data: ALWAYS use two-way clustering");
    println!("   • Better to be conservative with inference");
    println!("   • Cost: slightly wider confidence intervals");
    println!("   • Benefit: valid hypothesis tests!");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}
