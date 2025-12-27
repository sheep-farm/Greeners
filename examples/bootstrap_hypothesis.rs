use greeners::{Bootstrap, CovarianceType, DataFrame, Formula, HypothesisTest, OLS};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("   Greeners v0.8.0 - Bootstrap & Hypothesis Testing");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // CREATE DATASET
    // ═══════════════════════════════════════════════════════════════════════════

    println!("Dataset: Wage Determination (N = 30)");
    println!("Variables:");
    println!("  • wage: Hourly wage");
    println!("  • education: Years of education");
    println!("  • experience: Years of experience");
    println!("  • female: Gender dummy (1=female, 0=male)\n");

    let mut data = HashMap::new();

    // Generate realistic wage data
    let education = vec![
        12.0, 16.0, 14.0, 18.0, 12.0, 16.0, 14.0, 20.0, 12.0, 14.0, 16.0, 18.0, 12.0, 14.0, 16.0,
        18.0, 20.0, 12.0, 14.0, 16.0, 12.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 18.0, 16.0, 14.0,
    ];

    let experience = vec![
        2.0, 5.0, 3.0, 8.0, 1.0, 6.0, 4.0, 10.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 8.0, 12.0, 1.0,
        3.0, 5.0, 2.0, 4.0, 6.0, 9.0, 3.0, 7.0, 5.0, 10.0, 8.0, 6.0,
    ];

    let female = vec![
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    ];

    // Generate wages: wage = 5 + 2*education + 1*experience - 3*female + noise
    let wage: Vec<f64> = (0..30)
        .map(|i| {
            5.0 + 2.0 * education[i] + 1.0 * experience[i] - 3.0 * female[i]
                + (i as f64 % 5.0) * 0.8
        })
        .collect();

    data.insert("wage".to_string(), Array1::from(wage));
    data.insert("education".to_string(), Array1::from(education));
    data.insert("experience".to_string(), Array1::from(experience));
    data.insert("female".to_string(), Array1::from(female));

    let df = DataFrame::new(data)?;
    let formula = Formula::parse("wage ~ education + experience + female")?;

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. STANDARD OLS ESTIMATION
    // ═══════════════════════════════════════════════════════════════════════════

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("1. STANDARD OLS ESTIMATION (Asymptotic SE)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let result_hc3 = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;
    println!("{}", result_hc3);

    println!("\n📊 STANDARD ERRORS (HC3 - Robust):");
    for (i, se) in result_hc3.std_errors.iter().enumerate() {
        println!("   β{}: SE = {:.4}", i, se);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. BOOTSTRAP STANDARD ERRORS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("2. BOOTSTRAP STANDARD ERRORS (1000 replications)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let (y, x) = df.to_design_matrix(&formula)?;

    println!("\n⏳ Running 1000 bootstrap replications...");
    let boot_coefs = Bootstrap::pairs_bootstrap(&y, &x, 1000)?;

    let boot_se = Bootstrap::bootstrap_se(&boot_coefs);

    println!("\n✅ Bootstrap complete!");
    println!("\n📊 COMPARISON: Asymptotic vs Bootstrap SE:");
    println!("{:-^78}", "");
    println!(
        "{:<15} | {:>15} | {:>15} | {:>15}",
        "Coefficient", "HC3 SE", "Bootstrap SE", "Difference"
    );
    println!("{:-^78}", "");
    for i in 0..result_hc3.params.len() {
        println!(
            "{:<15} | {:>15.4} | {:>15.4} | {:>15.4}",
            format!("β{}", i),
            result_hc3.std_errors[i],
            boot_se[i],
            result_hc3.std_errors[i] - boot_se[i]
        );
    }
    println!("{:-^78}", "");

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. BOOTSTRAP CONFIDENCE INTERVALS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("3. BOOTSTRAP PERCENTILE CONFIDENCE INTERVALS (95%)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let (boot_lower, boot_upper) = Bootstrap::percentile_ci(&boot_coefs, 0.05);

    println!("\n📊 95% CONFIDENCE INTERVALS:");
    println!("{:-^78}", "");
    println!(
        "{:<15} | {:>12} | {:>18} | {:>18}",
        "Coefficient", "Estimate", "Lower (2.5%)", "Upper (97.5%)"
    );
    println!("{:-^78}", "");
    for i in 0..result_hc3.params.len() {
        println!(
            "{:<15} | {:>12.4} | {:>18.4} | {:>18.4}",
            format!("β{}", i),
            result_hc3.params[i],
            boot_lower[i],
            boot_upper[i]
        );
    }
    println!("{:-^78}", "");

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. WALD TEST - Joint Significance
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("4. WALD TEST - Joint Significance of All Slope Coefficients");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("\nH₀: β₁ = β₂ = β₃ = 0 (all slope coefficients are zero)");
    println!("H₁: At least one β ≠ 0\n");

    // Get covariance matrix from HC3 robust estimation
    let (y_test, x_test) = df.to_design_matrix(&formula)?;
    let result_for_test = OLS::fit(&y_test, &x_test, CovarianceType::HC3)?;

    // Need to compute covariance matrix (normally stored in OlsResult, but for demo we'll use SE)
    let mut cov_matrix = Array2::<f64>::zeros((4, 4));
    for i in 0..4 {
        cov_matrix[[i, i]] = result_for_test.std_errors[i].powi(2);
    }

    let (wald_stat, wald_p, df_wald) = HypothesisTest::joint_significance(
        &result_for_test.params,
        &cov_matrix,
        true, // Has intercept
    )?;

    println!("Wald Statistic: {:.4}", wald_stat);
    println!("Degrees of Freedom: {}", df_wald);
    println!("P-value: {:.6}", wald_p);

    if wald_p < 0.05 {
        println!("\n✅ REJECT H₀ at 5% level");
        println!("   → At least one coefficient is significantly different from zero");
    } else {
        println!("\n❌ FAIL TO REJECT H₀ at 5% level");
        println!("   → Cannot conclude that coefficients are significant");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. WALD TEST - Specific Linear Restriction
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("5. WALD TEST - Specific Linear Restriction");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("\nH₀: β(education) = β(experience) (equal returns)");
    println!("H₁: β(education) ≠ β(experience)\n");

    // Restriction: β₁ - β₂ = 0 → [0, 1, -1, 0] · β = 0
    let r = Array2::from_shape_vec((1, 4), vec![0.0, 1.0, -1.0, 0.0])?;
    let q = Array1::from(vec![0.0]);

    let (wald_stat2, wald_p2, df_wald2) =
        HypothesisTest::wald_test(&result_for_test.params, &cov_matrix, &r, &q)?;

    println!("Testing: β(education) - β(experience) = 0");
    println!(
        "Point estimate: β₁ - β₂ = {:.4}",
        result_for_test.params[1] - result_for_test.params[2]
    );
    println!("\nWald Statistic: {:.4}", wald_stat2);
    println!("Degrees of Freedom: {}", df_wald2);
    println!("P-value: {:.6}", wald_p2);

    if wald_p2 < 0.05 {
        println!("\n✅ REJECT H₀ at 5% level");
        println!("   → Returns to education and experience are significantly different");
    } else {
        println!("\n❌ FAIL TO REJECT H₀ at 5% level");
        println!("   → Cannot conclude that returns are different");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. F-TEST - Nested Models
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("6. F-TEST - Nested Model Comparison");
    println!("═══════════════════════════════════════════════════════════════════════════");

    // Full model: wage ~ education + experience + female
    let formula_full = Formula::parse("wage ~ education + experience + female")?;
    let result_full = OLS::from_formula(&formula_full, &df, CovarianceType::NonRobust)?;

    // Restricted model: wage ~ education (drop experience and female)
    let formula_restricted = Formula::parse("wage ~ education")?;
    let result_restricted = OLS::from_formula(&formula_restricted, &df, CovarianceType::NonRobust)?;

    println!("\nFull Model:       wage ~ education + experience + female");
    println!("Restricted Model: wage ~ education");
    println!("\nH₀: β(experience) = β(female) = 0");
    println!("H₁: At least one of them ≠ 0\n");

    // Calculate SSR for both models
    let (y_full, x_full) = df.to_design_matrix(&formula_full)?;
    let (y_rest, x_rest) = df.to_design_matrix(&formula_restricted)?;

    let fitted_full = x_full.dot(&result_full.params);
    let resid_full = &y_full - &fitted_full;
    let ssr_full = resid_full.dot(&resid_full);

    let fitted_rest = x_rest.dot(&result_restricted.params);
    let resid_rest = &y_rest - &fitted_rest;
    let ssr_restricted = resid_rest.dot(&resid_rest);

    let (f_stat, f_p, df_num, df_denom) = HypothesisTest::f_test_nested(
        ssr_restricted,
        ssr_full,
        y_full.len(),
        x_full.ncols(),
        x_rest.ncols(),
    )?;

    println!("SSR (Full):       {:.4}", ssr_full);
    println!("SSR (Restricted): {:.4}", ssr_restricted);
    println!("\nF-Statistic: {:.4}", f_stat);
    println!("DF: ({}, {})", df_num, df_denom);
    println!("P-value: {:.6}", f_p);

    if f_p < 0.05 {
        println!("\n✅ REJECT H₀ at 5% level");
        println!("   → Full model is significantly better than restricted model");
        println!("   → Experience and/or Female have significant explanatory power");
    } else {
        println!("\n❌ FAIL TO REJECT H₀ at 5% level");
        println!("   → Restricted model is adequate");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("✨ SUMMARY OF v0.8.0 FEATURES");
    println!("══════════════════════════════════════════════════════════════════════════════");

    println!("\n1. BOOTSTRAP METHODS:");
    println!("   • Pairs bootstrap with replacement");
    println!("   • Bootstrap standard errors");
    println!("   • Percentile confidence intervals");
    println!("   • Robust to non-normality and heteroscedasticity");

    println!("\n2. HYPOTHESIS TESTING:");
    println!("   • Wald test for linear restrictions (R·β = q)");
    println!("   • Joint significance tests");
    println!("   • F-test for nested models");
    println!("   • Flexible restriction matrices");

    println!("\n3. WHEN TO USE:");
    println!("   • Bootstrap: Small samples, non-normal errors, asymptotic skepticism");
    println!("   • Wald test: Testing multiple restrictions simultaneously");
    println!("   • F-test: Comparing nested OLS models");

    println!("\n4. STATA/R/PYTHON EQUIVALENTS:");
    println!("   • Stata: bootstrap, test, testparm");
    println!("   • R: boot package, waldtest(), anova()");
    println!("   • Python: statsmodels.tools.bootstrap, wald_test()");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}
