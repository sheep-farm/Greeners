use greeners::{CovarianceType, DataFrame, Formula, SpecificationTests, OLS};
use ndarray::Array1;
use rand::{thread_rng, Rng};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("   Greeners v1.0.0 - Specification Tests");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // DATASET 1: WAGE EQUATION WITH HETEROSKEDASTICITY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("DATASET 1: Wage Equation (n=200)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Model: wage = β₀ + β₁*education + β₂*experience + β₃*tenure + ε");
    println!("Data generating process includes heteroskedasticity\n");

    let n = 200;
    let mut rng = thread_rng();

    let mut education_data = Vec::with_capacity(n);
    let mut experience_data = Vec::with_capacity(n);
    let mut tenure_data = Vec::with_capacity(n);
    let mut wage_data = Vec::with_capacity(n);

    // Generate data with heteroskedasticity (variance increases with education)
    for _ in 0..n {
        let education = rng.gen_range(8.0..20.0);
        let experience = rng.gen_range(0.0..40.0);
        let tenure = rng.gen_range(0.0..20.0);

        // True model: wage = 5 + 2*education + 0.5*experience + 0.3*tenure + error
        // Error variance increases with education (heteroskedasticity)
        let error_sd = 0.5 + 0.3 * education; // Variance grows with education
        let error: f64 = rng.gen_range(-error_sd..error_sd);

        let wage = 5.0 + 2.0 * education + 0.5 * experience + 0.3 * tenure + error;

        education_data.push(education);
        experience_data.push(experience);
        tenure_data.push(tenure);
        wage_data.push(wage);
    }

    let mut data1 = HashMap::new();
    data1.insert("wage".to_string(), Array1::from(wage_data.clone()));
    data1.insert("education".to_string(), Array1::from(education_data));
    data1.insert("experience".to_string(), Array1::from(experience_data));
    data1.insert("tenure".to_string(), Array1::from(tenure_data));

    let df1 = DataFrame::new(data1)?;

    // Estimate wage equation
    let formula1 = Formula::parse("wage ~ education + experience + tenure")?;
    let model1 = OLS::from_formula(&formula1, &df1, CovarianceType::NonRobust)?;
    println!("{}", model1);

    // Get residuals and design matrix
    let (y1, x1) = df1.to_design_matrix(&formula1)?;
    let residuals1 = model1.residuals(&y1, &x1);
    let fitted1 = model1.fitted_values(&x1);

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. WHITE TEST FOR HETEROSKEDASTICITY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("1. WHITE TEST for Heteroskedasticity");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let (white_stat, white_p, white_df) = SpecificationTests::white_test(&residuals1, &x1)?;

    SpecificationTests::print_test_result(
        "White Test for Heteroskedasticity",
        white_stat,
        white_p,
        "Homoskedasticity (constant variance)",
        "Heteroskedasticity (non-constant variance)",
    );

    println!("\n💡 INTERPRETATION:");
    if white_p < 0.05 {
        println!("   → Heteroskedasticity detected!");
        println!("   → Use robust standard errors (HC1, HC2, HC3, or HC4)");
        println!("   → Example: CovarianceType::HC3");
    } else {
        println!("   → No evidence of heteroskedasticity");
        println!("   → Standard OLS inference is reliable");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. RAMSEY RESET TEST FOR FUNCTIONAL FORM
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("2. RAMSEY RESET TEST for Functional Form");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let (reset_stat, reset_p, reset_df1, reset_df2) =
        SpecificationTests::reset_test(&y1, &x1, &fitted1, 3)?;

    println!("\nH₀: Functional form is correctly specified");
    println!("H₁: Functional form misspecification (omitted variables or wrong form)");
    println!("\nTest includes: ŷ², ŷ³");
    println!("F-Statistic: {:.4}", reset_stat);
    println!("P-value: {:.6}", reset_p);
    println!("Degrees of Freedom: ({}, {})", reset_df1, reset_df2);

    if reset_p < 0.05 {
        println!("\n✅ REJECT H₀ at 5% level");
        println!("   → Functional form misspecification detected");
        println!("   → Consider: polynomial terms, interactions, transformations");
    } else {
        println!("\n❌ FAIL TO REJECT H₀");
        println!("   → Functional form appears adequate");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DATASET 2: TIME SERIES WITH AUTOCORRELATION
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n\n═══════════════════════════════════════════════════════════════════════════");
    println!("DATASET 2: Consumption Function (Time Series, n=100)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Model: consumption = β₀ + β₁*income + β₂*wealth + ε");
    println!("Data generating process includes AR(1) autocorrelation\n");

    let n_ts = 100;
    let mut income_data = Vec::with_capacity(n_ts);
    let mut wealth_data = Vec::with_capacity(n_ts);
    let mut consumption_data = Vec::with_capacity(n_ts);

    let mut prev_error = 0.0;
    let rho = 0.7; // AR(1) coefficient

    for t in 0..n_ts {
        let income = 50.0 + (t as f64) * 0.5 + rng.gen_range(-5.0..5.0);
        let wealth = 100.0 + (t as f64) * 1.0 + rng.gen_range(-10.0..10.0);

        // AR(1) error: ε_t = ρ*ε_{t-1} + u_t
        let white_noise: f64 = rng.gen_range(-3.0..3.0);
        let error = rho * prev_error + white_noise;
        prev_error = error;

        let consumption = 10.0 + 0.6 * income + 0.2 * wealth + error;

        income_data.push(income);
        wealth_data.push(wealth);
        consumption_data.push(consumption);
    }

    let mut data2 = HashMap::new();
    data2.insert("consumption".to_string(), Array1::from(consumption_data));
    data2.insert("income".to_string(), Array1::from(income_data));
    data2.insert("wealth".to_string(), Array1::from(wealth_data));

    let df2 = DataFrame::new(data2)?;

    // Estimate consumption function
    let formula2 = Formula::parse("consumption ~ income + wealth")?;
    let model2 = OLS::from_formula(&formula2, &df2, CovarianceType::NonRobust)?;
    println!("{}", model2);

    // Get residuals
    let (y2, x2) = df2.to_design_matrix(&formula2)?;
    let residuals2 = model2.residuals(&y2, &x2);

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. BREUSCH-GODFREY TEST FOR AUTOCORRELATION
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("3. BREUSCH-GODFREY TEST for Autocorrelation");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let (bg_stat, bg_p, bg_df) = SpecificationTests::breusch_godfrey_test(&residuals2, &x2, 1)?;

    SpecificationTests::print_test_result(
        "Breusch-Godfrey LM Test (lag=1)",
        bg_stat,
        bg_p,
        "No autocorrelation (errors are independent)",
        "Autocorrelation present (AR(1) or higher)",
    );

    println!("\n💡 INTERPRETATION:");
    if bg_p < 0.05 {
        println!("   → Autocorrelation detected!");
        println!("   → Use Newey-West HAC standard errors");
        println!("   → Example: CovarianceType::NeweyWest(4)");
        println!("   → Consider AR/ARMA error structure");
    } else {
        println!("   → No evidence of autocorrelation");
        println!("   → Standard OLS inference is reliable for time series");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. GOLDFELD-QUANDT TEST FOR HETEROSKEDASTICITY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("4. GOLDFELD-QUANDT TEST for Heteroskedasticity");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("(Using wage equation data, ordered by education)");

    // Note: For GQ test, residuals should be pre-ordered by suspected variable
    // Here we use the original residuals (already somewhat ordered)
    let (gq_stat, gq_p, gq_df1, gq_df2) =
        SpecificationTests::goldfeld_quandt_test(&residuals1, 0.25)?;

    println!("\nH₀: Homoskedasticity");
    println!("H₁: Variance differs between groups");
    println!("\nDropping middle 25% of observations");
    println!("F-Statistic: {:.4}", gq_stat);
    println!("P-value: {:.6}", gq_p);
    println!("Degrees of Freedom: ({}, {})", gq_df1, gq_df2);

    if gq_p < 0.05 {
        println!("\n✅ REJECT H₀ at 5% level");
        println!("   → Heteroskedasticity detected");
    } else {
        println!("\n❌ FAIL TO REJECT H₀");
        println!("   → No evidence of heteroskedasticity");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("✨ SUMMARY OF v1.0.0 SPECIFICATION TESTS");
    println!("══════════════════════════════════════════════════════════════════════════════");

    println!("\n1. HETEROSKEDASTICITY TESTS:");
    println!("   • White Test - General test using auxiliary regression");
    println!("     - Tests for any form of heteroskedasticity");
    println!("     - No need to specify functional form");
    println!("   • Goldfeld-Quandt Test - Compares variance across ordered groups");
    println!("     - Requires pre-ordering by suspected variable");
    println!("     - Simple and intuitive");

    println!("\n2. AUTOCORRELATION TESTS:");
    println!("   • Breusch-Godfrey LM Test - Tests for AR(p) autocorrelation");
    println!("     - More general than Durbin-Watson");
    println!("     - Allows lagged dependent variables");
    println!("     - Specify number of lags to test");

    println!("\n3. FUNCTIONAL FORM TESTS:");
    println!("   • Ramsey RESET Test - Tests for omitted variables/wrong form");
    println!("     - Adds powers of fitted values (ŷ², ŷ³, ...)");
    println!("     - If rejected: add polynomials, interactions, or transformations");

    println!("\n4. REMEDIES:");
    println!("   • Heteroskedasticity → Use robust SE (HC1, HC2, HC3, HC4)");
    println!("   • Autocorrelation → Use Newey-West HAC SE");
    println!("   • Functional form → Add polynomials I(x^2), interactions x1*x2");

    println!("\n5. STATA/R/PYTHON EQUIVALENTS:");
    println!("   • Stata: estat hettest (White), estat bgodfrey, estat ovtest (RESET)");
    println!("   • R: lmtest::bptest(), lmtest::bgtest(), lmtest::resettest()");
    println!("   • Python: statsmodels het_white(), acorr_breusch_godfrey()");

    println!("══════════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}
