use greeners::{CovarianceType, DataFrame, Formula, FGLS};
use ndarray::Array1;
use rand::prelude::*;
use statrs::distribution::Normal;
use std::collections::HashMap;

/// Example demonstrating Difference-in-Differences using formula syntax
/// Similar to: smf.wls('fte ~ tratado + t + effect', data=df).fit(cov_type='HC1')
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_per_group = 500; // 500 obs per group (2000 total)
    let total_n = n_per_group * 4;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 2.0).unwrap();

    // Universe Parameters
    let baseline = 10.0;
    let group_diff = 2.0;
    let time_trend = 1.5;
    let att_real = 5.0; // Real treatment effect

    let mut fte_vec = Vec::with_capacity(total_n);
    let mut tratado_vec = Vec::with_capacity(total_n);
    let mut t_vec = Vec::with_capacity(total_n);

    // Generate 4 groups: Control Pre/Post, Treated Pre/Post
    for _ in 0..n_per_group {
        // Control Pre (0, 0)
        fte_vec.push(baseline + normal.sample(&mut rng));
        tratado_vec.push(0.0);
        t_vec.push(0.0);

        // Control Post (0, 1)
        fte_vec.push(baseline + time_trend + normal.sample(&mut rng));
        tratado_vec.push(0.0);
        t_vec.push(1.0);

        // Treated Pre (1, 0)
        fte_vec.push(baseline + group_diff + normal.sample(&mut rng));
        tratado_vec.push(1.0);
        t_vec.push(0.0);

        // Treated Post (1, 1)
        fte_vec.push(
            baseline + group_diff + time_trend + att_real + normal.sample(&mut rng)
        );
        tratado_vec.push(1.0);
        t_vec.push(1.0);
    }

    // Calculate interaction effect (this is the DiD estimator)
    let effect_vec: Vec<f64> = (0..total_n)
        .map(|i| tratado_vec[i] * t_vec[i])
        .collect();

    // Create DataFrame
    let mut data = HashMap::new();
    data.insert("fte".to_string(), Array1::from(fte_vec));
    data.insert("tratado".to_string(), Array1::from(tratado_vec));
    data.insert("t".to_string(), Array1::from(t_vec));
    data.insert("effect".to_string(), Array1::from(effect_vec));

    let df = DataFrame::new(data)?;

    println!("\n{:=^78}", " Difference-in-Differences with Formula API ");
    println!("\nReal Parameters:");
    println!("  Time Trend (Counterfactual): +{:.1}", time_trend);
    println!("  Group Difference (Bias):     +{:.1}", group_diff);
    println!("  ATT (Treatment Effect):      +{:.1}", att_real);
    println!("\nDataset:");
    println!("  Total observations: {}", df.n_rows());
    println!("  Variables: {:?}\n", df.column_names());

    // Python equivalent:
    // CK1 = smf.wls('fte ~ tratado + t + effect', data=df).fit(cov_type='HC1')
    println!("{:=^78}", " Formula-based Estimation ");

    // Create formula (same syntax as Python/R)
    let formula = Formula::parse("fte ~ tratado + t + effect")?;

    println!("Formula: {} ~ {}",
        formula.dependent,
        formula.independents.join(" + "));
    println!("Intercept: {}\n", formula.intercept);

    // Create weights (all equal to 1 for WLS = OLS)
    let weights = Array1::from(vec![1.0; df.n_rows()]);

    // Fit the model with robust standard errors (HC1)
    // Note: FGLS::wls with weights=1 is equivalent to OLS
    // We use OLS::from_formula for this, but demonstrating WLS syntax
    let result = FGLS::wls_from_formula(&formula, &df, &weights)?;

    println!("Estimation Results:\n");
    println!("{}", result);

    println!("\nInterpretation:");
    println!("  Intercept (x0):  Baseline for control group at t=0");
    println!("  Tratado (x1):    Group fixed effect (selection bias)");
    println!("  t (x2):          Time fixed effect (common trend)");
    println!("  Effect (x3):     ** ATT (Difference-in-Differences Estimator) **");
    println!("\nThe coefficient on 'effect' (x3) should be close to {:.1}", att_real);

    // Also demonstrate with robust standard errors using OLS
    println!("\n{:=^78}", " Same Model with Robust SE (HC1) ");

    use greeners::OLS;
    let result_robust = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;
    println!("{}", result_robust);

    println!("\n{:=^78}", "");
    println!("\nKey Advantages of Formula API:");
    println!("  1. Clean, readable syntax like R/Python");
    println!("  2. Easy to specify models without manual matrix construction");
    println!("  3. Integrates with all estimators (OLS, WLS, etc.)");
    println!("  4. Type-safe and fast (Rust performance)");

    Ok(())
}
