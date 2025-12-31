use greeners::{CovarianceType, InferenceType, OLS};
use ndarray::Array1;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Greeners: InferenceType Example (t vs z distribution) ===\n");

    // Create sample data: y = 2 + 3*x + noise
    let x_data: Vec<f64> = (1..=30).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 + 3.0 * x + (x * 0.1)).collect();

    // Build design matrix with intercept
    let n = x_data.len();
    let mut x_matrix = Array2::zeros((n, 2));
    for i in 0..n {
        x_matrix[[i, 0]] = 1.0; // intercept
        x_matrix[[i, 1]] = x_data[i];
    }

    let y = Array1::from(y_data);

    // Fit OLS model with default inference type (Student's t)
    println!("1. Fitting OLS with default inference type (Student's t)...\n");
    let result_t = OLS::fit(&y, &x_matrix, CovarianceType::NonRobust)?;

    println!("{}", result_t);
    println!("\n{}", "=".repeat(78));

    // Switch to Normal distribution (z-statistics)
    println!("\n2. Switching to Normal distribution (z-statistics)...\n");
    let result_z = result_t.clone().with_inference(InferenceType::Normal)?;

    println!("{}", result_z);
    println!("\n{}", "=".repeat(78));

    // Compare the results
    println!("\n3. Comparison of t vs z inference:\n");
    println!("{:-^78}", "");
    println!(
        "{:<20} | {:>12} | {:>12} | {:>12}",
        "Metric", "t-distribution", "z-distribution", "Difference"
    );
    println!("{:-^78}", "");

    for i in 0..2 {
        let var_name = if i == 0 { "Intercept" } else { "x" };

        println!(
            "{:<20} | {:>12.6} | {:>12.6} | {:>12.6}",
            format!("{} coef", var_name),
            result_t.params[i],
            result_z.params[i],
            result_t.params[i] - result_z.params[i]
        );

        println!(
            "{:<20} | {:>12.6} | {:>12.6} | {:>12.6}",
            format!("{} std err", var_name),
            result_t.std_errors[i],
            result_z.std_errors[i],
            result_t.std_errors[i] - result_z.std_errors[i]
        );

        println!(
            "{:<20} | {:>12.6} | {:>12.6} | {:>12.6}",
            format!("{} p-value", var_name),
            result_t.p_values[i],
            result_z.p_values[i],
            result_t.p_values[i] - result_z.p_values[i]
        );

        println!(
            "{:<20} | {:>12.6} | {:>12.6} | {:>12.6}",
            format!("{} CI lower", var_name),
            result_t.conf_lower[i],
            result_z.conf_lower[i],
            result_t.conf_lower[i] - result_z.conf_lower[i]
        );

        println!(
            "{:<20} | {:>12.6} | {:>12.6} | {:>12.6}",
            format!("{} CI upper", var_name),
            result_t.conf_upper[i],
            result_z.conf_upper[i],
            result_t.conf_upper[i] - result_z.conf_upper[i]
        );

        println!("{:-^78}", "");
    }

    println!("\nKey Observations:");
    println!("• Coefficients and standard errors are IDENTICAL (as expected)");
    println!("• P-values differ slightly: t is more conservative (larger p-values)");
    println!(
        "• Confidence intervals differ: t has wider intervals (more conservative) with df={}",
        result_t.df_resid
    );
    println!("• For large samples (df > 30), differences become negligible");
    println!("• Student's t → Normal as df → ∞");

    println!("\nWhen to use each:");
    println!("• Student's t (default): Small/medium samples, exact inference");
    println!("• Normal (z): Large samples (n > 1000), compatibility with statsmodels");

    Ok(())
}
