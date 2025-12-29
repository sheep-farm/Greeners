use greeners::DataFrame;

fn main() {
    println!("=== Missing Data Handling - Essential for Real-World Data ===\n");

    // Create DataFrame with missing values (NaN)
    let df = DataFrame::builder()
        .add_column("id", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .add_column("age", vec![25.0, f64::NAN, 35.0, 40.0, f64::NAN, 50.0, 55.0, 60.0])
        .add_column("income", vec![30000.0, 45000.0, f64::NAN, 75000.0, 90000.0, f64::NAN, 120000.0, 135000.0])
        .add_column("score", vec![65.0, 70.0, 75.0, f64::NAN, 85.0, 90.0, f64::NAN, 100.0])
        .build()
        .unwrap();

    println!("Original DataFrame (with missing values):");
    println!("{}\n", df);

    // 1. Detect missing values
    println!("=== 1. DETECT MISSING VALUES ===");
    println!("Has any NaN? {}", df.has_na());
    let na_counts = df.count_na();
    println!("\nNaN count per column:");
    for (col, count) in &na_counts {
        println!("  {}: {} missing values", col, count);
    }
    println!();

    // 2. DROP rows with NaN
    println!("=== 2. DROPNA - Remove rows with missing values ===");
    let cleaned = df.dropna().unwrap();
    println!("After dropna() - {} rows remain (from {} original):",
             cleaned.n_rows(), df.n_rows());
    println!("{}\n", cleaned);

    // 3. FILL with constant value
    println!("=== 3. FILLNA - Fill with constant value (0) ===");
    let filled_zero = df.fillna(0.0).unwrap();
    println!("{}\n", filled_zero);

    // 4. FILL specific column
    println!("=== 4. FILLNA_COLUMN - Fill only 'age' column with 999 ===");
    let filled_age = df.fillna_column("age", 999.0).unwrap();
    println!("{}\n", filled_age);

    // 5. FILL with MEAN
    println!("=== 5. FILLNA_MEAN - Fill with column means ===");
    let filled_mean = df.fillna_mean().unwrap();
    println!("NaN values replaced with column averages:");
    println!("{}", filled_mean);

    // Show what the means were
    let means = df.select(&["age", "income", "score"])
        .unwrap()
        .dropna()
        .unwrap()
        .mean();
    println!("\nColumn means used:");
    for (col, mean) in means {
        println!("  {}: {:.2}", col, mean);
    }
    println!();

    // 6. FILL with MEDIAN
    println!("=== 6. FILLNA_MEDIAN - Fill with column medians ===");
    let filled_median = df.fillna_median().unwrap();
    println!("NaN values replaced with column medians:");
    println!("{}", filled_median);

    let medians = df.select(&["age", "income", "score"])
        .unwrap()
        .dropna()
        .unwrap()
        .median();
    println!("\nColumn medians used:");
    for (col, median) in medians {
        println!("  {}: {:.2}", col, median);
    }
    println!();

    // 7. Real-world workflow
    println!("=== 7. REAL-WORLD WORKFLOW ===");
    println!("Step 1: Load data and check for missing values");
    println!("Step 2: Decide strategy based on data");
    println!("Step 3: Apply appropriate method\n");

    println!("Example: Economic analysis");
    println!("- Income missing? Fill with median (robust to outliers)");
    println!("- Age missing? Fill with mean (normally distributed)");
    println!("- Score missing? Drop row (critical variable)\n");

    // Custom strategy
    let processed = df
        .fillna_median()  // Fill all with median first
        .unwrap()
        .filter(|row| {   // Then remove rows where critical columns still problematic
            row.get("id").map(|&v| !v.is_nan()).unwrap_or(false)
        })
        .unwrap();

    println!("After custom processing:");
    println!("{}\n", processed);

    // 8. Statistical comparison
    println!("=== 8. IMPACT ON STATISTICS ===");

    let stats_original = df.dropna().unwrap();
    let stats_mean = df.fillna_mean().unwrap();
    let stats_median = df.fillna_median().unwrap();

    println!("Mean income:");
    println!("  After dropna: {:.2}", stats_original.mean().get("income").unwrap());
    println!("  After fillna_mean: {:.2}", stats_mean.mean().get("income").unwrap());
    println!("  After fillna_median: {:.2}", stats_median.mean().get("income").unwrap());

    println!("\nDataFrame sizes:");
    println!("  Original: {} rows", df.n_rows());
    println!("  After dropna: {} rows ({:.1}% data loss)",
             stats_original.n_rows(),
             (1.0 - stats_original.n_rows() as f64 / df.n_rows() as f64) * 100.0);
    println!("  After fillna: {} rows (no data loss)", stats_mean.n_rows());

    println!("\n=== BEST PRACTICES ===");
    println!("1. Always check for missing values first: has_na(), count_na()");
    println!("2. Understand WHY data is missing (random vs systematic)");
    println!("3. Choose method based on data distribution:");
    println!("   - Mean: Normal distribution, few outliers");
    println!("   - Median: Skewed distribution, many outliers");
    println!("   - Drop: When missingness is informative or sample size is large");
    println!("4. Document your choice and test sensitivity");

    println!("\n=== Demo Complete! ===");
}
