use greeners::DataFrame;

fn main() {
    println!("=== MISSING DATA FEATURES - v1.8.0 ===\n");
    println!("Demonstrating comprehensive missing data handling in Greeners DataFrame\n");

    // ========== 1. UNDERSTANDING MISSING DATA ==========
    println!("=== 1. Understanding Missing Data in Greeners ===\n");

    println!("⚠️  NaN Concept:");
    println!("  • Only FLOAT columns can have NaN (Not a Number) values");
    println!("  • Int, Bool, DateTime, String, Categorical: NO NaN concept\n");

    let data_df = DataFrame::builder()
        .add_column("price", vec![100.0, f64::NAN, 105.0, f64::NAN, 110.0])
        .add_column("volume", vec![1000.0, 1200.0, f64::NAN, 1300.0, 1250.0])
        .add_int("id", vec![1, 2, 3, 4, 5])
        .build()
        .unwrap();

    println!("Sample data with missing values:");
    println!("{}\n", data_df);

    // ========== 2. DETECTING MISSING DATA ==========
    println!("=== 2. Detecting Missing Values ===\n");

    println!("--- count_na() - Count NaN per column ---");
    let na_counts = data_df.count_na();
    for (col, count) in &na_counts {
        println!("  {}: {} NaN values", col, count);
    }
    println!();

    println!("--- has_na() - Check if DataFrame has any NaN ---");
    println!("  Has NaN? {}\n", data_df.has_na());

    println!("--- isna() - Boolean mask showing NaN locations ---");
    let na_mask = data_df.isna().unwrap();
    println!("{}\n", na_mask);

    println!("--- notna() - Boolean mask showing non-NaN locations ---");
    let not_na_mask = data_df.notna().unwrap();
    println!("{}\n", not_na_mask);

    // ========== 3. DROPPING MISSING DATA ==========
    println!("=== 3. Dropping Rows with Missing Values ===\n");

    let messy_df = DataFrame::builder()
        .add_column("x", vec![1.0, f64::NAN, 3.0, 4.0, f64::NAN])
        .add_column("y", vec![10.0, 20.0, f64::NAN, 40.0, 50.0])
        .add_column("z", vec![100.0, 200.0, 300.0, 400.0, 500.0])
        .build()
        .unwrap();

    println!("Original data:");
    println!("{}\n", messy_df);

    println!("--- dropna() - Drop rows with ANY NaN ---");
    let cleaned_any = messy_df.dropna().unwrap();
    println!(
        "  Rows dropped: {} → {} rows",
        messy_df.n_rows(),
        cleaned_any.n_rows()
    );
    println!("{}\n", cleaned_any);

    println!("--- dropna_subset([\"x\"]) - Drop rows with NaN only in column 'x' ---");
    let cleaned_subset = messy_df.dropna_subset(&["x"]).unwrap();
    println!(
        "  Rows dropped: {} → {} rows",
        messy_df.n_rows(),
        cleaned_subset.n_rows()
    );
    println!("{}\n", cleaned_subset);

    let all_nan_df = DataFrame::builder()
        .add_column("a", vec![f64::NAN, 1.0, f64::NAN, 3.0])
        .add_column("b", vec![f64::NAN, 2.0, 4.0, 5.0])
        .build()
        .unwrap();

    println!("Data for dropna_all() test:");
    println!("{}\n", all_nan_df);

    println!("--- dropna_all() - Drop rows where ALL values are NaN ---");
    let cleaned_all = all_nan_df.dropna_all().unwrap();
    println!(
        "  Rows dropped: {} → {} rows",
        all_nan_df.n_rows(),
        cleaned_all.n_rows()
    );
    println!("{}\n", cleaned_all);

    // ========== 4. FILLING MISSING DATA - CONSTANT ==========
    println!("=== 4. Filling Missing Values with Constants ===\n");

    let gap_df = DataFrame::builder()
        .add_column("temperature", vec![20.0, f64::NAN, 22.0, f64::NAN, 24.0])
        .add_column("humidity", vec![60.0, 65.0, f64::NAN, 70.0, 72.0])
        .build()
        .unwrap();

    println!("Original data:");
    println!("{}\n", gap_df);

    println!("--- fillna(0.0) - Fill all NaN with 0 ---");
    let filled_zero = gap_df.fillna(0.0).unwrap();
    println!("{}\n", filled_zero);

    println!("--- fillna_column(\"temperature\", 21.0) - Fill specific column ---");
    let filled_specific = gap_df.fillna_column("temperature", 21.0).unwrap();
    println!("{}\n", filled_specific);

    // ========== 5. FILLING WITH STATISTICS ==========
    println!("=== 5. Filling with Statistical Values ===\n");

    let stats_df = DataFrame::builder()
        .add_column(
            "sales",
            vec![100.0, f64::NAN, 150.0, f64::NAN, 200.0, 180.0],
        )
        .build()
        .unwrap();

    println!("Original sales data:");
    println!("{}\n", stats_df);

    println!("--- fillna_mean() - Fill with column mean ---");
    let filled_mean = stats_df.fillna_mean().unwrap();
    println!("  Mean of [100, 150, 200, 180] = 157.5");
    println!("{}\n", filled_mean);

    println!("--- fillna_median() - Fill with column median ---");
    let filled_median = stats_df.fillna_median().unwrap();
    println!("  Median of [100, 150, 200, 180] = 165.0");
    println!("{}\n", filled_median);

    // ========== 6. FORWARD FILL (FFILL) ==========
    println!("=== 6. Forward Fill (ffill) - Carry Last Value Forward ===\n");

    let time_series = DataFrame::builder()
        .add_column(
            "price",
            vec![100.0, f64::NAN, f64::NAN, 105.0, f64::NAN, f64::NAN, 110.0],
        )
        .build()
        .unwrap();

    println!("Time series with gaps:");
    println!("{}\n", time_series);

    println!("--- fillna_ffill() - Propagate last valid value forward ---");
    let ffilled = time_series.fillna_ffill().unwrap();
    println!("  100 → 100, 100, 105 → 105, 105, 110");
    println!("{}\n", ffilled);

    // ========== 7. BACKWARD FILL (BFILL) ==========
    println!("=== 7. Backward Fill (bfill) - Carry Next Value Backward ===\n");

    let reverse_series = DataFrame::builder()
        .add_column(
            "price",
            vec![f64::NAN, f64::NAN, 100.0, f64::NAN, 105.0, f64::NAN],
        )
        .build()
        .unwrap();

    println!("Time series with leading gaps:");
    println!("{}\n", reverse_series);

    println!("--- fillna_bfill() - Propagate next valid value backward ---");
    let bfilled = reverse_series.fillna_bfill().unwrap();
    println!("  100 ← 100, 100, 105 ← 105, (trailing NaN remains)");
    println!("{}\n", bfilled);

    // ========== 8. COMBINING FILL METHODS ==========
    println!("=== 8. Combining Fill Methods ===\n");

    let combo_df = DataFrame::builder()
        .add_column(
            "value",
            vec![f64::NAN, f64::NAN, 10.0, f64::NAN, 20.0, f64::NAN, f64::NAN],
        )
        .build()
        .unwrap();

    println!("Data with leading and trailing NaN:");
    println!("{}\n", combo_df);

    println!("--- fillna_ffill() then fillna_bfill() - Complete fill ---");
    let complete_filled = combo_df.fillna_ffill().unwrap().fillna_bfill().unwrap();
    println!("  First ffill: NaN, NaN, 10, 10, 20, 20, 20");
    println!("  Then bfill: 10, 10, 10, 10, 20, 20, 20");
    println!("{}\n", complete_filled);

    // ========== 9. PRACTICAL EXAMPLE: STOCK PRICES ==========
    println!("=== 9. Practical Example: Stock Price Data ===\n");

    let stock_df = DataFrame::builder()
        .add_column("open", vec![150.0, f64::NAN, 152.0, f64::NAN, 155.0])
        .add_column("close", vec![151.0, 151.5, f64::NAN, 154.0, 156.0])
        .add_column(
            "volume",
            vec![1000000.0, f64::NAN, 1100000.0, 950000.0, f64::NAN],
        )
        .build()
        .unwrap();

    println!("Stock data with missing values:");
    println!("{}\n", stock_df);

    println!("Step 1: Check missing data pattern");
    let na_count = stock_df.count_na();
    println!("  Missing values per column:");
    for (col, count) in &na_count {
        println!("    {}: {}", col, count);
    }
    println!();

    println!("Step 2: Forward fill prices (carry last known price)");
    let filled_stock = stock_df.fillna_ffill().unwrap();
    println!("{}\n", filled_stock);

    println!("Step 3: Verify no missing data");
    println!("  Has NaN after filling? {}\n", filled_stock.has_na());

    // ========== 10. PRACTICAL EXAMPLE: SENSOR DATA ==========
    println!("=== 10. Practical Example: Sensor Data ===\n");

    let sensor_df = DataFrame::builder()
        .add_column("sensor_1", vec![22.5, 22.6, f64::NAN, f64::NAN, 23.0])
        .add_column("sensor_2", vec![60.0, f64::NAN, 62.0, 63.0, f64::NAN])
        .build()
        .unwrap();

    println!("Sensor readings with intermittent failures:");
    println!("{}\n", sensor_df);

    println!("Strategy: Use median fill (robust to outliers)");
    let robust_filled = sensor_df.fillna_median().unwrap();
    println!("{}\n", robust_filled);

    // ========== 11. PRACTICAL EXAMPLE: SURVEY DATA ==========
    println!("=== 11. Practical Example: Survey Data ===\n");

    let survey_df = DataFrame::builder()
        .add_column("q1_score", vec![5.0, 4.0, f64::NAN, 3.0, 5.0, f64::NAN])
        .add_column("q2_score", vec![4.0, f64::NAN, 5.0, f64::NAN, 4.0, 3.0])
        .add_column("q3_score", vec![5.0, 5.0, 4.0, 4.0, f64::NAN, f64::NAN])
        .build()
        .unwrap();

    println!("Survey responses (NaN = no response):");
    println!("{}\n", survey_df);

    println!("--- Analysis Strategy: Drop rows with >50% missing ---");
    println!("(In practice, you'd calculate missingness per row)");
    println!("\nFor now, using dropna() to keep only complete responses:");
    let complete_surveys = survey_df.dropna().unwrap();
    println!(
        "  Complete responses: {} / {}",
        complete_surveys.n_rows(),
        survey_df.n_rows()
    );
    println!("{}\n", complete_surveys);

    // ========== 12. EDGE CASES ==========
    println!("=== 12. Edge Cases ===\n");

    println!("--- All NaN DataFrame ---");
    let all_nan = DataFrame::builder()
        .add_column("x", vec![f64::NAN, f64::NAN, f64::NAN])
        .build()
        .unwrap();

    println!("{}", all_nan);
    let filled_all_nan = all_nan.fillna_mean().unwrap();
    println!("fillna_mean() on all-NaN: remains NaN (no valid mean)");
    println!("{}\n", filled_all_nan);

    println!("--- No NaN DataFrame ---");
    let no_nan = DataFrame::builder()
        .add_column("x", vec![1.0, 2.0, 3.0])
        .build()
        .unwrap();

    let filled_no_nan = no_nan.fillna_ffill().unwrap();
    println!("fillna_ffill() on no-NaN: unchanged");
    println!("{}\n", filled_no_nan);

    // ========== 13. CHAINING OPERATIONS ==========
    println!("=== 13. Chaining Missing Data Operations ===\n");

    let pipeline_df = DataFrame::builder()
        .add_column(
            "raw_data",
            vec![
                f64::NAN,
                10.0,
                f64::NAN,
                f64::NAN,
                15.0,
                f64::NAN,
                20.0,
                f64::NAN,
            ],
        )
        .build()
        .unwrap();

    println!("Original data:");
    println!("{}\n", pipeline_df);

    println!("Pipeline: ffill → bfill → dropna (if any remain)");
    let pipeline_result = pipeline_df
        .fillna_ffill()
        .unwrap()
        .fillna_bfill()
        .unwrap()
        .dropna()
        .unwrap();

    println!("After pipeline:");
    println!("{}\n", pipeline_result);
    println!("  All gaps filled? {}\n", !pipeline_result.has_na());

    // ========== 14. COMPARISON OF FILL METHODS ==========
    println!("=== 14. Comparison of Fill Methods ===\n");

    let compare_df = DataFrame::builder()
        .add_column(
            "value",
            vec![10.0, f64::NAN, f64::NAN, 20.0, f64::NAN, 30.0],
        )
        .build()
        .unwrap();

    println!("Original: [10, NaN, NaN, 20, NaN, 30]");
    println!();

    let filled_zero = compare_df.fillna(0.0).unwrap();
    println!(
        "fillna(0):    {:?}",
        filled_zero.get("value").unwrap().to_vec()
    );

    let filled_mean = compare_df.fillna_mean().unwrap();
    println!(
        "fillna_mean:  {:?}",
        filled_mean.get("value").unwrap().to_vec()
    );

    let filled_median = compare_df.fillna_median().unwrap();
    println!(
        "fillna_median:{:?}",
        filled_median.get("value").unwrap().to_vec()
    );

    let filled_ffill = compare_df.fillna_ffill().unwrap();
    println!(
        "fillna_ffill: {:?}",
        filled_ffill.get("value").unwrap().to_vec()
    );

    let filled_bfill = compare_df.fillna_bfill().unwrap();
    println!(
        "fillna_bfill: {:?}\n",
        filled_bfill.get("value").unwrap().to_vec()
    );

    // ========== 15. DECISION TREE ==========
    println!("=== 15. Missing Data Strategy Decision Tree ===\n");

    println!("When to use each method:");
    println!();
    println!("📊 dropna():");
    println!("  ✓ Small amount of missing data (<5%)");
    println!("  ✓ Data is missing completely at random");
    println!("  ✓ Large dataset (can afford to lose rows)");
    println!();
    println!("📊 dropna_subset([cols]):");
    println!("  ✓ Missing data only matters in specific columns");
    println!("  ✓ Other columns can be incomplete");
    println!();
    println!("📊 dropna_all():");
    println!("  ✓ Want to keep rows with at least some data");
    println!("  ✓ Only remove completely empty rows");
    println!();
    println!("📊 fillna_mean() / fillna_median():");
    println!("  ✓ Data is missing at random");
    println!("  ✓ Want to preserve distribution");
    println!("  ✓ Cross-sectional data (not time series)");
    println!();
    println!("📊 fillna_ffill():");
    println!("  ✓ Time series data");
    println!("  ✓ Values change slowly");
    println!("  ✓ Last observation is reasonable proxy");
    println!();
    println!("📊 fillna_bfill():");
    println!("  ✓ Future information is valid to use");
    println!("  ✓ Filling leading gaps");
    println!();
    println!("📊 fillna(constant):");
    println!("  ✓ Domain knowledge suggests specific value");
    println!("  ✓ Zero/default value is meaningful");
    println!();

    // ========== SUMMARY ==========
    println!("=== FEATURE SUMMARY ===\n");

    println!("✅ Detection Methods (v1.8.0):");
    println!("  • count_na() - Count NaN per column");
    println!("  • has_na() - Check if any NaN exists");
    println!("  • isna() - Boolean mask of NaN locations");
    println!("  • notna() - Boolean mask of non-NaN locations");

    println!("\n✅ Removal Methods:");
    println!("  • dropna() - Remove rows with ANY NaN");
    println!("  • dropna_subset([cols]) - Remove rows with NaN in specific columns");
    println!("  • dropna_all() - Remove rows where ALL values are NaN");

    println!("\n✅ Filling Methods:");
    println!("  • fillna(value) - Fill all NaN with constant");
    println!("  • fillna_column(col, value) - Fill specific column");
    println!("  • fillna_mean() - Fill with column mean");
    println!("  • fillna_median() - Fill with column median");
    println!("  • fillna_ffill() - Forward fill (carry last value)");
    println!("  • fillna_bfill() - Backward fill (carry next value)");

    println!("\n✅ Key Concepts:");
    println!("  • Only Float columns can have NaN");
    println!("  • Int, Bool, DateTime, String, Categorical: No NaN concept");
    println!("  • Methods can be chained for complex pipelines");
    println!("  • ffill + bfill can fill all gaps in time series");

    println!("\n✅ Best Practices:");
    println!("  1. Always inspect missing data pattern first (count_na, isna)");
    println!("  2. Understand WHY data is missing (random vs systematic)");
    println!("  3. Choose fill method based on data type and domain");
    println!("  4. Document your imputation strategy");
    println!("  5. Consider creating indicator variables for missingness");

    println!("\n=== Demo Complete! ===");
}
