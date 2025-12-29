use greeners::DataFrame;

fn main() {
    println!("=== CATEGORICAL FEATURES - v1.3.0 ===\n");
    println!("Demonstrating Categorical column support in Greeners DataFrame\n");

    // ========== 1. BASIC CATEGORICAL CREATION ==========
    println!("=== 1. Creating DataFrame with Categorical Columns ===\n");

    let sales_df = DataFrame::builder()
        .add_column(
            "amount",
            vec![1000.0, 1500.0, 1200.0, 1800.0, 900.0, 2000.0],
        )
        .add_column("quantity", vec![10.0, 15.0, 12.0, 20.0, 8.0, 18.0])
        .add_categorical(
            "region",
            vec![
                "North".to_string(),
                "South".to_string(),
                "North".to_string(),
                "East".to_string(),
                "South".to_string(),
                "East".to_string(),
            ],
        )
        .add_categorical(
            "product",
            vec![
                "Widget".to_string(),
                "Gadget".to_string(),
                "Widget".to_string(),
                "Widget".to_string(),
                "Gadget".to_string(),
                "Gadget".to_string(),
            ],
        )
        .build()
        .unwrap();

    println!("Sales DataFrame with Categorical columns:");
    println!("{}\n", sales_df);

    // ========== 2. INSPECTING CATEGORICAL COLUMNS ==========
    println!("=== 2. Inspecting Categorical Data ===\n");

    // Get categorical column
    let region_col = sales_df.get_categorical("region").unwrap();
    println!("Region column details:");
    println!("  Number of unique levels: {}", region_col.n_levels());
    println!("  Levels: {:?}", region_col.levels);
    println!("  Internal codes: {:?}", region_col.codes);

    // Value counts
    let region_counts = region_col.value_counts();
    println!("\nRegion value counts:");
    for (level, count) in &region_counts {
        println!("  {}: {}", level, count);
    }
    println!();

    // ========== 3. MIXED TYPE OPERATIONS ==========
    println!("=== 3. Operations with Mixed Types (Float + Categorical) ===\n");

    // Statistical operations work on all columns
    println!("--- Descriptive Statistics ---");
    let stats = sales_df.describe();
    for (col_name, col_stats) in &stats {
        println!("\n{}:", col_name);
        for (stat, value) in col_stats {
            println!("  {}: {:.2}", stat, value);
        }
    }
    println!();

    // Filtering works on mixed types
    println!("--- Filter: amount > 1200 ---");
    let filtered = sales_df
        .filter(|row| row.get("amount").map(|&v| v > 1200.0).unwrap_or(false))
        .unwrap();
    println!("{}\n", filtered);

    // Sorting by categorical column (uses numeric codes internally)
    println!("--- Sort by region (alphabetically via codes) ---");
    let sorted = sales_df.sort_by("region", true).unwrap();
    println!("{}\n", sorted);

    // ========== 4. GROUPBY WITH CATEGORICAL ==========
    println!("=== 4. GroupBy Operations ===\n");

    println!("--- Total sales by region ---");
    let by_region = sales_df.groupby(&["region"], "amount", "sum").unwrap();
    println!("{}\n", by_region);

    println!("--- Average quantity by product ---");
    let by_product = sales_df.groupby(&["product"], "quantity", "mean").unwrap();
    println!("{}\n", by_product);

    println!("--- Count transactions by region ---");
    let count_by_region = sales_df.groupby(&["region"], "amount", "count").unwrap();
    println!("{}\n", count_by_region);

    // ========== 5. CONCATENATION ==========
    println!("=== 5. Concatenating DataFrames with Categoricals ===\n");

    let new_sales = DataFrame::builder()
        .add_column("amount", vec![1100.0, 1600.0])
        .add_column("quantity", vec![11.0, 16.0])
        .add_categorical("region", vec!["West".to_string(), "North".to_string()])
        .add_categorical("product", vec!["Widget".to_string(), "Gadget".to_string()])
        .build()
        .unwrap();

    println!("New sales to append:");
    println!("{}\n", new_sales);

    let combined = sales_df.concat(&new_sales).unwrap();
    println!("Combined DataFrame:");
    println!("{}\n", combined);

    // Check updated levels
    let combined_region = combined.get_categorical("region").unwrap();
    println!(
        "Updated region levels: {:?} ({} unique)\n",
        combined_region.levels,
        combined_region.n_levels()
    );

    // ========== 6. SELECTION AND SLICING ==========
    println!("=== 6. Selection and Slicing ===\n");

    println!("--- Select specific columns ---");
    let selected = sales_df.select(&["region", "amount"]).unwrap();
    println!("{}\n", selected);

    println!("--- Head (first 3 rows) ---");
    let head = sales_df.head(3).unwrap();
    println!("{}\n", head);

    println!("--- Tail (last 2 rows) ---");
    let tail = sales_df.tail(2).unwrap();
    println!("{}\n", tail);

    // ========== 7. EXPORT/IMPORT ==========
    println!("=== 7. Export to CSV (preserves categorical as strings) ===\n");

    // Note: Categorical columns are exported as strings in CSV
    println!("When exported to CSV:");
    println!("  - Float columns → numeric values");
    println!("  - Categorical columns → string values (e.g., 'North', 'South')");
    println!("  - Use df.to_csv('output.csv') to save\n");

    // ========== 8. DUMMY VARIABLES (ONE-HOT ENCODING) ==========
    println!("=== 8. Creating Dummy Variables (One-Hot Encoding) ===\n");

    let region_col_dummies = sales_df.get_categorical("region").unwrap();

    println!("--- Without dropping first level ---");
    let dummies_full = region_col_dummies.get_dummies("region", false);
    println!("Created {} dummy columns:", dummies_full.len());
    for (col_name, _) in &dummies_full {
        println!("  {}", col_name);
    }

    println!("\n--- With dropping first level (for regression) ---");
    let dummies_drop = region_col_dummies.get_dummies("region", true);
    println!(
        "Created {} dummy columns (dropped baseline):",
        dummies_drop.len()
    );
    for (col_name, _) in &dummies_drop {
        println!("  {}", col_name);
    }
    println!();

    // Example: Create dummy for one category
    if let Some(region_north) = dummies_full.get("region_North") {
        println!("region_North dummy values: {:?}\n", region_north.to_vec());
    }

    // ========== 9. PRACTICAL EXAMPLE: SALES ANALYSIS ==========
    println!("=== 9. Practical Example: Sales Analysis Dashboard ===\n");

    let monthly_sales = DataFrame::builder()
        .add_column("month", vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        .add_column(
            "revenue",
            vec![
                50000.0, 55000.0, 60000.0, 48000.0, 52000.0, 58000.0, 51000.0, 56000.0, 62000.0,
            ],
        )
        .add_categorical(
            "store",
            vec![
                "Store A".to_string(),
                "Store A".to_string(),
                "Store A".to_string(),
                "Store B".to_string(),
                "Store B".to_string(),
                "Store B".to_string(),
                "Store C".to_string(),
                "Store C".to_string(),
                "Store C".to_string(),
            ],
        )
        .add_categorical(
            "quarter",
            vec![
                "Q1".to_string(),
                "Q1".to_string(),
                "Q1".to_string(),
                "Q1".to_string(),
                "Q1".to_string(),
                "Q1".to_string(),
                "Q1".to_string(),
                "Q1".to_string(),
                "Q1".to_string(),
            ],
        )
        .build()
        .unwrap();

    println!("Monthly sales data:");
    println!("{}\n", monthly_sales);

    // Analysis 1: Average revenue by store
    println!("--- Average Revenue by Store ---");
    let avg_by_store = monthly_sales
        .groupby(&["store"], "revenue", "mean")
        .unwrap();
    println!("{}\n", avg_by_store);

    // Analysis 2: Total revenue by month
    println!("--- Total Revenue by Month ---");
    let by_month = monthly_sales.groupby(&["month"], "revenue", "sum").unwrap();
    println!("{}\n", by_month);

    // Analysis 3: Best performing store (max revenue)
    println!("--- Maximum Revenue by Store ---");
    let max_by_store = monthly_sales.groupby(&["store"], "revenue", "max").unwrap();
    println!("{}\n", max_by_store);

    // ========== 10. SUMMARY ==========
    println!("=== FEATURE SUMMARY ===\n");

    println!("✅ Categorical Column Support:");
    println!("  • add_categorical(name, values) - Create categorical column");
    println!("  • get_categorical(name) - Access categorical data");
    println!("  • Automatic string-to-integer encoding");
    println!("  • Memory efficient (stores integers, not strings)");

    println!("\n✅ Operations:");
    println!("  • groupby() - Aggregate by categorical variables");
    println!("  • sort_by() - Sort by categorical columns");
    println!("  • filter() - Filter with categorical conditions");
    println!("  • concat() - Combine datasets with categoricals");

    println!("\n✅ Display:");
    println!("  • Categorical columns show as strings");
    println!("  • Float columns show as numbers");
    println!("  • Mixed display in single DataFrame");

    println!("\n✅ Export:");
    println!("  • to_csv() - Categorical exported as strings");
    println!("  • to_json() - Type-aware JSON export");

    println!("\n✅ Statistical:");
    println!("  • describe() - Works on all column types");
    println!("  • Categorical uses integer codes for calculations");

    println!("\n✅ Advanced:");
    println!("  • get_dummies() - One-hot encoding");
    println!("  • value_counts() - Frequency tables");
    println!("  • n_levels() - Count unique categories");

    println!("\n=== Demo Complete! ===");
}
