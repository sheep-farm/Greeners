use greeners::DataFrame;

fn main() {
    println!("=== INTEGER FEATURES - v1.5.0 ===\n");
    println!("Demonstrating Integer column support in Greeners DataFrame\n");

    // ========== 1. BASIC INTEGER CREATION ==========
    println!("=== 1. Creating DataFrame with Integer Columns ===\n");

    let panel_df = DataFrame::builder()
        .add_int("firm_id", vec![1, 1, 1, 2, 2, 2, 3, 3, 3])
        .add_int(
            "year",
            vec![2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022],
        )
        .add_column(
            "revenue",
            vec![
                1000.0, 1200.0, 1400.0, 800.0, 900.0, 1000.0, 1500.0, 1600.0, 1700.0,
            ],
        )
        .add_column(
            "employees",
            vec![10.0, 12.0, 15.0, 8.0, 9.0, 10.0, 20.0, 22.0, 25.0],
        )
        .build()
        .unwrap();

    println!("Panel DataFrame with Integer columns:");
    println!("{}\n", panel_df);

    // ========== 2. INSPECTING INTEGER COLUMNS ==========
    println!("=== 2. Inspecting Integer Data ===\n");

    // Get integer column
    let firm_ids = panel_df.get_int("firm_id").unwrap();
    println!("firm_id column details:");
    println!("  Length: {}", firm_ids.len());
    println!("  Values: {:?}", firm_ids.to_vec());
    println!("  Min: {}", firm_ids.iter().min().unwrap());
    println!("  Max: {}", firm_ids.iter().max().unwrap());

    let years = panel_df.get_int("year").unwrap();
    println!("\nyear column details:");
    println!("  Unique years: {:?}\n", {
        let mut unique: Vec<_> = years.to_vec();
        unique.sort();
        unique.dedup();
        unique
    });

    // ========== 3. MIXED TYPE OPERATIONS ==========
    println!("=== 3. Operations with Mixed Types (Float + Int) ===\n");

    // Statistical operations work on all columns (Int converts to f64)
    println!("--- Descriptive Statistics ---");
    let stats = panel_df.describe();
    for (col_name, col_stats) in &stats {
        println!("\n{}:", col_name);
        for (stat, value) in col_stats {
            println!("  {}: {:.2}", stat, value);
        }
    }
    println!();

    // ========== 4. INTEGER TO NUMERIC CONVERSION ==========
    println!("=== 4. Integer to Numeric Conversion ===\n");

    println!("Integer columns convert to f64 for numeric operations:");
    let int_col = panel_df.get_column("year").unwrap();
    let numeric = int_col.to_float();
    println!("  year as float: {:?}\n", &numeric.to_vec()[0..3]);

    // ========== 5. CONCATENATION ==========
    println!("=== 5. Concatenating DataFrames with Integers ===\n");

    let new_data = DataFrame::builder()
        .add_int("firm_id", vec![4, 4, 4])
        .add_int("year", vec![2020, 2021, 2022])
        .add_column("revenue", vec![1100.0, 1250.0, 1450.0])
        .add_column("employees", vec![11.0, 13.0, 16.0])
        .build()
        .unwrap();

    println!("New data to append:");
    println!("{}\n", new_data);

    let combined = panel_df.concat(&new_data).unwrap();
    println!("Combined DataFrame ({} rows):", combined.n_rows());
    println!("{}\n", combined);

    // ========== 6. SELECTION AND SLICING ==========
    println!("=== 6. Selection and Slicing ===\n");

    println!("--- Select specific columns ---");
    let selected = panel_df.select(&["firm_id", "year", "revenue"]).unwrap();
    println!("{}\n", selected);

    println!("--- Filter: year >= 2021 ---");
    let filtered = panel_df
        .filter(|row| row.get("year").map(|&v| v >= 2021.0).unwrap_or(false))
        .unwrap();
    println!("{}\n", filtered);

    // ========== 7. EXPORT/IMPORT ==========
    println!("=== 7. Export to CSV/JSON (preserves integers) ===\n");

    println!("When exported:");
    println!("  - CSV: Integer columns → integer strings (no decimals)");
    println!("  - JSON: Integer columns → JSON integer values");
    println!("  - Use df.to_csv('output.csv') or df.to_json('output.json')\n");

    // ========== 8. PRACTICAL EXAMPLE: PANEL DATA ANALYSIS ==========
    println!("=== 8. Practical Example: Panel Data Analysis ===\n");

    let panel = DataFrame::builder()
        .add_int("entity_id", vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        .add_int("time", vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        .add_column(
            "outcome",
            vec![
                10.0, 12.0, 15.0, 18.0, 8.0, 10.0, 13.0, 16.0, 12.0, 14.0, 17.0, 20.0,
            ],
        )
        .add_column(
            "treatment",
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .build()
        .unwrap();

    println!("Panel data (entities × time):");
    println!("{}\n", panel);

    // Analysis: Group by entity
    println!("--- Average outcome by entity ---");
    let by_entity = panel.groupby(&["entity_id"], "outcome", "mean").unwrap();
    println!("{}\n", by_entity);

    println!("--- Average outcome by time period ---");
    let by_time = panel.groupby(&["time"], "outcome", "mean").unwrap();
    println!("{}\n", by_time);

    // ========== 9. INTEGER USE CASES ==========
    println!("=== 9. Common Integer Use Cases ===\n");

    let transactions = DataFrame::builder()
        .add_int("transaction_id", vec![1001, 1002, 1003, 1004, 1005])
        .add_int("customer_id", vec![5, 3, 5, 7, 3])
        .add_int("product_id", vec![101, 102, 103, 101, 102])
        .add_int("quantity", vec![2, 1, 3, 1, 2])
        .add_column("price", vec![29.99, 49.99, 19.99, 29.99, 49.99])
        .build()
        .unwrap();

    println!("Transaction data:");
    println!("{}\n", transactions);

    println!("--- Total quantity by customer ---");
    let by_customer = transactions
        .groupby(&["customer_id"], "quantity", "sum")
        .unwrap();
    println!("{}\n", by_customer);

    println!("--- Transactions per product ---");
    let by_product = transactions
        .groupby(&["product_id"], "quantity", "count")
        .unwrap();
    println!("{}\n", by_product);

    // ========== 10. NEGATIVE INTEGERS ==========
    println!("=== 10. Negative Integers Support ===\n");

    let financial_df = DataFrame::builder()
        .add_int("quarter", vec![1, 2, 3, 4])
        .add_int("profit", vec![50000, -20000, 30000, -10000]) // Losses as negatives
        .add_column("margin", vec![0.15, -0.05, 0.10, -0.02])
        .build()
        .unwrap();

    println!("Financial data with negative values:");
    println!("{}\n", financial_df);

    let profit_col = financial_df.get_int("profit").unwrap();
    let total_profit: i64 = profit_col.iter().sum();
    let profitable_quarters = profit_col.iter().filter(|&&p| p > 0).count();

    println!("Analysis:");
    println!("  Total profit: ${}", total_profit);
    println!("  Profitable quarters: {}/4", profitable_quarters);
    println!("  Loss quarters: {}/4\n", 4 - profitable_quarters);

    // ========== 11. DATA TYPES SUMMARY ==========
    println!("=== 11. Data Type Information ===\n");

    let mixed_df = DataFrame::builder()
        .add_column("float_col", vec![1.5, 2.7, 3.9])
        .add_int("int_col", vec![1, 2, 3])
        .add_categorical(
            "cat_col",
            vec!["A".to_string(), "B".to_string(), "A".to_string()],
        )
        .add_bool("bool_col", vec![true, false, true])
        .build()
        .unwrap();

    println!("Mixed-type DataFrame:");
    println!("{}\n", mixed_df);

    println!("Column types:");
    for name in ["float_col", "int_col", "cat_col", "bool_col"].iter() {
        let col = mixed_df.get_column(name).unwrap();
        println!("  {}: {:?}", name, col.dtype());
    }
    println!();

    // ========== 12. INTEGER STATISTICS ==========
    println!("=== 12. Integer Statistics ===\n");

    let ages = DataFrame::builder()
        .add_int("age", vec![25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
        .build()
        .unwrap();

    let age_col = ages.get_int("age").unwrap();

    println!("Age distribution:");
    println!("  Count: {}", age_col.len());
    println!("  Min: {}", age_col.iter().min().unwrap());
    println!("  Max: {}", age_col.iter().max().unwrap());
    println!(
        "  Mean: {:.1}",
        age_col.iter().map(|&x| x as f64).sum::<f64>() / age_col.len() as f64
    );
    println!();

    // ========== SUMMARY ==========
    println!("=== FEATURE SUMMARY ===\n");

    println!("✅ Integer Column Support (v1.5.0):");
    println!("  • add_int(name, values) - Create integer column");
    println!("  • get_int(name) - Access integer data");
    println!("  • Signed 64-bit integers (i64) - supports negative values");
    println!("  • Memory efficient (8 bytes per value)");

    println!("\n✅ Operations:");
    println!("  • to_float() - Convert i64 → f64 for calculations");
    println!("  • describe() - Statistics (min, max, mean, std)");
    println!("  • groupby() - Aggregate by integer keys");
    println!("  • filter() - Filter with integer conditions");
    println!("  • concat() - Combine datasets with integers");
    println!("  • sort_by() - Sort by integer columns");

    println!("\n✅ Display:");
    println!("  • Integer columns show without decimals");
    println!("  • Right-aligned numeric formatting");
    println!("  • Mixed display in single DataFrame");

    println!("\n✅ Export:");
    println!("  • to_csv() - Integers exported without decimals");
    println!("  • to_json() - Integers exported as JSON numbers");

    println!("\n✅ Use Cases:");
    println!("  • Panel data - Entity IDs, Time periods");
    println!("  • Transactions - Customer IDs, Product IDs, Order numbers");
    println!("  • Counts - Quantities, Frequencies, Occurrences");
    println!("  • Years - Time series with annual data");
    println!("  • Financial - Profits/losses (negative integers)");
    println!("  • Categorical ordinal - Rankings, Levels");

    println!("\n✅ Comparison with Float:");
    println!("  • Int: Exact representation, no rounding errors");
    println!("  • Int: 8 bytes (same as Float f64)");
    println!("  • Int: Better for IDs, counts, discrete values");
    println!("  • Float: Better for continuous measurements");

    println!("\n=== Demo Complete! ===");
}
