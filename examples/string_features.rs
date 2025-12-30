use greeners::DataFrame;

fn main() {
    println!("=== STRING FEATURES - v1.7.0 ===\n");
    println!("Demonstrating String column support (free text) in Greeners DataFrame\n");

    // ========== 1. BASIC STRING CREATION ==========
    println!("=== 1. Creating DataFrame with String Columns ===\n");

    let customer_df = DataFrame::builder()
        .add_int("customer_id", vec![1, 2, 3, 4, 5])
        .add_string(
            "name",
            vec![
                "Alice Johnson".to_string(),
                "Bob Smith".to_string(),
                "Charlie Brown".to_string(),
                "Diana Prince".to_string(),
                "Eve Adams".to_string(),
            ],
        )
        .add_string(
            "email",
            vec![
                "alice@example.com".to_string(),
                "bob@example.com".to_string(),
                "charlie@example.com".to_string(),
                "diana@example.com".to_string(),
                "eve@example.com".to_string(),
            ],
        )
        .add_column("purchase_amount", vec![150.0, 200.0, 75.0, 300.0, 125.0])
        .build()
        .unwrap();

    println!("Customer DataFrame with String columns:");
    println!("{}\n", customer_df);

    // ========== 2. INSPECTING STRING COLUMNS ==========
    println!("=== 2. Inspecting String Data ===\n");

    let names = customer_df.get_string("name").unwrap();
    println!("name column details:");
    println!("  Length: {}", names.len());
    println!("  First name: {}", names[0]);
    println!("  Last name: {}", names[names.len() - 1]);

    let emails = customer_df.get_string("email").unwrap();
    println!("\nemail column details:");
    println!("  Sample: {:?}\n", &emails.to_vec()[0..3]);

    // ========== 3. STRING vs CATEGORICAL ==========
    println!("=== 3. String vs Categorical: Key Differences ===\n");

    println!("STRING columns (free text):");
    println!("  • Store raw text without encoding");
    println!("  • Use for: names, addresses, comments, descriptions");
    println!("  • Variable length, unique values");
    println!("  • Can't convert to numeric (to_float() → NaN)");

    println!("\nCATEGORICAL columns (encoded text):");
    println!("  • Encode text as integer codes");
    println!("  • Use for: categories, groups, fixed labels");
    println!("  • Repeated values, limited set");
    println!("  • Can convert to numeric codes for regression\n");

    let mixed_df = DataFrame::builder()
        .add_string(
            "product_name",
            vec![
                "MacBook Pro 14-inch".to_string(),
                "iPhone 15 Pro".to_string(),
                "AirPods Pro".to_string(),
            ],
        )
        .add_categorical(
            "category",
            vec![
                "Laptop".to_string(),
                "Phone".to_string(),
                "Accessory".to_string(),
            ],
        )
        .add_column("price", vec![1999.0, 999.0, 249.0])
        .build()
        .unwrap();

    println!("Example showing both types:");
    println!("{}\n", mixed_df);

    // ========== 4. MIXED TYPE OPERATIONS ==========
    println!("=== 4. Operations with Mixed Types (Float + String) ===\n");

    println!("--- Descriptive Statistics (numeric columns only) ---");
    let stats = customer_df.describe();
    for (col_name, col_stats) in &stats {
        if col_name != "customer_id" {
            // String columns don't appear in describe()
            println!("\n{}:", col_name);
            for (stat, value) in col_stats {
                println!("  {}: {:.2}", stat, value);
            }
        }
    }
    println!();

    // ========== 5. STRING TO NUMERIC CONVERSION ==========
    println!("=== 5. String to Numeric Conversion ===\n");

    println!("String columns can't convert to numeric (returns NaN):");
    let str_col = customer_df.get_column("name").unwrap();
    let numeric = str_col.to_float();
    println!("  name column to_float(): all NaN");
    println!("  First 3 values: {:?}", &numeric.to_vec()[0..3]);
    println!("  All NaN? {}\n", numeric.iter().all(|v| v.is_nan()));

    // ========== 6. CONCATENATION ==========
    println!("=== 6. Concatenating DataFrames with Strings ===\n");

    let new_customers = DataFrame::builder()
        .add_int("customer_id", vec![6, 7])
        .add_string(
            "name",
            vec!["Frank Miller".to_string(), "Grace Lee".to_string()],
        )
        .add_string(
            "email",
            vec![
                "frank@example.com".to_string(),
                "grace@example.com".to_string(),
            ],
        )
        .add_column("purchase_amount", vec![175.0, 225.0])
        .build()
        .unwrap();

    println!("New customers to append:");
    println!("{}\n", new_customers);

    let combined = customer_df.concat(&new_customers).unwrap();
    println!("Combined DataFrame ({} rows):", combined.n_rows());
    println!("{}\n", combined);

    // ========== 7. SELECTION AND SLICING ==========
    println!("=== 7. Selection and Slicing ===\n");

    println!("--- Select specific columns ---");
    let selected = customer_df.select(&["name", "purchase_amount"]).unwrap();
    println!("{}\n", selected);

    println!("--- Head (first 3 rows) ---");
    let head = customer_df.head(3).unwrap();
    println!("{}\n", head);

    println!("--- Filter: purchase_amount > 100 ---");
    let filtered = customer_df
        .filter(|row| {
            row.get("purchase_amount")
                .map(|&v| v > 100.0)
                .unwrap_or(false)
        })
        .unwrap();
    println!("{}\n", filtered);

    // ========== 8. EXPORT/IMPORT ==========
    println!("=== 8. Export to CSV/JSON (preserves strings) ===\n");

    println!("When exported:");
    println!("  - CSV: String columns → raw text values");
    println!("  - JSON: String columns → JSON string values");
    println!("  - Use df.to_csv('output.csv') or df.to_json('output.json')\n");

    // ========== 9. PRACTICAL EXAMPLE: RESEARCH DATA ==========
    println!("=== 9. Practical Example: Research Participant Data ===\n");

    let study_df = DataFrame::builder()
        .add_int("participant_id", vec![101, 102, 103, 104, 105])
        .add_string(
            "full_name",
            vec![
                "John Anderson".to_string(),
                "Sarah Martinez".to_string(),
                "Michael Chen".to_string(),
                "Emily Davis".to_string(),
                "Robert Wilson".to_string(),
            ],
        )
        .add_categorical(
            "treatment_group",
            vec![
                "Control".to_string(),
                "Treatment".to_string(),
                "Control".to_string(),
                "Treatment".to_string(),
                "Control".to_string(),
            ],
        )
        .add_column("outcome_score", vec![72.5, 85.3, 68.9, 91.2, 75.8])
        .add_string(
            "notes",
            vec![
                "Completed all sessions".to_string(),
                "Missed session 3".to_string(),
                "No issues reported".to_string(),
                "Excellent progress".to_string(),
                "Withdrew early".to_string(),
            ],
        )
        .build()
        .unwrap();

    println!("Research study data:");
    println!("{}\n", study_df);

    // ========== 10. EMPTY STRINGS AND EDGE CASES ==========
    println!("=== 10. Empty Strings and Edge Cases ===\n");

    let edge_case_df = DataFrame::builder()
        .add_string(
            "text",
            vec![
                "Normal text".to_string(),
                "".to_string(), // Empty string
                "Text with    spaces".to_string(),
                "Text\nwith\nnewlines".to_string(),
            ],
        )
        .add_column("value", vec![1.0, 2.0, 3.0, 4.0])
        .build()
        .unwrap();

    println!("DataFrame with edge cases:");
    println!("{}\n", edge_case_df);

    let text_col = edge_case_df.get_string("text").unwrap();
    println!("Edge case analysis:");
    println!(
        "  Empty strings: {}",
        text_col.iter().filter(|s| s.is_empty()).count()
    );
    println!(
        "  With newlines: {}",
        text_col.iter().filter(|s| s.contains('\n')).count()
    );
    println!(
        "  Average length: {:.1}\n",
        text_col.iter().map(|s| s.len()).sum::<usize>() as f64 / text_col.len() as f64
    );

    // ========== 11. SURVEY DATA EXAMPLE ==========
    println!("=== 11. Survey Data with Open-Ended Responses ===\n");

    let survey_df = DataFrame::builder()
        .add_int("response_id", vec![1, 2, 3, 4])
        .add_categorical(
            "satisfaction",
            vec![
                "Very Satisfied".to_string(),
                "Satisfied".to_string(),
                "Neutral".to_string(),
                "Very Satisfied".to_string(),
            ],
        )
        .add_string(
            "feedback",
            vec![
                "Great product! Would recommend to others.".to_string(),
                "Good overall, but delivery was slow.".to_string(),
                "Average experience, nothing special.".to_string(),
                "Exceeded my expectations in every way!".to_string(),
            ],
        )
        .build()
        .unwrap();

    println!("Survey responses:");
    println!("{}\n", survey_df);

    // ========== 12. PRODUCT CATALOG EXAMPLE ==========
    println!("=== 12. Product Catalog with Descriptions ===\n");

    let products_df = DataFrame::builder()
        .add_int("product_id", vec![101, 102, 103])
        .add_string("name", vec![
            "Wireless Noise-Canceling Headphones".to_string(),
            "Ultra-Slim Laptop Stand".to_string(),
            "Ergonomic Wireless Mouse".to_string(),
        ])
        .add_string("description", vec![
            "Premium over-ear headphones with active noise cancellation and 30-hour battery life.".to_string(),
            "Adjustable aluminum stand for laptops up to 17 inches. Improves ergonomics and airflow.".to_string(),
            "Comfortable wireless mouse with precision tracking and customizable buttons.".to_string(),
        ])
        .add_column("price", vec![299.99, 49.99, 79.99])
        .add_bool("in_stock", vec![true, true, false])
        .build()
        .unwrap();

    println!("Product catalog:");
    println!("{}\n", products_df);

    // ========== 13. ADDRESS DATA EXAMPLE ==========
    println!("=== 13. Address Data (Common String Use Case) ===\n");

    let addresses_df = DataFrame::builder()
        .add_int("id", vec![1, 2, 3])
        .add_string(
            "street",
            vec![
                "123 Main Street".to_string(),
                "456 Oak Avenue, Apt 2B".to_string(),
                "789 Pine Road".to_string(),
            ],
        )
        .add_string(
            "city",
            vec![
                "New York".to_string(),
                "Los Angeles".to_string(),
                "Chicago".to_string(),
            ],
        )
        .add_string(
            "postal_code",
            vec![
                "10001".to_string(),
                "90001".to_string(),
                "60601".to_string(),
            ],
        )
        .build()
        .unwrap();

    println!("Address database:");
    println!("{}\n", addresses_df);

    // ========== 14. DATA QUALITY: MISSING AS EMPTY ==========
    println!("=== 14. Data Quality: Empty Strings ===\n");

    let quality_df = DataFrame::builder()
        .add_int("id", vec![1, 2, 3, 4])
        .add_string(
            "comment",
            vec![
                "Good".to_string(),
                "".to_string(), // Missing/empty
                "Excellent".to_string(),
                "".to_string(), // Missing/empty
            ],
        )
        .build()
        .unwrap();

    println!("Data with empty strings (representing missing):");
    println!("{}\n", quality_df);

    let comments = quality_df.get_string("comment").unwrap();
    let missing_count = comments.iter().filter(|s| s.is_empty()).count();
    println!("Quality check:");
    println!("  Total responses: {}", comments.len());
    println!("  Missing comments: {}", missing_count);
    println!("  Complete comments: {}\n", comments.len() - missing_count);

    // ========== 15. PERFORMANCE COMPARISON ==========
    println!("=== 15. Performance: String vs Categorical ===\n");

    println!("Memory considerations:");
    println!("  String columns:");
    println!("    • Each value stored as full text");
    println!("    • Variable memory per value");
    println!("    • Best for: unique values, free text");

    println!("\n  Categorical columns:");
    println!("    • Text stored once, indices stored per row");
    println!("    • Fixed memory per value (integer)");
    println!("    • Best for: repeated categories, groups\n");

    // ========== 16. COMBINING STRING OPERATIONS ==========
    println!("=== 16. Combining String with Other Types ===\n");

    let complete_df = DataFrame::builder()
        .add_int("id", vec![1, 2, 3])
        .add_string(
            "name",
            vec![
                "Alice".to_string(),
                "Bob".to_string(),
                "Charlie".to_string(),
            ],
        )
        .add_column("score", vec![85.5, 92.3, 78.9])
        .add_bool("passed", vec![true, true, true])
        .add_categorical(
            "grade",
            vec!["B".to_string(), "A".to_string(), "C".to_string()],
        )
        .build()
        .unwrap();

    println!("Complete mixed-type DataFrame:");
    println!("{}\n", complete_df);

    println!("Column types:");
    println!("  id: {:?}", complete_df.get_column("id").unwrap().dtype());
    println!(
        "  name: {:?}",
        complete_df.get_column("name").unwrap().dtype()
    );
    println!(
        "  score: {:?}",
        complete_df.get_column("score").unwrap().dtype()
    );
    println!(
        "  passed: {:?}",
        complete_df.get_column("passed").unwrap().dtype()
    );
    println!(
        "  grade: {:?}\n",
        complete_df.get_column("grade").unwrap().dtype()
    );

    // ========== SUMMARY ==========
    println!("=== FEATURE SUMMARY ===\n");

    println!("✅ String Column Support (v1.7.0):");
    println!("  • add_string(name, values) - Create string column");
    println!("  • get_string(name) - Access string data");
    println!("  • Free text storage (not encoded like Categorical)");
    println!("  • Variable-length text support");

    println!("\n✅ Operations:");
    println!("  • to_float() - Returns NaN (text can't convert)");
    println!("  • concat() - Combine datasets with strings");
    println!("  • filter() - Filter by any column");
    println!("  • select() - Extract specific columns");
    println!("  • head/tail - Preview data");

    println!("\n✅ Display:");
    println!("  • String columns auto-sized to content");
    println!("  • Variable-width formatting");
    println!("  • Mixed display with all other types");

    println!("\n✅ Export:");
    println!("  • to_csv() - Strings exported as-is");
    println!("  • to_json() - Strings exported as JSON strings");

    println!("\n✅ Use Cases:");
    println!("  • Customer data - Names, emails, addresses");
    println!("  • Research - Participant names, notes, comments");
    println!("  • Surveys - Open-ended responses");
    println!("  • Products - Names, descriptions, SKUs");
    println!("  • Documents - Titles, authors, abstracts");
    println!("  • Any free-form text data");

    println!("\n✅ String vs Categorical:");
    println!("  • STRING: Free text, unique values, variable length");
    println!("    Examples: names, emails, comments, addresses");
    println!("  • CATEGORICAL: Repeated categories, encoded as integers");
    println!("    Examples: gender, country, treatment group, grade");

    println!("\n✅ Missing Data:");
    println!("  • String has no NaN concept (like Int, Bool, DateTime)");
    println!("  • Empty strings (\"\") can represent missing text");
    println!("  • count_na() returns 0 for String columns");

    println!("\n✅ Type Conversion:");
    println!("  • String → Float: Returns NaN array");
    println!("  • String can't be used in numeric operations");
    println!("  • Use Categorical if you need numeric encoding");

    println!("\n✅ Integration:");
    println!("  • Works with all DataFrame methods");
    println!("  • Filter, select, concat, head, tail");
    println!("  • Display alongside Float, Int, Bool, DateTime, Categorical");

    println!("\n=== Demo Complete! ===");
}
