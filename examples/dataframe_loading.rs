// Example demonstrating various ways to load data into a DataFrame
//
// This example shows:
// 1. CSV from local file
// 2. CSV from URL
// 3. JSON from local file
// 4. JSON from URL
// 5. Builder pattern
// 6. Direct construction from HashMap

use greeners::{DataFrame, Formula};
use ndarray::Array1;
use std::collections::HashMap;

fn main() {
    println!("=== DataFrame Loading Examples ===\n");

    // Method 1: Direct construction from HashMap
    println!("1. Direct construction from HashMap:");
    let mut data = HashMap::new();
    data.insert("x".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0]));
    data.insert("y".to_string(), Array1::from(vec![2.0, 4.0, 6.0, 8.0]));

    let df1 = DataFrame::new(data).expect("Failed to create DataFrame");
    println!("   Loaded {} rows x {} columns", df1.n_rows(), df1.n_cols());
    println!("   Columns: {:?}\n", df1.column_names());

    // Method 2: Builder pattern (most convenient for simple cases)
    println!("2. Builder pattern:");
    let df2 = DataFrame::builder()
        .add_column("price", vec![100.0, 150.0, 200.0])
        .add_column("quantity", vec![10.0, 20.0, 15.0])
        .add_column("profit", vec![50.0, 75.0, 100.0])
        .build()
        .expect("Failed to build DataFrame");

    println!("   Loaded {} rows x {} columns", df2.n_rows(), df2.n_cols());
    println!("   Columns: {:?}\n", df2.column_names());

    // Method 3: CSV from local file
    println!("3. CSV from local file:");
    println!("   Creating a test CSV file...");

    // Create a test CSV file
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create("test_data.csv").expect("Failed to create file");
    writeln!(file, "income,education,experience").unwrap();
    writeln!(file, "50000,16,5").unwrap();
    writeln!(file, "60000,18,7").unwrap();
    writeln!(file, "55000,16,6").unwrap();
    writeln!(file, "70000,20,10").unwrap();

    let df3 = DataFrame::from_csv("test_data.csv").expect("Failed to read CSV");
    println!("   Loaded {} rows x {} columns", df3.n_rows(), df3.n_cols());
    println!("   Columns: {:?}\n", df3.column_names());

    // Method 4: JSON from local file (column-oriented)
    println!("4. JSON from local file (column-oriented):");
    println!("   Creating a test JSON file...");

    let mut json_file = File::create("test_data_columns.json").expect("Failed to create file");
    writeln!(json_file, "{{").unwrap();
    writeln!(json_file, r#"  "age": [25, 30, 35, 40],"#).unwrap();
    writeln!(json_file, r#"  "salary": [40000, 50000, 60000, 70000],"#).unwrap();
    writeln!(json_file, r#"  "years_exp": [2, 5, 8, 12]"#).unwrap();
    writeln!(json_file, "}}").unwrap();

    let df4 = DataFrame::from_json("test_data_columns.json").expect("Failed to read JSON");
    println!("   Loaded {} rows x {} columns", df4.n_rows(), df4.n_cols());
    println!("   Columns: {:?}\n", df4.column_names());

    // Method 5: JSON from local file (record-oriented)
    println!("5. JSON from local file (record-oriented):");
    println!("   Creating a test JSON file...");

    let mut json_file2 = File::create("test_data_records.json").expect("Failed to create file");
    writeln!(json_file2, "[").unwrap();
    writeln!(json_file2, r#"  {{"height": 170.0, "weight": 70.0}},"#).unwrap();
    writeln!(json_file2, r#"  {{"height": 175.0, "weight": 75.0}},"#).unwrap();
    writeln!(json_file2, r#"  {{"height": 180.0, "weight": 80.0}}"#).unwrap();
    writeln!(json_file2, "]").unwrap();

    let df5 = DataFrame::from_json("test_data_records.json").expect("Failed to read JSON");
    println!("   Loaded {} rows x {} columns", df5.n_rows(), df5.n_cols());
    println!("   Columns: {:?}\n", df5.column_names());

    // Method 6: CSV from URL (example - requires internet connection)
    println!("6. CSV from URL:");
    println!("   Note: This requires internet connection and a valid CSV URL");
    println!("   Example usage:");
    println!(
        r#"   let df = DataFrame::from_csv_url("https://github.com/sheep-farm/Greeners/blob/main/data.csv")?;"#
    );
    println!();

    // Uncomment to test with a real URL:
    let df6 = DataFrame::from_csv_url(
        "https://raw.githubusercontent.com/sheep-farm/Greeners/refs/heads/main/data.csv",
    )
    .unwrap();
    println!("   Loaded {} rows x {} columns", df6.n_rows(), df6.n_cols());

    // Method 7: JSON from URL (example - requires internet connection)
    println!("7. JSON from URL:");
    println!("   Note: This requires internet connection and a valid JSON URL");
    println!("   Example usage:");
    println!(
        r#"   let df = DataFrame::from_json_url("https://raw.githubusercontent.com/sheep-farm/Greeners/refs/heads/main/data.json")?;"#
    );
    println!();

    // Bonus: Using the DataFrame with formulas
    println!("\n=== Bonus: Using DataFrame with OLS regression ===\n");

    let df_regression = DataFrame::builder()
        .add_column("wage", vec![30000.0, 40000.0, 50000.0, 60000.0, 70000.0])
        .add_column("education", vec![12.0, 14.0, 16.0, 18.0, 20.0])
        .add_column("experience", vec![2.0, 4.0, 6.0, 8.0, 10.0])
        .build()
        .expect("Failed to build DataFrame");

    let formula = Formula::parse("wage ~ education + experience").expect("Failed to parse formula");
    let (y, x) = df_regression
        .to_design_matrix(&formula)
        .expect("Failed to create design matrix");

    println!(
        "Design matrix shape: {} rows x {} columns",
        x.nrows(),
        x.ncols()
    );
    println!("Response vector length: {}", y.len());

    // You can now use this with OLS or any other model:
    // use greeners::OLS;
    // let model = OLS::fit(y, x)?;
    // model.summary();

    // Clean up test files
    println!("\n=== Cleaning up test files ===");
    std::fs::remove_file("test_data.csv").ok();
    std::fs::remove_file("test_data_columns.json").ok();
    std::fs::remove_file("test_data_records.json").ok();
    println!("Test files removed.");
}
