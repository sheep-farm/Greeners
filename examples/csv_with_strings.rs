// Example demonstrating CSV loading with string columns
// Shows that from_csv() and from_csv_url() now automatically detect column types

use greeners::DataFrame;

fn main() {
    println!("=== CSV with Strings - Automatic Type Detection ===\n");

    // Load the real-world dataset (cattaneo2.csv) which contains mixed types
    println!("Loading cattaneo2.csv (4642 rows with numbers, strings, and booleans)...\n");

    match DataFrame::from_csv("examples/data/cattaneo2.csv") {
        Ok(df) => {
            println!(
                "✓ Successfully loaded {} rows x {} columns\n",
                df.n_rows(),
                df.n_cols()
            );

            // Show all column types
            println!("Column types:");
            println!("{:<15} | Type", "Column");
            println!("{:-<15}-+{:-<20}", "", "");

            for col in df.column_names() {
                if let Ok(column) = df.get_column(&col) {
                    println!("{:<15} | {:?}", col, column.dtype());
                }
            }

            // Example: Access categorical data
            println!("\n--- Sample Categorical Data ---");
            if let Ok(married_col) = df.get_categorical("mmarried") {
                println!("mmarried categories: {:?}", married_col.levels);
                println!("Value counts: {:?}", married_col.value_counts());
            }

            if let Ok(msmoke_col) = df.get_categorical("msmoke") {
                println!("\nmsmoke categories: {:?}", msmoke_col.levels);
            }

            // Example: Access boolean data
            println!("\n--- Sample Boolean Data ---");
            if let Ok(fbaby_col) = df.get_bool("fbaby") {
                let true_count = fbaby_col.iter().filter(|&&x| x).count();
                let false_count = fbaby_col.len() - true_count;
                println!("fbaby: {} true, {} false", true_count, false_count);
            }

            // Example: Access numeric data
            println!("\n--- Sample Numeric Data ---");
            if let Ok(bweight_col) = df.get("bweight") {
                let mean = bweight_col.mean().unwrap();
                let std = bweight_col.std(1.0);
                println!("bweight: mean={:.2}, std={:.2}", mean, std);
            }
        }
        Err(e) => {
            println!("✗ Error loading file: {}", e);
        }
    }

    println!("\n=== Type Detection Rules ===");
    println!("1. Float: All values parse as numbers with decimals");
    println!("2. Int: All values parse as integers");
    println!("3. Bool: All values are true/false/yes/no/1/0");
    println!("4. Categorical: < 50% unique string values (efficient encoding)");
    println!("5. String: >= 50% unique string values (free text)");
}
