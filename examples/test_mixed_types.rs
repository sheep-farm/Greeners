// Example demonstrating automatic type detection in CSV/JSON loading
// Tests that the DataFrame correctly handles mixed types (floats, strings, categoricals)

use greeners::DataFrame;

fn main() {
    println!("=== Testing Automatic Type Detection ===\n");

    // Test 1: CSV with mixed types (cattaneo2.csv has numbers and strings)
    println!("1. Testing from_csv() with cattaneo2.csv (mixed types):");
    match DataFrame::from_csv("examples/data/cattaneo2.csv") {
        Ok(df) => {
            println!(
                "   ✓ SUCCESS: Loaded {} rows x {} columns",
                df.n_rows(),
                df.n_cols()
            );
            println!("   Columns: {:?}", df.column_names());

            // Display first few column types
            let cols = df.column_names();
            println!("\n   Column types (first 10):");
            // for (i, col) in cols.iter().take(10).enumerate() {
            for (i, col) in cols.iter().enumerate() {
                if let Ok(column) = df.get_column(col) {
                    println!("      {}. {} -> {:?}", i + 1, col, column.dtype());
                }
            }
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Test 2: CSV with only numbers (dataset.csv)
    println!("\n2. Testing from_csv() with dataset.csv (numeric only):");
    match DataFrame::from_csv("examples/data/dataset.csv") {
        Ok(df) => {
            println!(
                "   ✓ SUCCESS: Loaded {} rows x {} columns",
                df.n_rows(),
                df.n_cols()
            );
            println!("   Columns: {:?}", df.column_names());
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Test 3: Create a test CSV with mixed types
    println!("\n3. Testing from_csv() with custom mixed-type CSV:");
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("examples/data/test_mixed.csv").expect("Failed to create file");
    writeln!(file, "id,name,age,salary,active,region").unwrap();
    writeln!(file, "1,Alice,25,50000.50,true,North").unwrap();
    writeln!(file, "2,Bob,30,60000.75,false,South").unwrap();
    writeln!(file, "3,Charlie,35,70000.00,true,North").unwrap();
    writeln!(file, "4,Diana,28,55000.25,false,East").unwrap();
    writeln!(file, "5,Eve,32,65000.50,true,North").unwrap();

    match DataFrame::from_csv("examples/data/test_mixed.csv") {
        Ok(df) => {
            println!(
                "   ✓ SUCCESS: Loaded {} rows x {} columns",
                df.n_rows(),
                df.n_cols()
            );
            println!("   Columns: {:?}", df.column_names());

            println!("\n   Column types:");
            for col in df.column_names() {
                if let Ok(column) = df.get_column(&col) {
                    println!("      {} -> {:?}", col, column.dtype());
                }
            }

            // Test accessing different types
            println!("\n   Sample data:");
            if let Ok(id_col) = df.get_int("id") {
                println!("      First ID: {}", id_col[0]);
            }
            if let Ok(names) = df.get_categorical("name") {
                println!("      Names: {:?}", names.to_strings().get(0..2));
            }
            if let Ok(ages) = df.get_int("age") {
                println!("      First age: {}", ages[0]);
            }
            if let Ok(salaries) = df.get("salary") {
                println!("      First salary: {}", salaries[0]);
            }
            if let Ok(bools) = df.get_bool("active") {
                println!("      Active flags: {:?}", &bools.to_vec()[0..2]);
            }
            if let Ok(regions) = df.get_categorical("region") {
                println!("      Regions: {:?}", regions.to_strings().get(0..2));
            }
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Test 4: JSON with mixed types
    println!("\n4. Testing from_json() with mixed types:");
    let mut json_file =
        File::create("examples/data/test_mixed.json").expect("Failed to create file");
    writeln!(json_file, "[").unwrap();
    writeln!(
        json_file,
        r#"  {{"product": "Laptop", "price": 999.99, "in_stock": true, "quantity": 5}},"#
    )
    .unwrap();
    writeln!(
        json_file,
        r#"  {{"product": "Mouse", "price": 29.99, "in_stock": false, "quantity": 0}},"#
    )
    .unwrap();
    writeln!(
        json_file,
        r#"  {{"product": "Keyboard", "price": 79.99, "in_stock": true, "quantity": 15}}"#
    )
    .unwrap();
    writeln!(json_file, "]").unwrap();

    match DataFrame::from_json("examples/data/test_mixed.json") {
        Ok(df) => {
            println!(
                "   ✓ SUCCESS: Loaded {} rows x {} columns",
                df.n_rows(),
                df.n_cols()
            );
            println!("   Columns: {:?}", df.column_names());

            println!("\n   Column types:");
            for col in df.column_names() {
                if let Ok(column) = df.get_column(&col) {
                    println!("      {} -> {:?}", col, column.dtype());
                }
            }
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Clean up
    println!("\n=== Cleaning up test files ===");
    std::fs::remove_file("examples/data/test_mixed.csv").ok();
    std::fs::remove_file("examples/data/test_mixed.json").ok();
    println!("Test files removed.\n");

    println!("=== All tests completed! ===");
}
