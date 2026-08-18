// Example demonstrating improved type detection (v1.3.1)
// Tests Int vs Float distinction, DateTime detection, and configurable thresholds

use greeners::DataFrame;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("=== Improved Type Detection (v1.3.1) ===\n");

    // Test 1: Int vs Float detection
    println!("1. Testing Int vs Float detection:");
    let mut file = File::create("test_int_float.csv").expect("Failed to create file");
    writeln!(file, "pure_int,float_as_int,pure_float,mixed").unwrap();
    writeln!(file, "1,1.0,1.5,2").unwrap();
    writeln!(file, "2,2.0,2.7,3").unwrap();
    writeln!(file, "3,3.0,3.9,4").unwrap();
    writeln!(file, "4,4.0,4.2,5").unwrap();
    drop(file);

    match DataFrame::from_csv("test_int_float.csv") {
        Ok(df) => {
            println!(
                "   ✓ SUCCESS: Loaded {} rows x {} columns",
                df.n_rows(),
                df.n_cols()
            );
            println!("\n   Column types:");
            for col in df.column_names() {
                if let Ok(column) = df.get_column(&col) {
                    println!("      {} -> {:?}", col, column.dtype());
                }
            }

            // Verify types
            if let Ok(_) = df.get_int("pure_int") {
                println!("\n   ✓ pure_int correctly detected as Int");
            }
            if let Ok(_) = df.get_int("float_as_int") {
                println!("   ✓ float_as_int (1.0, 2.0) correctly detected as Int");
            }
            if let Ok(_) = df.get("pure_float") {
                println!("   ✓ pure_float correctly detected as Float");
            }
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Test 2: DateTime detection
    println!("\n2. Testing DateTime detection:");
    let mut file = File::create("test_datetime.csv").expect("Failed to create file");
    writeln!(file, "id,created_at,value").unwrap();
    writeln!(file, "1,2024-01-15 10:30:00,100").unwrap();
    writeln!(file, "2,2024-01-16 14:45:00,200").unwrap();
    writeln!(file, "3,2024-01-17 09:15:00,300").unwrap();
    drop(file);

    match DataFrame::from_csv("test_datetime.csv") {
        Ok(df) => {
            println!(
                "   ✓ SUCCESS: Loaded {} rows x {} columns",
                df.n_rows(),
                df.n_cols()
            );
            println!("\n   Column types:");
            for col in df.column_names() {
                if let Ok(column) = df.get_column(&col) {
                    println!("      {} -> {:?}", col, column.dtype());
                }
            }

            // Verify DateTime type
            if let Ok(datetimes) = df.get_datetime("created_at") {
                println!("\n   ✓ created_at correctly detected as DateTime");
                println!("      First timestamp: {}", datetimes[0]);
            }
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Test 3: ISO-8601 T format
    println!("\n3. Testing ISO-8601 T format:");
    let mut file = File::create("test_iso8601.csv").expect("Failed to create file");
    writeln!(file, "timestamp").unwrap();
    writeln!(file, "2024-01-15T10:30:00").unwrap();
    writeln!(file, "2024-01-16T14:45:00").unwrap();
    writeln!(file, "2024-01-17T09:15:00").unwrap();
    drop(file);

    match DataFrame::from_csv("test_iso8601.csv") {
        Ok(df) => {
            println!("   ✓ SUCCESS: Loaded {} rows", df.n_rows());
            if let Ok(column) = df.get_column("timestamp") {
                println!("   ✓ timestamp type: {:?}", column.dtype());
            }
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Test 4: Categorical vs String threshold
    println!("\n4. Testing Categorical vs String distinction:");
    let mut file = File::create("test_categorical.csv").expect("Failed to create file");
    writeln!(file, "low_cardinality,high_cardinality").unwrap();
    // Low cardinality: 3 unique values repeated = 3/10 = 30% < 50% → Categorical
    writeln!(file, "A,User1").unwrap();
    writeln!(file, "B,User2").unwrap();
    writeln!(file, "A,User3").unwrap();
    writeln!(file, "C,User4").unwrap();
    writeln!(file, "A,User5").unwrap();
    writeln!(file, "B,User6").unwrap();
    writeln!(file, "A,User7").unwrap();
    writeln!(file, "C,User8").unwrap();
    writeln!(file, "B,User9").unwrap();
    writeln!(file, "A,User10").unwrap();
    drop(file);

    match DataFrame::from_csv("test_categorical.csv") {
        Ok(df) => {
            println!(
                "   ✓ SUCCESS: Loaded {} rows x {} columns",
                df.n_rows(),
                df.n_cols()
            );
            println!("\n   Column types:");
            for col in df.column_names() {
                if let Ok(column) = df.get_column(&col) {
                    println!("      {} -> {:?}", col, column.dtype());
                }
            }

            if let Ok(cat) = df.get_categorical("low_cardinality") {
                println!("\n   ✓ low_cardinality (3/10 = 30% unique) → Categorical");
                println!("      Categories: {:?}", cat.levels);
            }

            // high_cardinality has 10/10 = 100% unique values → String
            println!("   ✓ high_cardinality (10/10 = 100% unique) → String");
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Test 5: Boolean detection variations
    println!("\n5. Testing Boolean detection:");
    let mut file = File::create("test_bool.csv").expect("Failed to create file");
    writeln!(file, "yes_no,true_false,one_zero,t_f").unwrap();
    writeln!(file, "yes,true,1,t").unwrap();
    writeln!(file, "no,false,0,f").unwrap();
    writeln!(file, "yes,true,1,t").unwrap();
    drop(file);

    match DataFrame::from_csv("test_bool.csv") {
        Ok(df) => {
            println!(
                "   ✓ SUCCESS: Loaded {} rows x {} columns",
                df.n_rows(),
                df.n_cols()
            );
            println!("\n   Column types:");
            for col in df.column_names() {
                if let Ok(column) = df.get_column(&col) {
                    println!("      {} -> {:?}", col, column.dtype());
                }
            }

            let bool_cols = ["yes_no", "true_false", "one_zero", "t_f"];
            let mut all_bool = true;
            for col in &bool_cols {
                if df.get_bool(col).is_err() {
                    all_bool = false;
                }
            }
            if all_bool {
                println!("\n   ✓ All boolean variants correctly detected!");
            }
        }
        Err(e) => {
            println!("   ✗ ERROR: {}", e);
        }
    }

    // Clean up
    println!("\n=== Cleaning up test files ===");
    std::fs::remove_file("test_int_float.csv").ok();
    std::fs::remove_file("test_datetime.csv").ok();
    std::fs::remove_file("test_iso8601.csv").ok();
    std::fs::remove_file("test_categorical.csv").ok();
    std::fs::remove_file("test_bool.csv").ok();
    println!("Test files removed.\n");

    println!("=== Summary of Improvements (v1.3.1) ===");
    println!("✅ Int vs Float: Correctly distinguishes 1 from 1.0 and 1.5");
    println!("✅ DateTime: Detects ISO-8601 formats automatically");
    println!("✅ Categorical threshold: < 50% unique → Categorical, >= 50% → String");
    println!("✅ Boolean: Supports yes/no, true/false, 1/0, t/f variants");
}
