// Complete demonstration of Greeners v1.3.1 type detection system
// Shows all supported types and real-world usage scenarios

use greeners::{CovarianceType, DataFrame, Formula, OLS};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Greeners v1.3.1 - Complete Type Detection Demo          ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create a comprehensive CSV with all supported types
    create_demo_csv();

    println!("📊 Loading mixed-type CSV with automatic detection...\n");

    match DataFrame::from_csv("complete_demo.csv") {
        Ok(df) => {
            println!(
                "✓ Successfully loaded {} rows x {} columns\n",
                df.n_rows(),
                df.n_cols()
            );

            // Display all column types
            println!("┌─────────────────────────────────────────────────┐");
            println!("│ DETECTED COLUMN TYPES                           │");
            println!("├─────────────────────┬───────────────────────────┤");
            println!("│ Column              │ Type                      │");
            println!("├─────────────────────┼───────────────────────────┤");

            for col in df.column_names() {
                if let Ok(column) = df.get_column(&col) {
                    println!("│ {:<19} │ {:?}", col, column.dtype());
                }
            }
            println!("└─────────────────────┴───────────────────────────┘\n");

            // Demonstrate type-safe access
            println!("┌─────────────────────────────────────────────────┐");
            println!("│ TYPE-SAFE DATA ACCESS                           │");
            println!("└─────────────────────────────────────────────────┘\n");

            // Int access
            if let Ok(id_col) = df.get_int("user_id") {
                println!("🔢 Integer column:");
                println!("   user_id[0] = {} (type: i64)", id_col[0]);
            }

            // Float access
            if let Ok(amount_col) = df.get("amount") {
                println!("\n💰 Float column:");
                println!("   amount[0] = {:.2} (type: f64)", amount_col[0]);
            }

            // Boolean access
            if let Ok(active_col) = df.get_bool("is_active") {
                println!("\n✓ Boolean column:");
                println!("   is_active[0] = {} (type: bool)", active_col[0]);
            }

            // DateTime access
            if let Ok(datetime_col) = df.get_datetime("created_at") {
                println!("\n📅 DateTime column:");
                println!(
                    "   created_at[0] = {} (type: NaiveDateTime)",
                    datetime_col[0]
                );
            }

            // String access (region detected as String due to high uniqueness in small sample)
            if let Ok(region_col) = df.get_string("region") {
                println!("\n🗺️  String column:");
                println!("   region[0] = {}", region_col[0]);
            } else if let Ok(region_col) = df.get_categorical("region") {
                println!("\n🏷️  Categorical column:");
                println!("   region categories: {:?}", region_col.levels);
                println!("   region[0] = {:?}", region_col.get_string(0));
            }

            // String access
            if let Ok(email_col) = df.get_string("email") {
                println!("\n📧 String column:");
                println!("   email[0] = {}", email_col[0]);
            }

            // Demonstrate regression with mixed types
            println!("\n┌─────────────────────────────────────────────────┐");
            println!("│ REGRESSION WITH AUTO-DETECTED TYPES              │");
            println!("└─────────────────────────────────────────────────┘\n");

            // Use Float columns for regression (amount ~ score)
            let formula = Formula::parse("amount ~ score")?;
            let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;

            println!("Model: amount ~ score");
            println!("\nCoefficients:");
            println!("  Intercept: {:.4}", result.params[0]);
            println!("  score:     {:.4}", result.params[1]);
            println!("  R²:        {:.4}", result.r_squared);

            // Demonstrate filtering with mixed types
            println!("\n┌─────────────────────────────────────────────────┐");
            println!("│ FILTERING EXAMPLES                               │");
            println!("└─────────────────────────────────────────────────┘\n");

            if let Ok(active) = df.get_bool("is_active") {
                let active_count = active.iter().filter(|&&x| x).count();
                println!("Active users: {}/{}", active_count, df.n_rows());
            }

            if let Ok(region) = df.get_categorical("region") {
                let counts = region.value_counts();
                println!("\nUsers by region:");
                for (region_name, count) in counts.iter() {
                    println!("  {}: {}", region_name, count);
                }
            }
        }
        Err(e) => {
            println!("✗ Error: {}", e);
        }
    }

    // Clean up
    std::fs::remove_file("complete_demo.csv").ok();

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  KEY TAKEAWAYS                                            ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  ✅ 6 types auto-detected: Int, Float, Bool, DateTime,    ║");
    println!("║     Categorical, String                                   ║");
    println!("║  ✅ Type-safe access prevents runtime errors              ║");
    println!("║  ✅ Smart detection: 1.0 → Int, 1.5 → Float               ║");
    println!("║  ✅ Works with OLS, IV, Panel, and all estimators         ║");
    println!("║  ✅ Zero configuration required!                          ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    Ok(())
}

fn create_demo_csv() {
    let mut file = File::create("complete_demo.csv").expect("Failed to create file");

    // Write header
    writeln!(
        file,
        "user_id,email,amount,quantity,is_active,created_at,region,score"
    )
    .unwrap();

    // Write data rows with all types
    writeln!(
        file,
        "1,alice@example.com,150.50,10,true,2024-01-15 10:30:00,North,4.5"
    )
    .unwrap();
    writeln!(
        file,
        "2,bob@example.com,275.75,20,false,2024-01-16 14:45:00,South,3.8"
    )
    .unwrap();
    writeln!(
        file,
        "3,charlie@example.com,99.99,5,true,2024-01-17 09:15:00,North,4.2"
    )
    .unwrap();
    writeln!(
        file,
        "4,diana@example.com,450.00,30,true,2024-01-18 16:20:00,East,4.9"
    )
    .unwrap();
    writeln!(
        file,
        "5,eve@example.com,325.25,15,false,2024-01-19 11:30:00,North,3.5"
    )
    .unwrap();
}
