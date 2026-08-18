// Test binary boolean detection for arbitrary two-value columns
// Examples: ['casado', 'solteiro'], ['M', 'F'], ['aprovado', 'reprovado']

use greeners::DataFrame;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Binary Boolean Detection ===\n");

    // Create CSV with various binary columns
    let mut file = File::create("test_binary_bool.csv")?;
    writeln!(file, "id,estado_civil,sexo,aprovado,status,answer")?;
    writeln!(file, "1,casado,M,sim,ativo,yes")?;
    writeln!(file, "2,solteiro,F,não,inativo,no")?;
    writeln!(file, "3,casado,M,sim,ativo,yes")?;
    writeln!(file, "4,solteiro,F,sim,ativo,no")?;
    writeln!(file, "5,casado,M,não,inativo,yes")?;
    drop(file);

    println!("📊 Loading CSV with binary columns...\n");

    let df = DataFrame::from_csv("test_binary_bool.csv")?;

    println!(
        "✓ Successfully loaded {} rows x {} columns\n",
        df.n_rows(),
        df.n_cols()
    );

    // Display all column types
    println!("┌─────────────────────┬───────────────────────────┐");
    println!("│ Column              │ Type                      │");
    println!("├─────────────────────┼───────────────────────────┤");

    for col in df.column_names() {
        if let Ok(column) = df.get_column(&col) {
            println!("│ {:<19} │ {:?}", col, column.dtype());
        }
    }
    println!("└─────────────────────┴───────────────────────────┘\n");

    // Test 1: estado_civil (casado/solteiro)
    println!("Test 1: estado_civil ['casado', 'solteiro']");
    if let Ok(civil_col) = df.get_bool("estado_civil") {
        println!("   ✓ Correctly detected as Bool");
        println!(
            "   First 3 values: {:?}",
            &civil_col.as_slice().unwrap()[..3]
        );
        // Alphabetically: 'casado' < 'solteiro', so casado=false, solteiro=true
        println!("   Mapping: 'casado' → false, 'solteiro' → true");
    } else {
        println!("   ✗ FAILED: Not detected as Bool");
    }

    // Test 2: sexo (M/F)
    println!("\nTest 2: sexo ['M', 'F']");
    if let Ok(sexo_col) = df.get_bool("sexo") {
        println!("   ✓ Correctly detected as Bool");
        println!(
            "   First 3 values: {:?}",
            &sexo_col.as_slice().unwrap()[..3]
        );
        // Alphabetically: 'F' < 'M', so F=false, M=true
        println!("   Mapping: 'F' → false, 'M' → true");
    } else {
        println!("   ✗ FAILED: Not detected as Bool");
    }

    // Test 3: aprovado (sim/não)
    println!("\nTest 3: aprovado ['sim', 'não']");
    if let Ok(aprovado_col) = df.get_bool("aprovado") {
        println!("   ✓ Correctly detected as Bool");
        println!(
            "   First 3 values: {:?}",
            &aprovado_col.as_slice().unwrap()[..3]
        );
        println!("   Mapping: 'não' → false, 'sim' → true");
    } else {
        println!("   ✗ FAILED: Not detected as Bool");
    }

    // Test 4: status (ativo/inativo)
    println!("\nTest 4: status ['ativo', 'inativo']");
    if let Ok(status_col) = df.get_bool("status") {
        println!("   ✓ Correctly detected as Bool");
        println!(
            "   First 3 values: {:?}",
            &status_col.as_slice().unwrap()[..3]
        );
        println!("   Mapping: 'ativo' → false, 'inativo' → true");
    } else {
        println!("   ✗ FAILED: Not detected as Bool");
    }

    // Test 5: answer (yes/no) - should still work
    println!("\nTest 5: answer ['yes', 'no']");
    if let Ok(answer_col) = df.get_bool("answer") {
        println!("   ✓ Correctly detected as Bool");
        println!(
            "   First 3 values: {:?}",
            &answer_col.as_slice().unwrap()[..3]
        );
        println!("   Mapping: 'no' → false, 'yes' → true");
    } else {
        println!("   ✗ FAILED: Not detected as Bool");
    }

    // Count bool columns
    let mut bool_count = 0;
    for col in df.column_names() {
        if df.get_bool(&col).is_ok() {
            bool_count += 1;
        }
    }

    println!("\n==================================================");
    println!(
        "Summary: {}/{} columns detected as Bool",
        bool_count,
        df.n_cols() - 1
    ); // -1 for id

    if bool_count == 5 {
        println!("✅ ALL binary columns correctly detected!");
    } else {
        println!("⚠️  Some binary columns not detected as Bool");
    }

    // Clean up
    std::fs::remove_file("test_binary_bool.csv").ok();

    Ok(())
}
