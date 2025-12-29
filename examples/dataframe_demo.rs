use greeners::DataFrame;
use std::collections::HashMap;

fn main() {
    println!("=== DataFrame Demo - Greeners ===\n");

    // Create a sample DataFrame
    let df = DataFrame::builder()
        .add_column("id", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .add_column("age", vec![25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0])
        .add_column("salary", vec![50000.0, 60000.0, 75000.0, 80000.0, 90000.0, 95000.0, 100000.0, 110000.0])
        .add_column("experience", vec![2.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0, 30.0])
        .build()
        .unwrap();

    println!("Original DataFrame:");
    println!("{}\n", df);

    // Test info()
    println!("=== DataFrame Info ===");
    println!("{}", df.info());

    // Test head() and tail()
    println!("=== First 3 rows (head) ===");
    let head = df.head(3).unwrap();
    println!("{}\n", head);

    println!("=== Last 3 rows (tail) ===");
    let tail = df.tail(3).unwrap();
    println!("{}\n", tail);

    // Test statistics
    println!("=== Statistics ===");
    let means = df.mean();
    println!("Means: {:?}", means);

    let medians = df.median();
    println!("Medians: {:?}", medians);

    let stds = df.std();
    println!("Standard Deviations: {:?}\n", stds);

    // Test describe()
    println!("=== Describe ===");
    let stats = df.describe();
    for (col, col_stats) in &stats {
        println!("Column: {}", col);
        println!("  count: {}", col_stats.get("count").unwrap());
        println!("  mean: {:.2}", col_stats.get("mean").unwrap());
        println!("  std: {:.2}", col_stats.get("std").unwrap());
        println!("  min: {:.2}", col_stats.get("min").unwrap());
        println!("  median: {:.2}", col_stats.get("median").unwrap());
        println!("  max: {:.2}", col_stats.get("max").unwrap());
    }
    println!();

    // Test filter()
    println!("=== Filter (salary > 80000) ===");
    let filtered = df.filter(|row| {
        row.get("salary").map(|&v| v > 80000.0).unwrap_or(false)
    }).unwrap();
    println!("{}\n", filtered);

    // Test sort_by()
    println!("=== Sort by experience (descending) ===");
    let sorted = df.sort_by("experience", false).unwrap();
    println!("{}\n", sorted);

    // Test drop()
    println!("=== Drop 'id' column ===");
    let dropped = df.drop(&["id"]).unwrap();
    println!("{}\n", dropped);

    // Test drop_rows()
    println!("=== Drop rows 0, 2, 4 ===");
    let dropped_rows = df.drop_rows(&[0, 2, 4]).unwrap();
    println!("{}\n", dropped_rows);

    // Test rename()
    println!("=== Rename 'age' to 'years_old' ===");
    let mut rename_map = HashMap::new();
    rename_map.insert("age".to_string(), "years_old".to_string());
    let renamed = df.rename(&rename_map).unwrap();
    println!("{}\n", renamed);

    // Test export to CSV and JSON
    println!("=== Exporting ===");
    df.to_csv("/tmp/dataframe_demo.csv").unwrap();
    println!("Exported to CSV: /tmp/dataframe_demo.csv");

    df.to_json("/tmp/dataframe_demo.json").unwrap();
    println!("Exported to JSON: /tmp/dataframe_demo.json");

    println!("\n=== Demo Complete! ===");
}
