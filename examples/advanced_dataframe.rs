use greeners::DataFrame;

fn main() {
    println!("=== Advanced DataFrame Features Demo ===\n");

    // Create sample data
    let df = DataFrame::builder()
        .add_column("id", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        .add_column("age", vec![25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0])
        .add_column("income", vec![30000.0, 45000.0, 60000.0, 75000.0, 90000.0, 105000.0, 120000.0, 135000.0, 150000.0, 165000.0])
        .add_column("score", vec![65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0])
        .build()
        .unwrap();

    println!("Original DataFrame:");
    println!("{}\n", df);

    // 1. SELECT - Select specific columns
    println!("=== 1. SELECT - Choose specific columns ===");
    let selected = df.select(&["age", "income"]).unwrap();
    println!("{}\n", selected);

    // 2. ILOC - Index-based selection
    println!("=== 2. ILOC - Index-based selection ===");
    println!("Rows 0, 2, 4 and all columns:");
    let iloc_rows = df.iloc(Some(&[0, 2, 4]), None).unwrap();
    println!("{}\n", iloc_rows);

    println!("All rows, columns 0 and 2:");
    let iloc_cols = df.iloc(None, Some(&[0, 2])).unwrap();
    println!("{}\n", iloc_cols);

    println!("Rows 1-3, columns 1-2:");
    let iloc_both = df.iloc(Some(&[1, 2, 3]), Some(&[1, 2])).unwrap();
    println!("{}\n", iloc_both);

    // 3. CONCAT - Concatenate DataFrames
    println!("=== 3. CONCAT - Combine DataFrames vertically ===");
    let df2 = DataFrame::builder()
        .add_column("id", vec![11.0, 12.0])
        .add_column("age", vec![75.0, 80.0])
        .add_column("income", vec![180000.0, 195000.0])
        .add_column("score", vec![115.0, 120.0])
        .build()
        .unwrap();

    let combined = df.concat(&df2).unwrap();
    println!("Combined DataFrame ({} rows):", combined.n_rows());
    println!("{}\n", combined);

    // 4. APPLY - Apply function to all columns
    println!("=== 4. APPLY - Transform all columns ===");
    println!("Normalize all values (divide by 10):");
    let normalized = df.apply(|col| col.mapv(|v| v / 10.0)).unwrap();
    println!("{}\n", normalized);

    // 5. MAP_COLUMN - Transform specific column
    println!("=== 5. MAP_COLUMN - Transform specific column ===");
    println!("Convert income to thousands:");
    let income_k = df.map_column("income", |v| v / 1000.0).unwrap();
    println!("{}\n", income_k);

    println!("Square all scores:");
    let squared_score = df.map_column("score", |v| v * v).unwrap();
    println!("{}\n", squared_score);

    // 6. CORR - Correlation matrix
    println!("=== 6. CORR - Correlation matrix ===");
    let corr = df.corr().unwrap();
    println!("Correlation matrix shape: {:?}", corr.shape());
    println!("Correlation matrix:");
    println!("{:.3}\n", corr);

    // 7. COV - Covariance matrix
    println!("=== 7. COV - Covariance matrix ===");
    let cov = df.cov().unwrap();
    println!("Covariance matrix shape: {:?}", cov.shape());
    println!("Covariance matrix:");
    println!("{:.1}\n", cov);

    // 8. SAMPLE - Random sampling
    println!("=== 8. SAMPLE - Random sampling ===");
    println!("Random sample of 5 rows:");
    let sample = df.sample(5).unwrap();
    println!("{}\n", sample);

    // Combining operations
    println!("=== CHAINING OPERATIONS ===");
    println!("1. Select age and income");
    println!("2. Filter where income > 75000");
    println!("3. Take first 3 rows");

    let result = df
        .select(&["age", "income"])
        .unwrap()
        .filter(|row| row.get("income").map(|&v| v > 75000.0).unwrap_or(false))
        .unwrap()
        .head(3)
        .unwrap();

    println!("{}\n", result);

    // Statistical analysis with correlation
    println!("=== STATISTICAL INSIGHTS ===");
    let corr_matrix = df.corr().unwrap();
    println!("Strong correlations detected:");
    println!("- age vs income: {:.3}", corr_matrix[[1, 2]]);
    println!("- age vs score: {:.3}", corr_matrix[[1, 3]]);
    println!("- income vs score: {:.3}", corr_matrix[[2, 3]]);

    println!("\n=== Demo Complete! ===");
}
