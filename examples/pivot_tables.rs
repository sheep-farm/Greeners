use greeners::DataFrame;

fn main() {
    println!("=== PIVOT TABLES - Reshape Data ===\n");

    // ========== PIVOT TABLE (Long to Wide) ==========
    println!("=== PIVOT TABLE - Long to Wide Format ===\n");

    // Sales data in LONG format (typical database format)
    let sales_long = DataFrame::builder()
        .add_column("date", vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        .add_column("product", vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        .add_column("region", vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        .add_column("sales", vec![100.0, 150.0, 200.0, 120.0, 180.0, 220.0, 110.0, 160.0, 210.0])
        .build()
        .unwrap();

    println!("Original data (LONG format):");
    println!("{}\n", sales_long);

    // Pivot: Products as rows, Regions as columns
    println!("--- Pivot: Products (rows) × Regions (columns) ---");
    let pivot1 = sales_long
        .pivot_table("product", "region", "sales", "sum")
        .unwrap();
    println!("Total sales by product and region:");
    println!("{}\n", pivot1);

    // Pivot: Dates as rows, Products as columns
    println!("--- Pivot: Dates (rows) × Products (columns) ---");
    let pivot2 = sales_long
        .pivot_table("date", "product", "sales", "sum")
        .unwrap();
    println!("Total sales by date and product:");
    println!("{}\n", pivot2);

    // Different aggregations
    println!("--- Pivot with MEAN aggregation ---");
    let pivot_mean = sales_long
        .pivot_table("product", "region", "sales", "mean")
        .unwrap();
    println!("Average sales by product and region:");
    println!("{}\n", pivot_mean);

    println!("--- Pivot with COUNT aggregation ---");
    let pivot_count = sales_long
        .pivot_table("product", "region", "sales", "count")
        .unwrap();
    println!("Number of transactions by product and region:");
    println!("{}\n", pivot_count);

    println!("--- Pivot with MAX aggregation ---");
    let pivot_max = sales_long
        .pivot_table("date", "product", "sales", "max")
        .unwrap();
    println!("Maximum sales by date and product:");
    println!("{}\n", pivot_max);

    // ========== MELT (Wide to Long) ==========
    println!("\n=== MELT - Wide to Long Format ===\n");

    // Monthly revenue in WIDE format (spreadsheet style)
    let revenue_wide = DataFrame::builder()
        .add_column("product", vec![1.0, 2.0, 3.0])
        .add_column("Jan", vec![100.0, 150.0, 200.0])
        .add_column("Feb", vec![120.0, 160.0, 210.0])
        .add_column("Mar", vec![110.0, 170.0, 220.0])
        .build()
        .unwrap();

    println!("Original data (WIDE format - spreadsheet style):");
    println!("{}\n", revenue_wide);

    // Melt to long format
    println!("--- After MELT (long format) ---");
    let revenue_long = revenue_wide
        .melt(&["product"], None, "month", "revenue")
        .unwrap();
    println!("Melted data (better for analysis):");
    println!("{}\n", revenue_long);

    // ========== REAL-WORLD EXAMPLE ==========
    println!("\n=== REAL-WORLD EXAMPLE - Store Performance ===\n");

    // Daily sales by store and department
    let store_sales = DataFrame::builder()
        .add_column("store", vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        .add_column("department", vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        .add_column("revenue", vec![1000.0, 1500.0, 2000.0, 1200.0, 1600.0, 2200.0, 900.0, 1400.0, 1900.0])
        .add_column("customers", vec![50.0, 75.0, 100.0, 60.0, 80.0, 110.0, 45.0, 70.0, 95.0])
        .build()
        .unwrap();

    println!("Store performance data:");
    println!("{}\n", store_sales);

    // Analysis 1: Revenue by store and department
    println!("--- Revenue Matrix: Stores × Departments ---");
    let revenue_matrix = store_sales
        .pivot_table("store", "department", "revenue", "sum")
        .unwrap();
    println!("{}\n", revenue_matrix);

    // Analysis 2: Customer traffic by store and department
    println!("--- Customer Matrix: Stores × Departments ---");
    let customer_matrix = store_sales
        .pivot_table("store", "department", "customers", "sum")
        .unwrap();
    println!("{}\n", customer_matrix);

    // Analysis 3: Average revenue per customer
    println!("--- Average Revenue per Customer: Stores × Departments ---");
    let avg_per_customer = store_sales
        .pivot_table("store", "department", "revenue", "mean")
        .unwrap();
    println!("{}\n", avg_per_customer);

    // ========== PIVOT vs MELT ROUNDTRIP ==========
    println!("\n=== ROUNDTRIP: Pivot → Melt → Original ===\n");

    println!("1. Start with LONG format:");
    let original = DataFrame::builder()
        .add_column("id", vec![1.0, 1.0, 2.0, 2.0])
        .add_column("category", vec![1.0, 2.0, 1.0, 2.0])
        .add_column("value", vec![10.0, 20.0, 30.0, 40.0])
        .build()
        .unwrap();
    println!("{}\n", original);

    println!("2. Pivot to WIDE format:");
    let pivoted = original
        .pivot_table("id", "category", "value", "sum")
        .unwrap();
    println!("{}\n", pivoted);

    println!("3. Melt back to LONG format:");
    let melted = pivoted.melt(&["id"], None, "category", "value").unwrap();
    println!("{}\n", melted);

    // ========== USE CASES ==========
    println!("\n=== COMMON USE CASES ===\n");

    println!("✅ PIVOT TABLE:");
    println!("  - Sales reports (products × regions)");
    println!("  - Time series analysis (dates × metrics)");
    println!("  - Cross-tabulation (demographics × behavior)");
    println!("  - Dashboard summaries");
    println!("  - Excel-style pivot tables");

    println!("\n✅ MELT:");
    println!("  - Prepare spreadsheet data for analysis");
    println!("  - Normalize denormalized data");
    println!("  - Convert wide to long for plotting");
    println!("  - Database import preparation");
    println!("  - Time series from wide format");

    println!("\n=== AGGREGATION FUNCTIONS AVAILABLE ===");
    println!("  • sum     - Total");
    println!("  • mean    - Average");
    println!("  • count   - Number of records");
    println!("  • min     - Minimum value");
    println!("  • max     - Maximum value");
    println!("  • median  - Middle value");

    println!("\n=== Demo Complete! ===");
}
