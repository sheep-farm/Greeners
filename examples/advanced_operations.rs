use greeners::DataFrame;
use std::collections::HashMap;

fn main() {
    println!("=== Advanced DataFrame Operations ===\n");
    println!("Demonstrating: append_row, merge (SQL joins), and groupby\n");

    // ========== 1. APPEND_ROW ==========
    println!("=== 1. APPEND_ROW - Add individual rows ===\n");

    let mut sales = DataFrame::builder()
        .add_column("date", vec![1.0, 2.0, 3.0])
        .add_column("product", vec![1.0, 2.0, 1.0])
        .add_column("revenue", vec![100.0, 150.0, 120.0])
        .build()
        .unwrap();

    println!("Original sales data:");
    println!("{}\n", sales);

    // Add new row
    let mut new_sale = HashMap::new();
    new_sale.insert("date".to_string(), 4.0);
    new_sale.insert("product".to_string(), 3.0);
    new_sale.insert("revenue".to_string(), 200.0);

    sales = sales.append_row(&new_sale).unwrap();
    println!("After appending new sale:");
    println!("{}\n", sales);

    // ========== 2. MERGE (SQL JOINS) ==========
    println!("=== 2. MERGE - SQL-style joins ===\n");

    // Create two related datasets
    let customers = DataFrame::builder()
        .add_column("customer_id", vec![1.0, 2.0, 3.0, 4.0])
        .add_column("age", vec![25.0, 30.0, 35.0, 40.0])
        .add_column("region", vec![1.0, 1.0, 2.0, 2.0])
        .build()
        .unwrap();

    let orders = DataFrame::builder()
        .add_column("customer_id", vec![2.0, 3.0, 3.0, 5.0])
        .add_column("order_value", vec![100.0, 150.0, 200.0, 300.0])
        .add_column("quantity", vec![2.0, 3.0, 4.0, 5.0])
        .build()
        .unwrap();

    println!("Customers:");
    println!("{}\n", customers);

    println!("Orders:");
    println!("{}\n", orders);

    // INNER JOIN - Only matching customers
    println!("--- INNER JOIN ---");
    println!("Only customers who have placed orders:");
    let inner = customers.merge(&orders, "customer_id", "inner").unwrap();
    println!("{}\n", inner);

    // LEFT JOIN - All customers, with orders if available
    println!("--- LEFT JOIN ---");
    println!("All customers, with order info (NaN if no orders):");
    let left = customers.merge(&orders, "customer_id", "left").unwrap();
    println!("{}\n", left);

    // RIGHT JOIN - All orders, with customer info if available
    println!("--- RIGHT JOIN ---");
    println!("All orders, with customer info (NaN if customer not found):");
    let right = customers.merge(&orders, "customer_id", "right").unwrap();
    println!("{}\n", right);

    // OUTER JOIN - Everything
    println!("--- OUTER JOIN ---");
    println!("All customers and all orders (NaN where no match):");
    let outer = customers.merge(&orders, "customer_id", "outer").unwrap();
    println!("{}\n", outer);

    // ========== 3. GROUPBY ==========
    println!("=== 3. GROUPBY - Aggregations ===\n");

    let transactions = DataFrame::builder()
        .add_column("category", vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
        .add_column("region", vec![1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0])
        .add_column(
            "revenue",
            vec![100.0, 150.0, 200.0, 300.0, 250.0, 400.0, 500.0, 350.0],
        )
        .add_column(
            "quantity",
            vec![5.0, 7.0, 10.0, 15.0, 12.0, 20.0, 25.0, 18.0],
        )
        .build()
        .unwrap();

    println!("Transaction data:");
    println!("{}\n", transactions);

    // Group by category and sum revenue
    println!("--- Total revenue by category ---");
    let by_category = transactions
        .groupby(&["category"], "revenue", "sum")
        .unwrap();
    println!("{}\n", by_category);

    // Group by category and count transactions
    println!("--- Transaction count by category ---");
    let count_by_cat = transactions
        .groupby(&["category"], "revenue", "count")
        .unwrap();
    println!("{}\n", count_by_cat);

    // Group by category and calculate mean
    println!("--- Average revenue by category ---");
    let avg_by_cat = transactions
        .groupby(&["category"], "revenue", "mean")
        .unwrap();
    println!("{}\n", avg_by_cat);

    // Group by region
    println!("--- Total revenue by region ---");
    let by_region = transactions.groupby(&["region"], "revenue", "sum").unwrap();
    println!("{}\n", by_region);

    // Group by multiple columns
    println!("--- Total revenue by category AND region ---");
    let by_both = transactions
        .groupby(&["category", "region"], "revenue", "sum")
        .unwrap();
    println!("{}\n", by_both);

    // Other aggregations
    println!("--- Maximum revenue by category ---");
    let max_by_cat = transactions
        .groupby(&["category"], "revenue", "max")
        .unwrap();
    println!("{}\n", max_by_cat);

    println!("--- Minimum revenue by category ---");
    let min_by_cat = transactions
        .groupby(&["category"], "revenue", "min")
        .unwrap();
    println!("{}\n", min_by_cat);

    println!("--- Median revenue by category ---");
    let median_by_cat = transactions
        .groupby(&["category"], "revenue", "median")
        .unwrap();
    println!("{}\n", median_by_cat);

    // ========== 4. COMBINED WORKFLOW ==========
    println!("=== 4. REAL-WORLD WORKFLOW - Combining Operations ===\n");

    println!("Scenario: Analyze customer orders by region\n");

    // Step 1: Join customers and orders
    let customer_orders = customers.merge(&orders, "customer_id", "inner").unwrap();
    println!("Step 1: Join customers with their orders");
    println!("{}\n", customer_orders);

    // Step 2: Group by region and calculate metrics
    let region_analysis = customer_orders
        .groupby(&["region"], "order_value", "sum")
        .unwrap();
    println!("Step 2: Total order value by region");
    println!("{}\n", region_analysis);

    let region_count = customer_orders
        .groupby(&["region"], "order_value", "count")
        .unwrap();
    println!("Step 3: Number of orders by region");
    println!("{}\n", region_count);

    let region_avg = customer_orders
        .groupby(&["region"], "order_value", "mean")
        .unwrap();
    println!("Step 4: Average order value by region");
    println!("{}\n", region_avg);

    // Step 5: Add a new high-value order and recalculate
    let mut high_value_order = HashMap::new();
    high_value_order.insert("customer_id".to_string(), 1.0);
    high_value_order.insert("order_value".to_string(), 500.0);
    high_value_order.insert("quantity".to_string(), 10.0);

    let orders_updated = orders.append_row(&high_value_order).unwrap();
    let updated_analysis = customers
        .merge(&orders_updated, "customer_id", "inner")
        .unwrap()
        .groupby(&["region"], "order_value", "sum")
        .unwrap();

    println!("Step 5: After adding high-value order");
    println!("{}\n", updated_analysis);

    println!("=== Summary of Capabilities ===");
    println!("✅ append_row: Add individual records dynamically");
    println!("✅ merge: 4 join types (inner, left, right, outer)");
    println!("✅ groupby: 6 aggregations (sum, mean, count, min, max, median)");
    println!("✅ Multi-column grouping supported");
    println!("✅ Chain operations for complex workflows");

    println!("\n=== Demo Complete! ===");
}
