use greeners::DataFrame;

fn main() {
    println!("=== TIME SERIES & ADVANCED FEATURES ===\n");
    println!(
        "Demonstrating: rolling, cumulative, shift, quantile, rank, drop_duplicates, interpolate\n"
    );

    // ========== 1. ROLLING WINDOW FUNCTIONS ==========
    println!("=== 1. ROLLING - Moving Averages & Window Aggregations ===\n");

    let stock_prices = DataFrame::builder()
        .add_column(
            "day",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .add_column(
            "price",
            vec![
                100.0, 102.0, 98.0, 105.0, 110.0, 108.0, 115.0, 112.0, 118.0, 120.0,
            ],
        )
        .add_column(
            "volume",
            vec![
                1000.0, 1200.0, 900.0, 1500.0, 1800.0, 1600.0, 2000.0, 1700.0, 2100.0, 2200.0,
            ],
        )
        .build()
        .unwrap();

    println!("Stock prices (10 days):");
    println!("{}\n", stock_prices);

    // Rolling mean (3-day moving average)
    println!("--- 3-Day Moving Average (price) ---");
    let ma3 = stock_prices.rolling("price", 3, "mean").unwrap();
    println!(
        "{}\n",
        ma3.select(&["day", "price", "price_rolling_mean"]).unwrap()
    );

    // Rolling sum (3-day volume)
    println!("--- 3-Day Rolling Volume Sum ---");
    let vol_sum = stock_prices.rolling("volume", 3, "sum").unwrap();
    println!(
        "{}\n",
        vol_sum
            .select(&["day", "volume", "volume_rolling_sum"])
            .unwrap()
    );

    // Rolling max (5-day high)
    println!("--- 5-Day High Price ---");
    let high5 = stock_prices.rolling("price", 5, "max").unwrap();
    println!(
        "{}\n",
        high5
            .select(&["day", "price", "price_rolling_max"])
            .unwrap()
    );

    // Rolling min (5-day low)
    println!("--- 5-Day Low Price ---");
    let low5 = stock_prices.rolling("price", 5, "min").unwrap();
    println!(
        "{}\n",
        low5.select(&["day", "price", "price_rolling_min"]).unwrap()
    );

    // Rolling std (volatility)
    println!("--- 3-Day Price Volatility (std) ---");
    let volatility = stock_prices.rolling("price", 3, "std").unwrap();
    println!(
        "{}\n",
        volatility
            .select(&["day", "price", "price_rolling_std"])
            .unwrap()
    );

    // ========== 2. CUMULATIVE OPERATIONS ==========
    println!("\n=== 2. CUMULATIVE OPERATIONS - Running Totals ===\n");

    let sales = DataFrame::builder()
        .add_column("month", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .add_column(
            "revenue",
            vec![1000.0, 1200.0, 1500.0, 1300.0, 1800.0, 2000.0],
        )
        .add_column("growth", vec![1.0, 1.2, 1.15, 1.1, 1.25, 1.3])
        .build()
        .unwrap();

    println!("Monthly sales data:");
    println!("{}\n", sales);

    // Cumulative sum - year-to-date revenue
    println!("--- Cumulative Revenue (YTD) ---");
    let ytd = sales.cumsum("revenue").unwrap();
    println!("{}\n", ytd);

    // Cumulative product - compound growth
    println!("--- Cumulative Growth Factor ---");
    let compound = sales.cumprod("growth").unwrap();
    println!("{}\n", compound);

    // Cumulative max - all-time high
    println!("--- All-Time High Revenue ---");
    let ath = sales.cummax("revenue").unwrap();
    println!("{}\n", ath);

    // Cumulative min - all-time low
    println!("--- All-Time Low Revenue ---");
    let atl = sales.cummin("revenue").unwrap();
    println!("{}\n", atl);

    // ========== 3. SHIFT - LAG/LEAD ==========
    println!("\n=== 3. SHIFT - Lag & Lead for Time Series ===\n");

    let metrics = DataFrame::builder()
        .add_column("period", vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .add_column("value", vec![100.0, 120.0, 115.0, 130.0, 125.0])
        .build()
        .unwrap();

    println!("Original data:");
    println!("{}\n", metrics);

    // Lag 1 (previous period)
    println!("--- Lag 1 (previous value) ---");
    let lag1 = metrics.shift("value", 1).unwrap();
    println!("{}\n", lag1);

    // Lead 1 (next period)
    println!("--- Lead 1 (next value) ---");
    let lead1 = metrics.shift("value", -1).unwrap();
    println!("{}\n", lead1);

    // Lag 2 (two periods ago)
    println!("--- Lag 2 (two periods ago) ---");
    let lag2 = metrics.shift("value", 2).unwrap();
    println!("{}\n", lag2);

    // ========== 4. QUANTILE - PERCENTILES ==========
    println!("\n=== 4. QUANTILE - Percentile Analysis ===\n");

    let scores = DataFrame::builder()
        .add_column(
            "student",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .add_column(
            "score",
            vec![45.0, 67.0, 78.0, 82.0, 88.0, 91.0, 93.0, 95.0, 98.0, 100.0],
        )
        .build()
        .unwrap();

    println!("Student scores:");
    println!("{}\n", scores);

    println!("Percentile analysis:");
    println!(
        "  25th percentile (Q1): {:.1}",
        scores.quantile("score", 0.25).unwrap()
    );
    println!(
        "  50th percentile (median): {:.1}",
        scores.quantile("score", 0.50).unwrap()
    );
    println!(
        "  75th percentile (Q3): {:.1}",
        scores.quantile("score", 0.75).unwrap()
    );
    println!(
        "  90th percentile: {:.1}",
        scores.quantile("score", 0.90).unwrap()
    );
    println!(
        "  95th percentile: {:.1}\n",
        scores.quantile("score", 0.95).unwrap()
    );

    // ========== 5. RANK - VALUE RANKING ==========
    println!("=== 5. RANK - Ranking Values ===\n");

    let competitors = DataFrame::builder()
        .add_column("competitor", vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .add_column("revenue", vec![5000.0, 8000.0, 3000.0, 12000.0, 7000.0])
        .build()
        .unwrap();

    println!("Competitor data:");
    println!("{}\n", competitors);

    // Rank descending (1 = highest revenue)
    println!("--- Rank by Revenue (descending, 1 = highest) ---");
    let ranked_desc = competitors.rank("revenue", false).unwrap();
    println!("{}\n", ranked_desc);

    // Rank ascending (1 = lowest revenue)
    println!("--- Rank by Revenue (ascending, 1 = lowest) ---");
    let ranked_asc = competitors.rank("revenue", true).unwrap();
    println!("{}\n", ranked_asc);

    // ========== 6. DROP_DUPLICATES - REMOVE DUPLICATES ==========
    println!("\n=== 6. DROP_DUPLICATES - Remove Duplicate Values ===\n");

    let transactions = DataFrame::builder()
        .add_column("transaction_id", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .add_column(
            "customer",
            vec![101.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0],
        )
        .add_column("amount", vec![50.0, 75.0, 60.0, 100.0, 80.0, 90.0, 110.0])
        .build()
        .unwrap();

    println!("Transaction log (with duplicate customers):");
    println!("{}\n", transactions);

    println!("--- After dropping duplicate customers (keeps first) ---");
    let unique_customers = transactions.drop_duplicates("customer").unwrap();
    println!(
        "Unique customers: {} (from {} transactions)\n",
        unique_customers.n_rows(),
        transactions.n_rows()
    );
    println!("{}\n", unique_customers);

    // ========== 7. INTERPOLATE - FILL MISSING VALUES ==========
    println!("\n=== 7. INTERPOLATE - Linear Interpolation for NaN ===\n");

    let sensor_data = DataFrame::builder()
        .add_column("time", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .add_column(
            "temperature",
            vec![20.0, 22.0, f64::NAN, f64::NAN, 28.0, f64::NAN, 32.0, 34.0],
        )
        .build()
        .unwrap();

    println!("Sensor data with missing values:");
    println!("{}\n", sensor_data);

    println!("--- After linear interpolation ---");
    let interpolated = sensor_data.interpolate("temperature").unwrap();
    println!("{}\n", interpolated);

    // ========== 8. REAL-WORLD WORKFLOW ==========
    println!("\n=== 8. REAL-WORLD EXAMPLE - Stock Analysis ===\n");

    let stock = DataFrame::builder()
        .add_column(
            "day",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .add_column(
            "close",
            vec![
                100.0, 102.0, 98.0, 105.0, 110.0, 108.0, 115.0, 112.0, 118.0, 120.0,
            ],
        )
        .build()
        .unwrap();

    println!("Step 1: Original stock prices");
    println!("{}\n", stock);

    // Calculate 3-day moving average
    let with_ma = stock.rolling("close", 3, "mean").unwrap();
    println!("Step 2: Add 3-day moving average");
    println!(
        "{}\n",
        with_ma
            .select(&["day", "close", "close_rolling_mean"])
            .unwrap()
    );

    // Calculate daily change (lag)
    let with_prev = stock.shift("close", 1).unwrap();
    println!("Step 3: Add previous day's close");
    println!(
        "{}\n",
        with_prev
            .select(&["day", "close", "close_shift_1"])
            .unwrap()
    );

    // Calculate all-time high
    let with_ath = stock.cummax("close").unwrap();
    println!("Step 4: Add all-time high");
    println!(
        "{}\n",
        with_ath.select(&["day", "close", "close_cummax"]).unwrap()
    );

    // Rank by price
    let with_rank = stock.rank("close", false).unwrap();
    println!("Step 5: Rank by price (1 = highest)");
    println!("{}\n", with_rank);

    // ========== 9. COMBINING FEATURES ==========
    println!("\n=== 9. COMBINING MULTIPLE FEATURES ===\n");

    let timeseries = DataFrame::builder()
        .add_column("t", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .add_column(
            "value",
            vec![10.0, f64::NAN, 14.0, 16.0, f64::NAN, 22.0, 25.0, 28.0],
        )
        .build()
        .unwrap();

    println!("Original data with gaps:");
    println!("{}\n", timeseries);

    // Step 1: Interpolate missing values
    let filled = timeseries.interpolate("value").unwrap();
    println!("After interpolation:");
    println!("{}\n", filled);

    // Step 2: Add 3-period moving average
    let with_ma = filled.rolling("value", 3, "mean").unwrap();
    println!("After adding rolling mean:");
    println!(
        "{}\n",
        with_ma
            .select(&["t", "value", "value_rolling_mean"])
            .unwrap()
    );

    // Step 3: Add cumulative sum
    let with_cumsum = with_ma.cumsum("value").unwrap();
    println!("After adding cumulative sum:");
    println!(
        "{}\n",
        with_cumsum.select(&["t", "value", "value_cumsum"]).unwrap()
    );

    // ========== SUMMARY ==========
    println!("\n=== FEATURE SUMMARY ===\n");

    println!("✅ ROLLING - Window functions:");
    println!("  • rolling(col, window, 'mean') - Moving average");
    println!("  • rolling(col, window, 'sum')  - Rolling sum");
    println!("  • rolling(col, window, 'min')  - Rolling minimum");
    println!("  • rolling(col, window, 'max')  - Rolling maximum");
    println!("  • rolling(col, window, 'std')  - Rolling volatility");

    println!("\n✅ CUMULATIVE operations:");
    println!("  • cumsum(col)   - Running total");
    println!("  • cumprod(col)  - Compound growth");
    println!("  • cummax(col)   - All-time high");
    println!("  • cummin(col)   - All-time low");

    println!("\n✅ SHIFT - Time series:");
    println!("  • shift(col, 1)  - Lag (previous value)");
    println!("  • shift(col, -1) - Lead (next value)");
    println!("  • shift(col, n)  - n periods back");

    println!("\n✅ QUANTILE - Statistics:");
    println!("  • quantile(col, 0.25) - Q1");
    println!("  • quantile(col, 0.50) - Median");
    println!("  • quantile(col, 0.75) - Q3");

    println!("\n✅ RANK - Ordering:");
    println!("  • rank(col, false) - Descending (1 = highest)");
    println!("  • rank(col, true)  - Ascending (1 = lowest)");

    println!("\n✅ DROP_DUPLICATES - Data cleaning:");
    println!("  • drop_duplicates(col) - Keep first occurrence");

    println!("\n✅ INTERPOLATE - Missing data:");
    println!("  • interpolate(col) - Linear interpolation for NaN");

    println!("\n=== USE CASES ===\n");
    println!("📈 Financial Analysis:");
    println!("  - Moving averages (SMA, EMA simulation)");
    println!("  - Price momentum (shift for returns)");
    println!("  - All-time highs/lows (cummax/cummin)");
    println!("  - Volatility windows (rolling std)");

    println!("\n📊 Time Series:");
    println!("  - Seasonal patterns (rolling aggregations)");
    println!("  - Trend analysis (cumulative operations)");
    println!("  - Lag features for ML (shift)");
    println!("  - Missing data handling (interpolate)");

    println!("\n🔬 Statistical Analysis:");
    println!("  - Percentile rankings (quantile)");
    println!("  - Outlier detection (quantile + rank)");
    println!("  - Distribution analysis");

    println!("\n🧹 Data Cleaning:");
    println!("  - Remove duplicates (drop_duplicates)");
    println!("  - Fill gaps (interpolate)");
    println!("  - Smooth noisy data (rolling mean)");

    println!("\n=== Demo Complete! ===");
}
