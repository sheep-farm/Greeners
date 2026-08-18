use greeners::DataFrame;

fn main() {
    println!("=== TIME SERIES OPERATIONS - v1.9.0 ===\n");
    println!("Demonstrating: lag(), lead(), diff(), pct_change()\n");

    // ========== 1. LAG OPERATOR - CREATING LAGGED VARIABLES ==========
    println!("=== 1. LAG OPERATOR - Creating Lagged Variables ===\n");

    let prices_df = DataFrame::builder()
        .add_column("day", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .add_column(
            "price",
            vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 107.0, 110.0],
        )
        .build()
        .unwrap();

    println!("Original price data:");
    println!("{}\n", prices_df);

    // Lag 1 - previous day's price
    println!("--- Lag 1 (previous period) ---");
    let lag1_df = prices_df.lag("price", 1).unwrap();
    println!(
        "{}\n",
        lag1_df.select(&["day", "price", "price_lag_1"]).unwrap()
    );

    // Lag 2 - two days ago
    println!("--- Lag 2 (two periods ago) ---");
    let lag2_df = prices_df.lag("price", 2).unwrap();
    println!(
        "{}\n",
        lag2_df.select(&["day", "price", "price_lag_2"]).unwrap()
    );

    // Multiple lags for autoregressive models
    println!("--- Multiple Lags for AR(3) Model ---");
    let ar_df = prices_df
        .lag("price", 1)
        .unwrap()
        .lag("price", 2)
        .unwrap()
        .lag("price", 3)
        .unwrap();
    println!(
        "{}\n",
        ar_df
            .select(&["day", "price", "price_lag_1", "price_lag_2", "price_lag_3"])
            .unwrap()
    );

    // ========== 2. LEAD OPERATOR - FORWARD-LOOKING VARIABLES ==========
    println!("\n=== 2. LEAD OPERATOR - Forward-Looking Variables ===\n");

    let forecast_df = DataFrame::builder()
        .add_column("quarter", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .add_column("sales", vec![100.0, 110.0, 105.0, 120.0, 115.0, 130.0])
        .build()
        .unwrap();

    println!("Quarterly sales:");
    println!("{}\n", forecast_df);

    // Lead 1 - next quarter's sales
    println!("--- Lead 1 (next period) ---");
    let lead1_df = forecast_df.lead("sales", 1).unwrap();
    println!(
        "{}\n",
        lead1_df
            .select(&["quarter", "sales", "sales_lead_1"])
            .unwrap()
    );

    // Lead 2 - two quarters ahead
    println!("--- Lead 2 (two periods ahead) ---");
    let lead2_df = forecast_df.lead("sales", 2).unwrap();
    println!(
        "{}\n",
        lead2_df
            .select(&["quarter", "sales", "sales_lead_2"])
            .unwrap()
    );

    // ========== 3. DIFF OPERATOR - FIRST DIFFERENCES ==========
    println!("\n=== 3. DIFF OPERATOR - First Differences ===\n");

    let gdp_df = DataFrame::builder()
        .add_column("year", vec![2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0])
        .add_column("gdp", vec![100.0, 102.5, 105.0, 107.8, 110.2, 108.0])
        .build()
        .unwrap();

    println!("GDP levels (non-stationary):");
    println!("{}\n", gdp_df);

    // First difference - make stationary
    println!("--- First Difference (absolute change) ---");
    let diff1_df = gdp_df.diff("gdp", 1).unwrap();
    println!(
        "{}\n",
        diff1_df.select(&["year", "gdp", "gdp_diff_1"]).unwrap()
    );

    // Second difference
    println!("--- Second Difference (change in change) ---");
    let diff2_df = gdp_df.diff("gdp", 2).unwrap();
    println!(
        "{}\n",
        diff2_df.select(&["year", "gdp", "gdp_diff_2"]).unwrap()
    );

    // ========== 4. PERCENTAGE CHANGE - RETURNS CALCULATION ==========
    println!("\n=== 4. PERCENTAGE CHANGE - Returns Calculation ===\n");

    let stock_df = DataFrame::builder()
        .add_column("week", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .add_column(
            "close",
            vec![100.0, 105.0, 103.0, 110.0, 108.0, 115.0, 118.0, 120.0],
        )
        .build()
        .unwrap();

    println!("Weekly stock prices:");
    println!("{}\n", stock_df);

    // 1-week return
    println!("--- 1-Week Returns (%) ---");
    let ret1_df = stock_df.pct_change("close", 1).unwrap();
    let close_vals = ret1_df.get("close").unwrap();
    let ret1_vals = ret1_df.get("close_pct_1").unwrap();

    println!("╭─────┬────────┬────────────╮");
    println!("│ week│  close │ close_pct_1│");
    println!("├─────┼────────┼────────────┤");
    for i in 0..close_vals.len() {
        if ret1_vals[i].is_nan() {
            println!(
                "│ {:>4.0}│ {:>7.2}│        NaN │",
                ret1_df.get("week").unwrap()[i],
                close_vals[i]
            );
        } else {
            println!(
                "│ {:>4.0}│ {:>7.2}│ {:>9.4} │",
                ret1_df.get("week").unwrap()[i],
                close_vals[i],
                ret1_vals[i]
            );
        }
    }
    println!("╰─────┴────────┴────────────╯\n");

    // 2-week return
    println!("--- 2-Week Returns (%) ---");
    let ret2_df = stock_df.pct_change("close", 2).unwrap();
    println!(
        "{}\n",
        ret2_df.select(&["week", "close", "close_pct_2"]).unwrap()
    );

    // ========== 5. PRACTICAL EXAMPLE: STOCK PRICE ANALYSIS ==========
    println!("\n=== 5. PRACTICAL EXAMPLE - Stock Price Analysis ===\n");

    let apple_df = DataFrame::builder()
        .add_column(
            "date",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .add_column(
            "price",
            vec![
                150.0, 152.0, 151.0, 155.0, 153.0, 158.0, 160.0, 159.0, 162.0, 165.0,
            ],
        )
        .build()
        .unwrap();

    println!("AAPL daily prices:");
    println!("{}\n", apple_df);

    // Build complete analysis DataFrame
    let analysis_df = apple_df
        .lag("price", 1)
        .unwrap() // Previous day's price
        .diff("price", 1)
        .unwrap() // Daily price change
        .pct_change("price", 1)
        .unwrap(); // Daily return

    println!("Complete analysis:");
    println!(
        "{}\n",
        analysis_df
            .select(&[
                "date",
                "price",
                "price_lag_1",
                "price_diff_1",
                "price_pct_1"
            ])
            .unwrap()
    );

    // ========== 6. PRACTICAL EXAMPLE: GDP GROWTH RATES ==========
    println!("\n=== 6. PRACTICAL EXAMPLE - GDP Growth Rates ===\n");

    let macro_df = DataFrame::builder()
        .add_column(
            "year",
            vec![
                2010.0, 2011.0, 2012.0, 2013.0, 2014.0, 2015.0, 2016.0, 2017.0,
            ],
        )
        .add_column(
            "gdp_billions",
            vec![
                14500.0, 15000.0, 15500.0, 16000.0, 16800.0, 17400.0, 18000.0, 18500.0,
            ],
        )
        .build()
        .unwrap();

    println!("GDP data:");
    println!("{}\n", macro_df);

    // Calculate growth rates
    let growth_df = macro_df
        .diff("gdp_billions", 1)
        .unwrap()
        .pct_change("gdp_billions", 1)
        .unwrap();

    println!("GDP growth analysis:");
    println!(
        "{}\n",
        growth_df
            .select(&[
                "year",
                "gdp_billions",
                "gdp_billions_diff_1",
                "gdp_billions_pct_1"
            ])
            .unwrap()
    );

    // ========== 7. PRACTICAL EXAMPLE: AR(2) MODEL PREPARATION ==========
    println!("\n=== 7. PRACTICAL EXAMPLE - AR(2) Model Preparation ===\n");

    let ar_data = DataFrame::builder()
        .add_column("t", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        .add_column(
            "y",
            vec![10.0, 12.0, 15.0, 14.0, 18.0, 20.0, 19.0, 23.0, 25.0, 24.0],
        )
        .build()
        .unwrap();

    println!("Time series data:");
    println!("{}\n", ar_data);

    // Prepare for AR(2): y_t = β0 + β1*y_{t-1} + β2*y_{t-2} + ε_t
    let ar2_df = ar_data.lag("y", 1).unwrap().lag("y", 2).unwrap();

    println!("AR(2) regression ready:");
    println!(
        "{}\n",
        ar2_df.select(&["t", "y", "y_lag_1", "y_lag_2"]).unwrap()
    );

    println!("Ready for regression: y ~ y_lag_1 + y_lag_2");
    println!("First 2 observations are NaN (lost to lagging)\n");

    // ========== 8. PRACTICAL EXAMPLE: MOMENTUM TRADING STRATEGY ==========
    println!("\n=== 8. PRACTICAL EXAMPLE - Momentum Trading Strategy ===\n");

    let trading_df = DataFrame::builder()
        .add_column(
            "day",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .add_column(
            "price",
            vec![
                100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 108.0, 112.0, 115.0, 118.0,
            ],
        )
        .build()
        .unwrap();

    println!("Daily prices:");
    println!("{}\n", trading_df);

    // Calculate short-term (1-day) and medium-term (3-day) momentum
    let momentum_df = trading_df
        .pct_change("price", 1)
        .unwrap()
        .pct_change("price", 3)
        .unwrap();

    println!("Momentum signals:");
    println!(
        "{}\n",
        momentum_df
            .select(&["day", "price", "price_pct_1", "price_pct_3"])
            .unwrap()
    );

    println!("Trading rules:");
    println!("  • BUY if price_pct_3 > 0.05 (5% gain in 3 days)");
    println!("  • SELL if price_pct_1 < -0.02 (2% loss in 1 day)\n");

    // ========== 9. LAG vs SHIFT - UNDERSTANDING THE DIFFERENCE ==========
    println!("\n=== 9. LAG vs SHIFT - Understanding the Difference ===\n");

    let compare_df = DataFrame::builder()
        .add_column("t", vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .add_column("value", vec![10.0, 20.0, 30.0, 40.0, 50.0])
        .build()
        .unwrap();

    println!("Original data:");
    println!("{}\n", compare_df);

    // lag() - econometric approach (positive periods = backward)
    let with_lag = compare_df.lag("value", 1).unwrap();
    println!("--- Using lag(1) - econometric style ---");
    println!(
        "{}\n",
        with_lag.select(&["t", "value", "value_lag_1"]).unwrap()
    );
    println!("lag(1) creates value_lag_1: shifts backward, NaN at start\n");

    // shift() - pandas approach (positive = forward, negative = backward)
    let with_shift = compare_df.shift("value", 1).unwrap();
    println!("--- Using shift(1) - pandas style ---");
    println!(
        "{}\n",
        with_shift.select(&["t", "value", "value_shift_1"]).unwrap()
    );
    println!("shift(1) shifts forward, NaN at end\n");

    println!("Key differences:");
    println!("  • lag(n): n periods BACK (econometric: y_t-1)");
    println!("  • shift(n): n periods FORWARD (pandas: positive shifts down)");
    println!("  • shift(-n): n periods BACK (pandas: negative shifts up)");
    println!("  • lag(1) ≈ shift(-1) in pandas terminology\n");

    // ========== 10. COMBINING OPERATIONS - DIFFERENCE OF DIFFERENCES ==========
    println!("\n=== 10. COMBINING OPERATIONS - Difference of Differences ===\n");

    let accel_df = DataFrame::builder()
        .add_column("month", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .add_column(
            "sales",
            vec![100.0, 110.0, 125.0, 145.0, 170.0, 200.0, 235.0, 275.0],
        )
        .build()
        .unwrap();

    println!("Monthly sales (accelerating growth):");
    println!("{}\n", accel_df);

    // First difference - growth
    let growth = accel_df.diff("sales", 1).unwrap();
    println!("First difference (growth):");
    println!(
        "{}\n",
        growth.select(&["month", "sales", "sales_diff_1"]).unwrap()
    );

    // Second difference - acceleration
    let acceleration = growth.diff("sales_diff_1", 1).unwrap();
    println!("Second difference (acceleration of growth):");
    println!(
        "{}\n",
        acceleration
            .select(&["month", "sales", "sales_diff_1", "sales_diff_1_diff_1"])
            .unwrap()
    );

    println!("Interpretation:");
    println!("  • sales: Level (original values)");
    println!("  • sales_diff_1: Growth (month-over-month change)");
    println!("  • sales_diff_1_diff_1: Acceleration (change in growth rate)\n");

    // ========== 11. LEAD-LAG ANALYSIS ==========
    println!("\n=== 11. LEAD-LAG ANALYSIS - Causality Testing ===\n");

    let causality_df = DataFrame::builder()
        .add_column("week", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .add_column(
            "google_searches",
            vec![100.0, 110.0, 105.0, 120.0, 115.0, 130.0, 125.0, 140.0],
        )
        .add_column(
            "product_sales",
            vec![50.0, 55.0, 53.0, 60.0, 58.0, 65.0, 63.0, 70.0],
        )
        .build()
        .unwrap();

    println!("Google searches vs product sales:");
    println!("{}\n", causality_df);

    // Hypothesis: searches predict future sales
    let lead_lag_df = causality_df
        .lag("google_searches", 1)
        .unwrap()
        .lag("google_searches", 2)
        .unwrap()
        .lead("product_sales", 1)
        .unwrap();

    println!("Lead-lag structure for Granger causality:");
    println!(
        "{}\n",
        lead_lag_df
            .select(&[
                "week",
                "google_searches",
                "google_searches_lag_1",
                "google_searches_lag_2",
                "product_sales",
                "product_sales_lead_1"
            ])
            .unwrap()
    );

    println!("Analysis:");
    println!("  • google_searches_lag_1: Do searches 1 week ago predict sales today?");
    println!("  • google_searches_lag_2: Do searches 2 weeks ago predict sales today?");
    println!("  • product_sales_lead_1: What will sales be next week?\n");

    // ========== SUMMARY ==========
    println!("\n=== FEATURE SUMMARY - v1.9.0 ===\n");

    println!("✅ TIME SERIES OPERATIONS:");
    println!("  • lag(column, periods)        - Create lagged variables (t-n)");
    println!("  • lead(column, periods)       - Create lead variables (t+n)");
    println!("  • diff(column, periods)       - First differences (Δy)");
    println!("  • pct_change(column, periods) - Percentage changes (returns)");

    println!("\n✅ PROPERTIES:");
    println!("  • All methods return Result<DataFrame>");
    println!("  • periods must be >= 1 (error otherwise)");
    println!("  • NaN for initial/final values where calculation impossible");
    println!("  • Creates new columns with descriptive names");

    println!("\n✅ COLUMN NAMING:");
    println!("  • lag: 'price' → 'price_lag_1', 'price_lag_2'");
    println!("  • lead: 'sales' → 'sales_lead_1', 'sales_lead_2'");
    println!("  • diff: 'gdp' → 'gdp_diff_1', 'gdp_diff_2'");
    println!("  • pct_change: 'close' → 'close_pct_1', 'close_pct_2'");

    println!("\n✅ USE CASES:");
    println!("  📈 Finance:");
    println!("    - Stock returns: pct_change(price, 1)");
    println!("    - Momentum: pct_change(price, n) for n periods");
    println!("    - Price changes: diff(price, 1)");

    println!("\n  📊 Econometrics:");
    println!("    - AR models: lag(y, 1), lag(y, 2), ...");
    println!("    - Stationarity: diff(series, 1)");
    println!("    - GDP growth: pct_change(gdp, 1)");

    println!("\n  🔬 Research:");
    println!("    - Lead-lag analysis: lag(x, n) + lead(y, m)");
    println!("    - Granger causality: multiple lags");
    println!("    - Panel data: lag within groups");

    println!("\n  🤖 Machine Learning:");
    println!("    - Feature engineering: lag(features, 1..n)");
    println!("    - Time series forecasting: create lagged predictors");
    println!("    - Sequence modeling: lead(target, 1) for prediction");

    println!("\n✅ MATHEMATICAL RELATIONSHIPS:");
    println!("  • lag(x, n)[t] = x[t-n]");
    println!("  • lead(x, n)[t] = x[t+n]");
    println!("  • diff(x, n)[t] = x[t] - x[t-n]");
    println!("  • pct_change(x, n)[t] = (x[t] - x[t-n]) / x[t-n]");
    println!("  • pct_change = diff / lag");

    println!("\n✅ ERROR HANDLING:");
    println!("  • InvalidOperation: periods = 0");
    println!("  • VariableNotFound: column doesn't exist");
    println!("  • Division by zero in pct_change → NaN");

    println!("\n✅ INTEGRATION:");
    println!("  • Works with all DataFrame types");
    println!("  • Chain operations: df.lag().diff().pct_change()");
    println!("  • Compatible with regression models");
    println!("  • Export to CSV/JSON preserves all columns");

    println!("\n=== Demo Complete! ===");
}
