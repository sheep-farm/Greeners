use chrono::{Duration, NaiveDate, NaiveDateTime};
use greeners::DataFrame;

fn main() {
    println!("=== DATETIME FEATURES - v1.6.0 ===\n");
    println!("Demonstrating DateTime column support in Greeners DataFrame\n");

    // ========== 1. BASIC DATETIME CREATION ==========
    println!("=== 1. Creating DataFrame with DateTime Columns ===\n");

    let time_series_df = DataFrame::builder()
        .add_datetime(
            "date",
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(9, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 2)
                    .unwrap()
                    .and_hms_opt(9, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 3)
                    .unwrap()
                    .and_hms_opt(9, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 4)
                    .unwrap()
                    .and_hms_opt(9, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 5)
                    .unwrap()
                    .and_hms_opt(9, 0, 0)
                    .unwrap(),
            ],
        )
        .add_column("price", vec![100.0, 102.5, 101.8, 103.2, 104.5])
        .add_column("volume", vec![1000.0, 1200.0, 1100.0, 1300.0, 1250.0])
        .build()
        .unwrap();

    println!("Time Series DataFrame with DateTime column:");
    println!("{}\n", time_series_df);

    // ========== 2. INSPECTING DATETIME COLUMNS ==========
    println!("=== 2. Inspecting DateTime Data ===\n");

    let date_col = time_series_df.get_datetime("date").unwrap();
    println!("date column details:");
    println!("  Length: {}", date_col.len());
    println!("  First date: {}", date_col[0].format("%Y-%m-%d %H:%M:%S"));
    println!(
        "  Last date: {}",
        date_col[date_col.len() - 1].format("%Y-%m-%d %H:%M:%S")
    );
    println!("  Date range: {} days\n", date_col.len());

    // ========== 3. DATETIME WITH DIFFERENT TIMES ==========
    println!("=== 3. DateTime with Hour/Minute/Second ===\n");

    let intraday_df = DataFrame::builder()
        .add_datetime(
            "timestamp",
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(9, 30, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(12, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(15, 45, 30)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(16, 0, 0)
                    .unwrap(),
            ],
        )
        .add_column("price", vec![100.0, 101.5, 102.0, 101.8])
        .build()
        .unwrap();

    println!("Intraday data with precise timestamps:");
    println!("{}\n", intraday_df);

    // ========== 4. MIXED TYPE OPERATIONS ==========
    println!("=== 4. Operations with Mixed Types (Float + DateTime) ===\n");

    println!("--- Descriptive Statistics ---");
    let stats = time_series_df.describe();
    for (col_name, col_stats) in &stats {
        if col_name != "date" {
            // DateTime não aparece em describe (não é numérico)
            println!("\n{}:", col_name);
            for (stat, value) in col_stats {
                println!("  {}: {:.2}", stat, value);
            }
        }
    }
    println!();

    // ========== 5. DATETIME TO NUMERIC CONVERSION ==========
    println!("=== 5. DateTime to Numeric Conversion ===\n");

    println!("DateTime columns convert to Unix timestamp for numeric operations:");
    let dt_col = time_series_df.get_column("date").unwrap();
    let numeric = dt_col.to_float();
    println!("  First 3 timestamps: {:?}\n", &numeric.to_vec()[0..3]);

    // ========== 6. CONCATENATION ==========
    println!("=== 6. Concatenating DataFrames with DateTime ===\n");

    let new_data = DataFrame::builder()
        .add_datetime(
            "date",
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 6)
                    .unwrap()
                    .and_hms_opt(9, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 7)
                    .unwrap()
                    .and_hms_opt(9, 0, 0)
                    .unwrap(),
            ],
        )
        .add_column("price", vec![105.2, 106.0])
        .add_column("volume", vec![1400.0, 1350.0])
        .build()
        .unwrap();

    println!("New data to append:");
    println!("{}\n", new_data);

    let combined = time_series_df.concat(&new_data).unwrap();
    println!("Combined DataFrame ({} rows):", combined.n_rows());
    println!("{}\n", combined);

    // ========== 7. SELECTION AND SLICING ==========
    println!("=== 7. Selection and Slicing ===\n");

    println!("--- Select specific columns ---");
    let selected = time_series_df.select(&["date", "price"]).unwrap();
    println!("{}\n", selected);

    println!("--- Head (first 3 rows) ---");
    let head = time_series_df.head(3).unwrap();
    println!("{}\n", head);

    // ========== 8. EXPORT/IMPORT ==========
    println!("=== 8. Export to CSV/JSON (preserves datetime format) ===\n");

    println!("When exported:");
    println!("  - CSV: DateTime columns → 'YYYY-MM-DD HH:MM:SS' strings");
    println!("  - JSON: DateTime columns → formatted string values");
    println!("  - Use df.to_csv('output.csv') or df.to_json('output.json')\n");

    // ========== 9. PRACTICAL EXAMPLE: FINANCIAL TIME SERIES ==========
    println!("=== 9. Practical Example: Financial Time Series ===\n");

    let stock_data = DataFrame::builder()
        .add_datetime(
            "date",
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 2)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 3)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 4)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 5)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 8)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
            ],
        )
        .add_column("open", vec![150.0, 152.0, 151.5, 153.0, 154.5])
        .add_column("high", vec![152.5, 153.0, 154.0, 155.0, 156.0])
        .add_column("low", vec![149.5, 151.0, 150.5, 152.5, 153.5])
        .add_column("close", vec![152.0, 151.5, 153.5, 154.0, 155.5])
        .add_column(
            "volume",
            vec![1000000.0, 1100000.0, 950000.0, 1200000.0, 1050000.0],
        )
        .build()
        .unwrap();

    println!("Stock OHLCV data:");
    println!("{}\n", stock_data);

    // ========== 10. EVENT STUDY EXAMPLE ==========
    println!("=== 10. Event Study: Before/After Analysis ===\n");

    let event_date = NaiveDate::from_ymd_opt(2024, 6, 15)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let event_study = DataFrame::builder()
        .add_datetime(
            "date",
            vec![
                event_date - Duration::days(3),
                event_date - Duration::days(2),
                event_date - Duration::days(1),
                event_date,
                event_date + Duration::days(1),
                event_date + Duration::days(2),
                event_date + Duration::days(3),
            ],
        )
        .add_column("returns", vec![-0.5, 0.3, -0.2, 2.5, 1.2, 0.8, -0.3])
        .add_bool(
            "is_after_event",
            vec![false, false, false, false, true, true, true],
        )
        .build()
        .unwrap();

    println!(
        "Event study around {} announcement:",
        event_date.format("%Y-%m-%d")
    );
    println!("{}\n", event_study);

    // Calculate average returns before/after
    let returns = event_study.get("returns").unwrap();
    let is_after = event_study.get_bool("is_after_event").unwrap();

    let before_returns: Vec<f64> = returns
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_after[*i])
        .map(|(_, &r)| r)
        .collect();

    let after_returns: Vec<f64> = returns
        .iter()
        .enumerate()
        .filter(|(i, _)| is_after[*i])
        .map(|(_, &r)| r)
        .collect();

    let avg_before: f64 = before_returns.iter().sum::<f64>() / before_returns.len() as f64;
    let avg_after: f64 = after_returns.iter().sum::<f64>() / after_returns.len() as f64;

    println!("Event Study Results:");
    println!("  Average return before event: {:.2}%", avg_before);
    println!("  Average return after event: {:.2}%", avg_after);
    println!("  Impact: {:.2}%\n", avg_after - avg_before);

    // ========== 11. PANEL DATA WITH TIME DIMENSION ==========
    println!("=== 11. Panel Data: Entities × Time ===\n");

    let panel_data = DataFrame::builder()
        .add_int("firm_id", vec![1, 1, 1, 2, 2, 2, 3, 3, 3])
        .add_datetime(
            "date",
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 2, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 3, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 2, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 3, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 2, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                NaiveDate::from_ymd_opt(2024, 3, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
            ],
        )
        .add_column(
            "revenue",
            vec![
                1000.0, 1100.0, 1200.0, 800.0, 850.0, 900.0, 1500.0, 1550.0, 1600.0,
            ],
        )
        .build()
        .unwrap();

    println!("Panel data (firms × months):");
    println!("{}\\n", panel_data);

    // ========== 12. DATETIME RANGE EXAMPLE ==========
    println!("=== 12. Creating Date Ranges ===\n");

    let start_date = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let date_range: Vec<NaiveDateTime> = (0..7).map(|i| start_date + Duration::days(i)).collect();

    let weekly_data = DataFrame::builder()
        .add_datetime("date", date_range.clone())
        .add_column(
            "value",
            vec![100.0, 105.0, 103.0, 108.0, 110.0, 107.0, 112.0],
        )
        .build()
        .unwrap();

    println!("Weekly data (7 days):");
    println!("{}\n", weekly_data);

    // ========== SUMMARY ==========
    println!("=== FEATURE SUMMARY ===\n");

    println!("✅ DateTime Column Support (v1.6.0):");
    println!("  • add_datetime(name, values) - Create datetime column");
    println!("  • get_datetime(name) - Access datetime data");
    println!("  • NaiveDateTime type (no timezone) - precise to the second");
    println!("  • Format: YYYY-MM-DD HH:MM:SS");

    println!("\n✅ Operations:");
    println!("  • to_float() - Convert datetime → Unix timestamp for calculations");
    println!("  • concat() - Combine time series datasets");
    println!("  • filter() - Filter by date ranges");
    println!("  • sort_by() - Sort chronologically");
    println!("  • select() - Extract date + specific columns");

    println!("\n✅ Display:");
    println!("  • DateTime columns formatted as readable strings");
    println!("  • Consistent 19-character format (YYYY-MM-DD HH:MM:SS)");
    println!("  • Mixed display with numeric columns");

    println!("\n✅ Export:");
    println!("  • to_csv() - DateTime exported as formatted strings");
    println!("  • to_json() - DateTime exported as ISO-like strings");

    println!("\n✅ Use Cases:");
    println!("  • Financial time series - Stock prices, OHLCV data");
    println!("  • Event studies - Before/after analysis");
    println!("  • Panel data - Entity-time observations");
    println!("  • Intraday data - High-frequency trading");
    println!("  • Economic indicators - Quarterly/monthly data");
    println!("  • Survival analysis - Time-to-event data");

    println!("\n✅ Integration with chrono:");
    println!("  • Uses NaiveDateTime from chrono crate");
    println!("  • Duration arithmetic for date ranges");
    println!("  • Flexible date creation (from_ymd_opt + and_hms_opt)");
    println!("  • Format strings for custom display");

    println!("\n✅ Time Series Features:");
    println!("  • Create date ranges with Duration::days()");
    println!("  • Convert to Unix timestamp for regression");
    println!("  • Panel data: multiple entities over time");
    println!("  • Event windows: before/during/after periods");

    println!("\n=== Demo Complete! ===");
}
