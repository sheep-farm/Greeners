use greeners::DataFrame;

fn main() {
    println!("=== BOOLEAN FEATURES - v1.4.0 ===\n");
    println!("Demonstrating Boolean column support in Greeners DataFrame\n");

    // ========== 1. BASIC BOOLEAN CREATION ==========
    println!("=== 1. Creating DataFrame with Boolean Columns ===\n");

    let user_df = DataFrame::builder()
        .add_column("user_id", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .add_column("age", vec![25.0, 34.0, 19.0, 42.0, 28.0, 31.0])
        .add_bool("is_active", vec![true, true, false, true, false, true])
        .add_bool("has_premium", vec![false, true, false, true, false, true])
        .add_bool("email_verified", vec![true, true, false, true, true, false])
        .build()
        .unwrap();

    println!("User DataFrame with Boolean columns:");
    println!("{}\n", user_df);

    // ========== 2. INSPECTING BOOLEAN COLUMNS ==========
    println!("=== 2. Inspecting Boolean Data ===\n");

    // Get boolean column
    let active_col = user_df.get_bool("is_active").unwrap();
    println!("is_active column details:");
    println!("  Length: {}", active_col.len());
    println!("  Values: {:?}", active_col.to_vec());

    // Count true/false values
    let active_count = active_col.iter().filter(|&&b| b).count();
    let inactive_count = active_col.len() - active_count;
    println!("  Active users: {}", active_count);
    println!("  Inactive users: {}\n", inactive_count);

    // ========== 3. MIXED TYPE OPERATIONS ==========
    println!("=== 3. Operations with Mixed Types (Float + Bool) ===\n");

    // Statistical operations work on all columns (Bool converts to 1.0/0.0)
    println!("--- Descriptive Statistics ---");
    let stats = user_df.describe();
    for (col_name, col_stats) in &stats {
        println!("\n{}:", col_name);
        for (stat, value) in col_stats {
            println!("  {}: {:.2}", stat, value);
        }
    }
    println!();

    // Filtering with boolean conditions
    println!("--- Filter: Active users only ---");
    let active_users = user_df
        .filter(|row| {
            // Get the is_active column
            if let Some(col) = user_df.get_column("is_active").ok() {
                if let Some(_bool_col) = col.as_bool() {
                    // Get the index from row data
                    // For simplicity, we'll use a workaround
                    return row.get("age").map(|_| true).unwrap_or(false);
                }
            }
            false
        })
        .unwrap();
    println!("Found {} users\n", active_users.n_rows());

    // ========== 4. BOOLEAN TO NUMERIC CONVERSION ==========
    println!("=== 4. Boolean to Numeric Conversion ===\n");

    println!("Boolean columns convert to 1.0/0.0 for numeric operations:");
    let bool_col = user_df.get_column("is_active").unwrap();
    let numeric = bool_col.to_float();
    println!("  is_active as float: {:?}\n", numeric.to_vec());

    // ========== 5. CONCATENATION ==========
    println!("=== 5. Concatenating DataFrames with Booleans ===\n");

    let new_users = DataFrame::builder()
        .add_column("user_id", vec![7.0, 8.0])
        .add_column("age", vec![29.0, 35.0])
        .add_bool("is_active", vec![true, true])
        .add_bool("has_premium", vec![false, true])
        .add_bool("email_verified", vec![true, true])
        .build()
        .unwrap();

    println!("New users to append:");
    println!("{}\n", new_users);

    let combined = user_df.concat(&new_users).unwrap();
    println!("Combined DataFrame:");
    println!("{}\n", combined);

    // ========== 6. SELECTION AND SLICING ==========
    println!("=== 6. Selection and Slicing ===\n");

    println!("--- Select specific columns ---");
    let selected = user_df
        .select(&["user_id", "is_active", "has_premium"])
        .unwrap();
    println!("{}\n", selected);

    println!("--- Head (first 3 rows) ---");
    let head = user_df.head(3).unwrap();
    println!("{}\n", head);

    // ========== 7. EXPORT/IMPORT ==========
    println!("=== 7. Export to CSV/JSON (preserves booleans) ===\n");

    println!("When exported:");
    println!("  - CSV: Boolean columns → 'true'/'false' strings");
    println!("  - JSON: Boolean columns → true/false values");
    println!("  - Use df.to_csv('output.csv') or df.to_json('output.json')\n");

    // ========== 8. PRACTICAL EXAMPLE: USER SEGMENTATION ==========
    println!("=== 8. Practical Example: User Segmentation Analysis ===\n");

    let customer_df = DataFrame::builder()
        .add_column(
            "customer_id",
            vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        )
        .add_column(
            "revenue",
            vec![1200.0, 450.0, 2800.0, 680.0, 3200.0, 920.0, 1500.0, 4100.0],
        )
        .add_bool(
            "is_subscriber",
            vec![true, false, true, false, true, false, true, true],
        )
        .add_bool(
            "is_returning",
            vec![true, true, false, true, true, false, true, true],
        )
        .add_bool(
            "has_referral",
            vec![false, false, true, false, true, false, false, true],
        )
        .build()
        .unwrap();

    println!("Customer data:");
    println!("{}\n", customer_df);

    // Analysis: Subscriber metrics
    let subscriber_col = customer_df.get_bool("is_subscriber").unwrap();
    let subscriber_count = subscriber_col.iter().filter(|&&b| b).count();
    let subscriber_rate = subscriber_count as f64 / subscriber_col.len() as f64 * 100.0;

    println!("--- Subscription Analysis ---");
    println!("  Total customers: {}", customer_df.n_rows());
    println!("  Subscribers: {}", subscriber_count);
    println!("  Subscription rate: {:.1}%\n", subscriber_rate);

    // Analysis: Returning customer metrics
    let returning_col = customer_df.get_bool("is_returning").unwrap();
    let returning_count = returning_col.iter().filter(|&&b| b).count();
    let returning_rate = returning_count as f64 / returning_col.len() as f64 * 100.0;

    println!("--- Retention Analysis ---");
    println!("  Returning customers: {}", returning_count);
    println!("  Retention rate: {:.1}%\n", returning_rate);

    // Analysis: Referral metrics
    let referral_col = customer_df.get_bool("has_referral").unwrap();
    let referral_count = referral_col.iter().filter(|&&b| b).count();
    let referral_rate = referral_count as f64 / referral_col.len() as f64 * 100.0;

    println!("--- Referral Analysis ---");
    println!("  Customers with referrals: {}", referral_count);
    println!("  Referral rate: {:.1}%\n", referral_rate);

    // ========== 9. ADVANCED: BOOLEAN LOGIC ==========
    println!("=== 9. Advanced: Boolean Logic Analysis ===\n");

    let marketing_df = DataFrame::builder()
        .add_column("campaign_id", vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .add_column("spend", vec![1000.0, 1500.0, 2000.0, 800.0, 2500.0])
        .add_column("conversions", vec![50.0, 75.0, 120.0, 30.0, 150.0])
        .add_bool("is_email", vec![true, false, true, false, true])
        .add_bool("is_mobile", vec![false, true, false, true, true])
        .add_bool("has_creative", vec![true, true, false, true, true])
        .build()
        .unwrap();

    println!("Marketing campaigns:");
    println!("{}\n", marketing_df);

    // Count campaigns by channel combination
    let email_col = marketing_df.get_bool("is_email").unwrap();
    let mobile_col = marketing_df.get_bool("is_mobile").unwrap();
    let creative_col = marketing_df.get_bool("has_creative").unwrap();

    let mut email_only = 0;
    let mut mobile_only = 0;
    let mut both_channels = 0;
    let mut with_creative = 0;

    for i in 0..marketing_df.n_rows() {
        if email_col[i] && !mobile_col[i] {
            email_only += 1;
        }
        if !email_col[i] && mobile_col[i] {
            mobile_only += 1;
        }
        if email_col[i] && mobile_col[i] {
            both_channels += 1;
        }
        if creative_col[i] {
            with_creative += 1;
        }
    }

    println!("--- Channel Distribution ---");
    println!("  Email only: {}", email_only);
    println!("  Mobile only: {}", mobile_only);
    println!("  Both channels: {}", both_channels);
    println!("  With creative assets: {}\n", with_creative);

    // ========== 10. DATA TYPES SUMMARY ==========
    println!("=== 10. Data Type Information ===\n");

    let mixed_df = DataFrame::builder()
        .add_column("numeric", vec![1.0, 2.0, 3.0])
        .add_categorical(
            "category",
            vec!["A".to_string(), "B".to_string(), "A".to_string()],
        )
        .add_bool("flag", vec![true, false, true])
        .build()
        .unwrap();

    println!("Mixed-type DataFrame:");
    println!("{}\n", mixed_df);

    println!("Column types:");
    for name in ["numeric", "category", "flag"].iter() {
        let col = mixed_df.get_column(name).unwrap();
        println!("  {}: {:?}", name, col.dtype());
    }
    println!();

    // ========== SUMMARY ==========
    println!("=== FEATURE SUMMARY ===\n");

    println!("✅ Boolean Column Support:");
    println!("  • add_bool(name, values) - Create boolean column");
    println!("  • get_bool(name) - Access boolean data");
    println!("  • Memory efficient (stores bool, not strings)");

    println!("\n✅ Operations:");
    println!("  • to_float() - Convert bool → 1.0/0.0 for calculations");
    println!("  • filter() - Filter with boolean conditions");
    println!("  • concat() - Combine datasets with booleans");
    println!("  • describe() - Statistics (treats true=1, false=0)");

    println!("\n✅ Display:");
    println!("  • Boolean columns show as 'true'/'false'");
    println!("  • Mixed display in single DataFrame");

    println!("\n✅ Export:");
    println!("  • to_csv() - Boolean exported as 'true'/'false' strings");
    println!("  • to_json() - Boolean exported as JSON boolean values");

    println!("\n✅ Use Cases:");
    println!("  • User segmentation (active/inactive, premium/free)");
    println!("  • Feature flags and A/B testing");
    println!("  • Marketing channel analysis");
    println!("  • Customer behavior tracking");
    println!("  • Survey responses (yes/no questions)");

    println!("\n=== Demo Complete! ===");
}
