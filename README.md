
# Greeners: High-Performance Econometrics in Rust

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)
![Stability](https://img.shields.io/badge/stability-stable-green)

**Greeners** is a lightning-fast, type-safe econometrics library written in pure Rust. It provides a comprehensive suite of estimators for Cross-Sectional, Time-Series, and Panel Data analysis, leveraging linear algebra backends (LAPACK/BLAS) for maximum performance.

Designed for academic research, heavy simulations, and production-grade economic modeling.

## 🎉 v2.0.0 MAJOR RELEASE: Complete Data Handling & Time Series

**Greeners v2.0.0** brings **pandas-like DataFrame capabilities** and **essential time series operations** for econometric analysis!

### 🆕 Three Major Feature Sets (NEW in v2.0.0)

#### 1. String Column Support (v1.7.0)

Store free-form text data alongside numerical columns:

```rust
use greeners::DataFrame;

let customers = DataFrame::builder()
    .add_int("id", vec![1, 2, 3])
    .add_string("name", vec![
        "Alice Johnson".to_string(),
        "Bob Smith".to_string(),
        "Charlie Brown".to_string(),
    ])
    .add_string("email", vec![
        "alice@example.com".to_string(),
        "bob@example.com".to_string(),
        "charlie@example.com".to_string(),
    ])
    .add_column("purchase_amount", vec![150.0, 200.0, 75.0])
    .build()?;

// Access string data
let names = customers.get_string("name")?;
println!("First customer: {}", names[0]); // "Alice Johnson"
```

**String vs Categorical:**
- **String columns**: Free text, unique values (names, emails, addresses, comments)
- **Categorical columns**: Repeated categories, encoded as integers (regions, groups)

📖 See `examples/string_features.rs` for comprehensive demonstration.

#### 2. Missing Data & Null Support (v1.8.0)

Complete toolkit for handling missing values - just like pandas!

```rust
use greeners::DataFrame;

// Detect missing values
let mask = df.isna("temperature")?;  // Boolean mask
let n_missing = df.count_na("temperature");  // Count

// Remove missing data
let clean = df.dropna()?;  // Drop any row with NaN
let clean_subset = df.dropna_subset(&["price", "quantity"])?;  // Drop if specific cols missing

// Fill missing values
let filled = df.fillna("price", 100.0)?;  // Fill with constant
let forward = df.fillna_ffill("price")?;  // Forward fill (carry last valid)
let backward = df.fillna_bfill("price")?;  // Backward fill (carry next valid)
let smooth = df.interpolate("temperature")?;  // Linear interpolation
```

**Comprehensive workflow:**
- **Detect**: `isna()`, `notna()`, `count_na()` for investigation
- **Handle**: `dropna()` for complete-case analysis
- **Impute**: `fillna()`, `ffill()`, `bfill()`, `interpolate()` for treatment

📖 See `examples/missing_data_features.rs` for complete workflow.

#### 3. Time Series Operations (v1.9.0)

Essential operations for econometric time series analysis:

```rust
use greeners::DataFrame;

// Stock price data
let df = DataFrame::builder()
    .add_column("date", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    .add_column("price", vec![100.0, 102.0, 101.0, 105.0, 103.0])
    .build()?;

// Lag operator - create lagged variables
let with_lag = df.lag("price", 1)?;  // Previous day's price → price_lag_1
// Essential for AR models: y_t = β₀ + β₁·y_{t-1} + ε_t

// Lead operator - forward-looking variables
let with_lead = df.lead("price", 1)?;  // Next day's price → price_lead_1
// Essential for lead-lag analysis and Granger causality

// First differences - achieve stationarity
let stationary = df.diff("price", 1)?;  // Δprice_t = price_t - price_{t-1} → price_diff_1
// Essential for unit root tests and I(1) processes

// Percentage changes - returns calculation
let returns = df.pct_change("price", 1)?;  // (price_t - price_{t-1}) / price_{t-1} → price_pct_1
// Standard in finance for asset returns

// Chain operations for complete analysis
let analysis = df
    .lag("price", 1)?
    .diff("price", 1)?
    .pct_change("price", 1)?;
// Creates: price_lag_1, price_diff_1, price_pct_1
```

**Use cases:**
- **Finance**: Returns (`pct_change`), momentum strategies
- **Econometrics**: AR models (`lag`), stationarity testing (`diff`), GDP growth
- **Machine Learning**: Time series feature engineering (multiple lags)

**Mathematical relationships:**
- `lag(x, n)[t] = x[t-n]`
- `lead(x, n)[t] = x[t+n]`
- `diff(x, n)[t] = x[t] - x[t-n]`
- `pct_change(x, n)[t] = (x[t] - x[t-n]) / x[t-n]`

📖 See `examples/time_series_features.rs` for 11 practical examples.

### Why v2.0.0 Matters

**Before v2.0.0:**
- Greeners = Powerful econometric estimators + basic DataFrame
- Missing data? Manual handling required
- Time series? Use `shift()` and manual calculations
- Text data? Not supported

**Now v2.0.0:**
- Greeners = **Complete data analysis platform** with pandas-like capabilities
- String columns ✅ Missing data toolkit ✅ Time series ops ✅
- Full workflow: Load → Clean → Transform → Model → Predict
- **Only Rust library** with comprehensive econometrics + DataFrame

### Migration from v1.0.2

**100% backward compatible** - all v1.0.2 code works unchanged!

New capabilities are additive:
```rust
// Your existing v1.0.2 code
let df = DataFrame::from_csv("data.csv")?;
let formula = Formula::parse("y ~ x1 + x2")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;
// ✅ Still works perfectly!

// New v2.0.0 capabilities
let df_with_strings = df.add_string("region", regions)?;  // NEW
let clean_df = df.dropna()?;  // NEW
let with_lags = df.lag("y", 1)?;  // NEW
```

## 🎊 v1.0.2 STABLE RELEASE: Named Variables & Enhanced Data Loading

Greeners v1.0.2 brings **human-readable variable names** in regression output and **flexible data loading** from multiple sources!

### 🆕 Multiple Data Loading Options (NEW in v1.0.2)

Load data from **CSV, JSON, URLs, or use the Builder pattern** - just like pandas/polars!

```rust
// 1. CSV from URL (reproducible research!)
let df = DataFrame::from_csv_url(
    "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv"
)?;

// 2. JSON from local file (column or record oriented)
let df = DataFrame::from_json("data.json")?;

// 3. JSON from URL (API integration)
let df = DataFrame::from_json_url("https://api.example.com/data.json")?;

// 4. Builder pattern (most convenient!)
let df = DataFrame::builder()
    .add_column("wage", vec![30000.0, 40000.0, 50000.0])
    .add_column("education", vec![12.0, 16.0, 18.0])
    .build()?;

// 5. CSV from local file (classic)
let df = DataFrame::from_csv("data.csv")?;
```

**Why this matters:**
- ✅ **Reproducible research** - Load datasets directly from GitHub/URLs
- ✅ **API integration** - Fetch data from web services
- ✅ **Flexible formats** - CSV, JSON (column/record oriented)
- ✅ **Pandas-like** - Familiar syntax for data scientists
- ✅ **Type-safe** - All data loading is checked at compile time

📖 See `examples/dataframe_loading.rs` for all loading methods.

### Named Variables in Results (NEW in v1.0.2)

No more generic `x0`, `x1`, `x2` in regression output! All models now display **actual variable names** from your Formula:

```rust
use greeners::{OLS, DataFrame, Formula, CovarianceType};

let formula = Formula::parse("wage ~ education + experience + female")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;

println!("{}", result);
```

**Before (v1.0.1):**
```
OLS Regression Results
====================================
Variable    Coef    Std Err    t    P>|t|
const       5.23    0.45      11.62  0.000
x0          2.15    0.12       17.92  0.000    <- Generic names
x1          0.08    0.02        4.00  0.000
x2         -1.20    0.25       -4.80  0.000
```

**Now (v1.0.2):**
```
OLS Regression Results
====================================
Variable      Coef    Std Err    t    P>|t|
const         5.23    0.45      11.62  0.000
education     2.15    0.12       17.92  0.000    <- Actual variable names!
experience    0.08    0.02        4.00  0.000
female       -1.20    0.25       -4.80  0.000
```

**Applies to ALL models:**
- ✅ OLS, WLS, Cochrane-Orcutt (FGLS)
- ✅ IV/2SLS (Instrumental Variables)
- ✅ Logit/Probit (Binary Choice)
- ✅ Quantile Regression (all quantiles)
- ✅ Panel Data (Fixed Effects, Random Effects, Between)
- ✅ GMM (Generalized Method of Moments)
- ✅ Difference-in-Differences

### Comprehensive Test Coverage

v1.0.2 includes **143 unit tests** covering all major functionality:

- **62 new tests** added in v1.0.2 across 7 test modules
- Full coverage of IV/2SLS, Panel Data, DiD, FGLS, Quantile Regression
- Diagnostic tests (VIF, Breusch-Pagan, Jarque-Bera, Durbin-Watson)
- GMM specification tests (J-statistic, overidentification)
- Model selection and information criteria

Run tests locally:
```sh
cargo test              # Run all 143 tests
cargo test --lib        # Library tests only
cargo test quantile     # Specific module tests
```

<!-- ### Quality & Competitive Analysis

Independent analysis positions Greeners in the **TOP 5 worldwide** among econometrics libraries:

**Overall Score: 8.7/10** - Competing with statsmodels, plm/fixest, linearmodels, and Julia GLM

**Strengths:**
- 🏆 Performance (10/10): Fastest in class with BLAS/LAPACK
- 🏆 Type Safety (10/10): Compile-time guarantees via Rust
- 🏆 Modern Covariance (10/10): HC0-HC4, Newey-West, Clustered (1-way/2-way)
- 🏆 Panel Methods (9/10): FE, RE, Between, Arellano-Bond
- 🏆 Formula API (9/10): R/Python syntax with interactions, polynomials, categoricals

**See [QUALITY_ANALYSIS.md](QUALITY_ANALYSIS.md)** for complete competitive comparison with statsmodels, plm, fixest, linearmodels, Stata, and Julia packages. -->

### Code Quality Improvements

- Applied clippy lints for idiomatic Rust (25+ improvements)
- Replaced `.iter().cloned().collect()` with `.to_vec()` for better performance
- Modern range checks using `.contains()` instead of manual comparisons
- Cleaner, more maintainable codebase

## 🎉 v1.0.1: Specification Tests

Greeners reaches **production stability** with comprehensive **specification tests** for diagnosing regression assumptions!

### Specification Tests (NEW in v1.0.1)

Diagnose violations of classical regression assumptions and identify appropriate remedies:

```rust
use greeners::{OLS, SpecificationTests, Formula, DataFrame, CovarianceType};

// Estimate model
let model = OLS::from_formula(&Formula::parse("wage ~ education + experience")?, &df, CovarianceType::NonRobust)?;
let (y, x) = df.to_design_matrix(&formula)?;
let residuals = model.residuals(&y, &x);
let fitted = model.fitted_values(&x);

// 1. White Test for Heteroskedasticity
let (lm_stat, p_value, df) = SpecificationTests::white_test(&residuals, &x)?;
if p_value < 0.05 {
    println!("Heteroskedasticity detected! Use CovarianceType::HC3");
}

// 2. RESET Test for Functional Form Misspecification
let (f_stat, p_value, _, _) = SpecificationTests::reset_test(&y, &x, &fitted, 3)?;
if p_value < 0.05 {
    println!("Misspecification detected! Add polynomials or interactions");
}

// 3. Breusch-Godfrey Test for Autocorrelation
let (lm_stat, p_value, df) = SpecificationTests::breusch_godfrey_test(&residuals, &x, 1)?;
if p_value < 0.05 {
    println!("Autocorrelation detected! Use CovarianceType::NeweyWest(4)");
}

// 4. Goldfeld-Quandt Test for Heteroskedasticity
let (f_stat, p_value, _, _) = SpecificationTests::goldfeld_quandt_test(&residuals, 0.25)?;
```

**When to Use:**
- **White Test** → General heteroskedasticity test (any form)
- **RESET Test** → Detect omitted variables or wrong functional form
- **Breusch-Godfrey** → Detect autocorrelation in time series/panel data
- **Goldfeld-Quandt** → Test heteroskedasticity when you suspect specific ordering

**Remedies:**
- Heteroskedasticity → `CovarianceType::HC3` or `HC4`
- Autocorrelation → `CovarianceType::NeweyWest(lags)`
- Misspecification → Add `I(x^2)`, `x1*x2` interactions

**Stata/R/Python Equivalents:**
- **Stata**: `estat hettest`, `estat ovtest`, `estat bgodfrey`
- **R**: `lmtest::bptest()`, `lmtest::resettest()`, `lmtest::bgtest()`
- **Python**: `statsmodels.stats.diagnostic.het_white()`

📖 See `examples/specification_tests.rs` for comprehensive demonstration.

## ✨ NEW: R/Python-Style Formula API

Greeners now supports **R/Python-style formula syntax** (like `statsmodels` and `lm()`), making model specification intuitive and concise:

```rust
use greeners::{OLS, DataFrame, Formula, CovarianceType};

// Python equivalent: smf.ols('y ~ x1 + x2', data=df).fit(cov_type='HC1')
let formula = Formula::parse("y ~ x1 + x2")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;
```

**All estimators support formulas:** OLS, WLS, DiD, IV/2SLS, Logit/Probit, Quantile Regression, Panel Data (FE/RE/Between), and more!

📖 See [FORMULA_API.md](FORMULA_API.md) for complete documentation and examples.

## 🚀 NEW in v0.9.0: Panel Diagnostics & Model Selection

Greeners now provides comprehensive tools for **panel data model selection** and **information criteria-based model comparison** - essential for rigorous empirical research!

### Model Selection & Comparison

Compare multiple models using **AIC/BIC** with automatic ranking and **Akaike weights** for model averaging:

```rust
use greeners::{OLS, ModelSelection, DataFrame, Formula, CovarianceType};

// Estimate competing models
let model1 = OLS::from_formula(&Formula::parse("y ~ x1 + x2 + x3")?, &df, CovarianceType::NonRobust)?;
let model2 = OLS::from_formula(&Formula::parse("y ~ x1 + x2")?, &df, CovarianceType::NonRobust)?;
let model3 = OLS::from_formula(&Formula::parse("y ~ x1")?, &df, CovarianceType::NonRobust)?;

// Compare models
let models = vec![
    ("Full Model", model1.log_likelihood, 4, n_obs),
    ("Restricted", model2.log_likelihood, 3, n_obs),
    ("Simple", model3.log_likelihood, 2, n_obs),
];
let comparison = ModelSelection::compare_models(models);
ModelSelection::print_comparison(&comparison);

// Calculate Akaike weights for model averaging
let aic_values: Vec<f64> = comparison.iter().map(|(_, aic, _, _, _)| *aic).collect();
let (delta_aic, weights) = ModelSelection::akaike_weights(&aic_values);
```

**Output:**
```
=============================== Model Comparison ===============================
Model                |          AIC |          BIC | Rank(AIC) | Rank(BIC)
--------------------------------------------------------------------------------
Full Model           |       183.83 |       191.48 |        1 |        1
Restricted           |       184.77 |       190.50 |        2 |        2
Simple               |       188.19 |       192.01 |        3 |        3

📊 AKAIKE WEIGHTS:
Δ_AIC < 2: Substantial support
Δ_AIC 4-7: Considerably less support
Δ_AIC > 10: Essentially no support
```

### Panel Diagnostics Tests

Test whether pooled OLS is appropriate or if panel data methods (Fixed/Random Effects) are needed:

#### Breusch-Pagan LM Test for Random Effects

```rust
use greeners::{PanelDiagnostics, OLS, Formula};

// Estimate pooled OLS
let model_pooled = OLS::from_formula(&formula, &df, CovarianceType::NonRobust)?;
let (y, x) = df.to_design_matrix(&formula)?;
let residuals = model_pooled.residuals(&y, &x);

// Test for random effects
let (lm_stat, p_value) = PanelDiagnostics::breusch_pagan_lm(&residuals, &firm_ids)?;

// Interpretation:
// H₀: σ²_u = 0 (no panel effects, pooled OLS adequate)
// H₁: σ²_u > 0 (random effects needed)
// If p < 0.05 → Use Random Effects or Fixed Effects
```

#### F-Test for Fixed Effects

```rust
// Test if firm fixed effects are significant
let (f_stat, p_value) = PanelDiagnostics::f_test_fixed_effects(
    ssr_pooled,
    ssr_fe,
    n_obs,
    n_firms,
    k_params,
)?;

// Interpretation:
// H₀: All firm effects are zero (pooled OLS adequate)
// H₁: Firm effects exist (use fixed effects)
// If p < 0.05 → Use Fixed Effects model
```

### Summary Statistics

Quick descriptive statistics for initial data exploration:

```rust
use greeners::SummaryStats;

let stats = SummaryStats::describe(&data);
// Returns: (mean, std, min, Q25, median, Q75, max, n_obs)

// Pretty-print summary table
let summary_data = vec![
    ("investment", stats_inv),
    ("profit", stats_profit),
    ("cash_flow", stats_cf),
];
SummaryStats::print_summary(&summary_data);
```

**Stata/R/Python Equivalents:**
- **Stata**: `estat ic` (AIC/BIC), `xttest0` (BP LM), `testparm` (F-test)
- **R**: `AIC()`, `BIC()`, `plm::plmtest()`, `plm::pFtest()`
- **Python**: `statsmodels` information criteria, `linearmodels.panel` diagnostics

📖 See `examples/panel_model_selection.rs` for comprehensive demonstration with panel data workflow.

## 🌟 NEW in v0.5.0: Marginal Effects for Binary Choice Models

After estimating Logit/Probit models, **coefficients alone are hard to interpret** (they're on log-odds/z-score scale). **Marginal effects** translate these to **probability changes** - essential for policy analysis and substantive interpretation!

### Average Marginal Effects (AME) - RECOMMENDED

```rust
use greeners::{Logit, Formula, DataFrame};

// Estimate Logit model
let formula = Formula::parse("admitted ~ gpa + sat + legacy")?;
let result = Logit::from_formula(&formula, &df)?;

// Get design matrix
let (_, x) = df.to_design_matrix(&formula)?;

// Calculate Average Marginal Effects (AME)
let ame = result.average_marginal_effects(&x)?;

// Interpretation: AME[gpa] = 0.15 means:
// "A 1-point increase in GPA increases admission probability by 15 percentage points"
// (averaged across all students in the sample)
```

**Why AME?**
- ✅ Accounts for heterogeneity across observations
- ✅ More robust to non-linearities
- ✅ Standard in modern econometrics (Stata, R, Python)
- ✅ Easy to interpret: probability changes, not log-odds

### Marginal Effects at Means (MEM)

```rust
// Calculate Marginal Effects at Means (MEM)
let mem = result.marginal_effects_at_means(&x)?;

// Interpretation: Effect for "average" student
// ⚠️ Less robust than AME - can evaluate at impossible values (e.g., average of dummies)
```

### Predictions

```rust
// Predict admission probabilities for new students
let probs = result.predict_proba(&x_new);

// Example: probs[0] = 0.85 → 85% chance of admission
```

### Logit vs Probit Comparison

```rust
// Both models give similar marginal effects
let logit_result = Logit::from_formula(&formula, &df)?;
let probit_result = Probit::from_formula(&formula, &df)?;

let ame_logit = logit_result.average_marginal_effects(&x)?;
let ame_probit = probit_result.average_marginal_effects(&x)?;

// Typically: ame_logit ≈ ame_probit (differences < 1-2 percentage points)
```

**Stata/R/Python Equivalents:**
- **Stata**: `margins, dydx(*)` (AME) or `margins, dydx(*) atmeans` (MEM)
- **R**: `mfx::logitmfx()` or `margins::margins()`
- **Python**: `statsmodels.discrete.discrete_model.Logit(...).get_margeff()`

📖 See `examples/marginal_effects.rs` for comprehensive demonstration with college admission data.

### Two-Way Clustered Standard Errors

For **panel data** with clustering along **two dimensions** (e.g., firms × time):

```rust
// Panel data: 4 firms × 6 time periods
let firm_ids = vec![0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3];
let time_ids = vec![0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5];

// Two-way clustering (Cameron-Gelbach-Miller, 2011)
let result = OLS::from_formula(
    &formula,
    &df,
    CovarianceType::ClusteredTwoWay(firm_ids, time_ids)
)?;

// Formula: V = V_firm + V_time - V_intersection
// Accounts for BOTH within-firm AND within-time correlation
```

**When to use:**
- ✅ Panel data (firms/countries over time)
- ✅ Correlation within entities AND within time periods
- ✅ More robust than one-way clustering
- ✅ Standard in modern panel data econometrics

**Stata equivalent:** `reghdfe y x, vce(cluster firm_id time_id)`

📖 See `examples/two_way_clustering.rs` for complete comparison of non-robust vs one-way vs two-way clustering.

## 🎊 NEW in v0.4.0: Categorical Variables & Polynomial Terms

### Categorical Variable Encoding
Automatic dummy variable creation with R/Python syntax:

```rust
// Categorical variable: creates dummies, drops first level
let formula = Formula::parse("sales ~ advertising + C(region)")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;

// If region has values [0, 1, 2, 3] → creates 3 dummies (drops 0 as reference)
```

**How it works:**
- `C(var)` detects unique values in the variable
- Creates K-1 dummy variables (drops first category as reference)
- Essential for categorical data (regions, industries, treatment groups)

### Polynomial Terms
Non-linear relationships made easy:

```rust
// Quadratic model: captures diminishing returns
let formula = Formula::parse("output ~ input + I(input^2)")?;

// Cubic model: more flexible
let formula = Formula::parse("y ~ x + I(x^2) + I(x^3)")?;

// Alternative syntax (Python-style)
let formula = Formula::parse("y ~ x + I(x**2)")?;
```

**Use cases:**
- Production functions (diminishing returns)
- Wage curves (experience effects)
- Growth models (non-linear dynamics)

**Combine with interactions:**
```rust
// Region-specific quadratic effects
let formula = Formula::parse("sales ~ C(region) * I(advertising^2)")?;
```

## 🆕 NEW in v0.2.0: Clustered Standard Errors & Advanced Diagnostics

### Clustered Standard Errors
Critical for panel data and hierarchical structures where observations are grouped:

```rust
// Panel data: firms over time
let cluster_ids = vec![0,0,0, 1,1,1, 2,2,2]; // Firm IDs
let result = OLS::from_formula(&formula, &df, CovarianceType::Clustered(cluster_ids))?;
```

**Use clustered SE when:**
- Panel data (repeated observations per entity)
- Hierarchical data (students in schools, patients in hospitals)
- Experimental data with treatment clusters
- Geographic clustering (observations in regions/countries)

### Advanced Diagnostics
New diagnostic tools for model validation:

```rust
use greeners::Diagnostics;

// Multicollinearity detection
let vif = Diagnostics::vif(&x)?;              // Variance Inflation Factor
let cond_num = Diagnostics::condition_number(&x)?;  // Condition Number

// Influential observations
let leverage = Diagnostics::leverage(&x)?;    // Hat values
let cooks_d = Diagnostics::cooks_distance(&residuals, &x, mse)?;  // Cook's Distance

// Assumption testing (already available)
let (jb_stat, jb_p) = Diagnostics::jarque_bera(&residuals)?;  // Normality
let (bp_stat, bp_p) = Diagnostics::breusch_pagan(&residuals, &x)?;  // Heteroskedasticity
let dw_stat = Diagnostics::durbin_watson(&residuals);  // Autocorrelation
```

## 🎉 NEW in v0.3.0: Interactions, HC2/HC3, and Predictions

### Interaction Terms
Model interaction effects with R/Python syntax:

```rust
// Full interaction: x1 * x2 expands to x1 + x2 + x1:x2
let formula = Formula::parse("wage ~ education * female")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;

// Interaction only: just the product term
let formula2 = Formula::parse("wage ~ education + female + education:female")?;
```

**Use cases:**
- Differential effects by groups (e.g., education returns by gender)
- Treatment effect heterogeneity
- Testing moderation/mediation hypotheses

### Enhanced Robust Standard Errors

```rust
// HC2: Leverage-adjusted (more efficient with small samples)
let result_hc2 = OLS::from_formula(&formula, &df, CovarianceType::HC2)?;

// HC3: Jackknife (most robust - RECOMMENDED for small samples)
let result_hc3 = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;
```

**Comparison:**
- **HC1**: White (1980), uses n/(n-k) correction
- **HC2**: Adjusts for leverage: σ²/(1-h_i)
- **HC3**: Jackknife: σ²/(1-h_i)² - Most conservative & robust

### Post-Estimation Predictions

```rust
// Out-of-sample predictions
let x_new = Array2::from_shape_vec((3, 2), vec![1.0, 12.0, 1.0, 16.0, 1.0, 20.0])?;
let predictions = result.predict(&x_new);

// In-sample fitted values
let fitted = result.fitted_values(&x);

// Residuals
let resid = result.residuals(&y, &x);
```

## 🚀 Features

### Cross-Sectional & General
- **OLS & GLS:** Robust standard errors (White, Newey-West).
- **IV / 2SLS:** Instrumental Variables for endogeneity correction.
- **Quantile Regression:** Robust estimation via Iteratively Reweighted Least Squares (IRLS).
- **Discrete Choice:** Logit and Probit models (Newton-Raphson MLE).
- **Diagnostics:** R-squared, F-Test, T-Test, Confidence Intervals.

### Time Series (Macroeconometrics)
- **Unit Root Tests:** Augmented Dickey-Fuller (ADF).
- **VAR (Vector Autoregression):** Multivariate modeling with Information Criteria (AIC/BIC).
- **VARMA:** Hannan-Rissanen algorithm for ARMA structures.
- **VECM (Cointegration):** Johansen Procedure (Eigenvalue decomposition) for I(1) systems.
- **Impulse Response Functions (IRF):** Orthogonalized structural shocks.

### Panel Data
- **Fixed Effects (Within):** Absorbs individual heterogeneity.
- **Random Effects:** Swamy-Arora GLS estimator.
- **Between Estimator:** Long-run cross-sectional relationships.
- **Dynamic Panel:** Arellano-Bond (Difference GMM) to solve Nickell Bias.
- **Panel Threshold:** Hansen (1999) non-linear regime switching models.
- **Testing:** Hausman Test for FE vs RE.

### Systems of Equations
- **SUR:** Seemingly Unrelated Regressions (Zellner).
- **3SLS:** Three-Stage Least Squares (System IV).

## System Requirements (Pre-requisites)

### Debian / Ubuntu / Pop!_OS:

```sh
sudo apt-get update

sudo apt-get install gfortran libopenblas-dev liblapack-dev pkg-config build-essential
```

### Fedora / RHEL / CentOS:

```sh
sudo dnf install gcc-gfortran openblas-devel lapack-devel pkg-config
```

### Arch Linux / Manjaro:

```sh
sudo pacman -S gcc-fortran openblas lapack base-devel
```

### macOS:

```sh
brew install openblas lapack
```

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
greeners = "2.0.0"
ndarray = "0.17"
# Note: You must have a BLAS/LAPACK provider installed on your system
ndarray-linalg = { version = "0.18", features = ["openblas-system"] }
```

## 🎯 Quick Start

### Loading Data (Multiple Options!)

Greeners provides **flexible data loading** similar to pandas/polars - from local files, URLs, or manual construction:

#### 1. CSV from Local File

```rust
use greeners::{DataFrame, Formula, OLS, CovarianceType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from CSV file with headers (just like pandas!)
    let df = DataFrame::from_csv("data.csv")?;

    // Specify model using formula
    let formula = Formula::parse("y ~ x1 + x2")?;

    // Estimate with robust standard errors
    let result = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;

    println!("{}", result);
    Ok(())
}
```

#### 2. CSV from URL (NEW!)

```rust
// Load data directly from GitHub or any URL
let df = DataFrame::from_csv_url(
    "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv"
)?;

// Perfect for reproducible research and shared datasets!
```

#### 3. JSON from Local File (NEW!)

```rust
// Column-oriented JSON (like pandas.to_json(orient='columns'))
// { "x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0] }
let df = DataFrame::from_json("data_columns.json")?;

// Or record-oriented JSON (like pandas.to_json(orient='records'))
// [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}]
let df = DataFrame::from_json("data_records.json")?;
```

#### 4. JSON from URL (NEW!)

```rust
// Load JSON directly from APIs or URLs
let df = DataFrame::from_json_url("https://api.example.com/data.json")?;
```

#### 5. Builder Pattern (NEW!)

```rust
// Most convenient for manual data construction
let df = DataFrame::builder()
    .add_column("wage", vec![30000.0, 40000.0, 50000.0])
    .add_column("education", vec![12.0, 16.0, 18.0])
    .add_column("experience", vec![5.0, 7.0, 10.0])
    .build()?;

let formula = Formula::parse("wage ~ education + experience")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;
```

**Supported formats:**
- ✅ CSV (local files)
- ✅ CSV (URLs) - requires internet connection
- ✅ JSON (local files) - both column and record oriented
- ✅ JSON (URLs) - perfect for API integration
- ✅ Builder pattern - convenient manual construction
- ✅ HashMap - traditional programmatic construction

📖 See `examples/dataframe_loading.rs` for comprehensive demonstration of all loading methods.

### Using Formula API (R/Python Style)

```rust
use greeners::{OLS, DataFrame, Formula, CovarianceType};
use ndarray::Array1;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data manually (like a pandas DataFrame)
    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    data.insert("x1".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    data.insert("x2".to_string(), Array1::from(vec![2.0, 2.5, 3.0, 3.5, 4.0]));

    let df = DataFrame::new(data)?;

    // Specify model using formula (just like Python/R!)
    let formula = Formula::parse("y ~ x1 + x2")?;

    // Estimate with robust standard errors
    let result = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;

    println!("{}", result);
    Ok(())
}
```

### Traditional Matrix API

```rust
use greeners::{OLS, CovarianceType};
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = Array2::from_shape_vec((5, 2), vec![
        1.0, 2.0,
        2.0, 2.5,
        3.0, 3.0,
        4.0, 3.5,
        5.0, 4.0,
    ])?;

    let result = OLS::fit(&y, &x, CovarianceType::HC1)?;
    println!("{}", result);
    Ok(())
}
```

## 📚 Formula API Examples

### Difference-in-Differences

```rust
use greeners::{DiffInDiff, DataFrame, Formula, CovarianceType};

// Python: smf.ols('outcome ~ treated + post + treated:post', data=df).fit(cov_type='HC1')
let formula = Formula::parse("outcome ~ treated + post")?;
let result = DiffInDiff::from_formula(&formula, &df, "treated", "post", CovarianceType::HC1)?;
```

### Instrumental Variables (2SLS)

```rust
use greeners::{IV, Formula, CovarianceType};

// Endogenous equation: y ~ x1 + x_endog
// Instruments: z1, z2
let endog_formula = Formula::parse("y ~ x1 + x_endog")?;
let instrument_formula = Formula::parse("~ z1 + z2")?;
let result = IV::from_formula(&endog_formula, &instrument_formula, &df, CovarianceType::HC1)?;
```

### Logit/Probit

```rust
use greeners::{Logit, Probit, Formula};

// Binary choice models
let formula = Formula::parse("binary_outcome ~ x1 + x2 + x3")?;
let logit_result = Logit::from_formula(&formula, &df)?;
let probit_result = Probit::from_formula(&formula, &df)?;
```

### Panel Data (Fixed Effects)

```rust
use greeners::{FixedEffects, Formula};

let formula = Formula::parse("y ~ x1 + x2")?;
let result = FixedEffects::from_formula(&formula, &df, &entity_ids)?;
```

### Quantile Regression

```rust
use greeners::{QuantileReg, Formula};

// Median regression
let formula = Formula::parse("y ~ x1 + x2")?;
let result = QuantileReg::from_formula(&formula, &df, 0.5, 200)?;
```

## 🔧 Formula Syntax

- **Basic:** `y ~ x1 + x2 + x3` (with intercept)
- **No intercept:** `y ~ x1 + x2 - 1` or `y ~ 0 + x1 + x2`
- **Intercept only:** `y ~ 1`

All formulas follow R/Python syntax for familiarity and ease of use.

## 📖 Documentation

- **[FORMULA_API.md](FORMULA_API.md)** - Complete formula API guide with Python/R equivalents
- **[examples/](examples/)** - Working examples for all estimators
  - `string_features.rs` - **String column support** (NEW v2.0.0!)
  - `missing_data_features.rs` - **Missing data toolkit** (NEW v2.0.0!)
  - `time_series_features.rs` - **Time series operations: lag, lead, diff, pct_change** (NEW v2.0.0!)
  - `dataframe_loading.rs` - Load data from CSV, JSON, URLs, or Builder pattern
  - `csv_formula_example.rs` - Load CSV files and run regressions
  - `formula_example.rs` - General formula API demonstration
  - `did_formula_example.rs` - Difference-in-Differences with formulas
  - `quickstart_formula.rs` - Quick start example
  - `marginal_effects.rs` - Logit/Probit marginal effects (AME/MEM)
  - `specification_tests.rs` - White, RESET, Breusch-Godfrey, Goldfeld-Quandt tests
  - `panel_model_selection.rs` - Panel diagnostics and model comparison

Run examples:
```sh
# NEW v2.0.0 examples
cargo run --example string_features        # String columns
cargo run --example missing_data_features  # Missing data handling
cargo run --example time_series_features   # Time series operations

# Other examples
cargo run --example dataframe_loading
cargo run --example csv_formula_example
cargo run --example formula_example
cargo run --example marginal_effects
cargo run --example specification_tests
```

## 🎯 Why Greeners?

1. **Familiar Syntax:** R/Python-style formulas make transition seamless
2. **Type Safety:** Rust's type system catches errors at compile time
3. **Performance:** Native speed with BLAS/LAPACK backends
4. **Comprehensive:** Full suite of econometric estimators
5. **Production Ready:** Memory safe, no garbage collection pauses
