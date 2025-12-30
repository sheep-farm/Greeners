# Changelog

All notable changes to the Greeners project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX 🎉 STABLE RELEASE

### Added
- **Specification Tests Module** - Comprehensive diagnostic tests for regression assumptions
  - `SpecificationTests::white_test(residuals, x)` - White's test for heteroskedasticity
    - Tests H₀: Homoskedasticity vs H₁: Heteroskedasticity
    - Uses auxiliary regression with u² on X and X²
    - Returns (LM_statistic, p_value, degrees_of_freedom)
    - If p < 0.05 → Use robust standard errors (HC1-HC4)

  - `SpecificationTests::reset_test(y, x, fitted, power)` - Ramsey RESET test for functional form
    - Tests H₀: Correctly specified vs H₁: Misspecification
    - Adds powers of fitted values (ŷ², ŷ³, ...) to regression
    - Returns (F_statistic, p_value, df_num, df_denom)
    - If p < 0.05 → Add polynomials, interactions, or transformations

  - `SpecificationTests::breusch_godfrey_test(residuals, x, lags)` - LM test for autocorrelation
    - Tests H₀: No autocorrelation vs H₁: AR(p) autocorrelation
    - More general than Durbin-Watson, allows lagged dependent variables
    - Returns (LM_statistic, p_value, degrees_of_freedom)
    - If p < 0.05 → Use Newey-West HAC standard errors

  - `SpecificationTests::goldfeld_quandt_test(residuals, split_fraction)` - Test for heteroskedasticity
    - Tests H₀: Homoskedasticity vs H₁: Variance differs across groups
    - Compares variance between first and last portions of ordered data
    - Returns (F_statistic, p_value, df1, df2)
    - Simple and intuitive alternative to White test

  - `SpecificationTests::print_test_result()` - Pretty-printed test results with interpretation

### Changed
- First stable release (v1.0.0) with production-ready API
- All core econometric estimators fully tested and documented
- Comprehensive test coverage for regression diagnostics

### Documentation
- Added comprehensive specification tests example (`examples/specification_tests.rs`)
- Demonstrates all v1.0.0 tests with realistic datasets:
  - Wage equation with heteroskedasticity
  - Time series consumption function with autocorrelation
- Complete interpretation guidelines and remedies
- Comparison with Stata/R/Python equivalents

### Examples
- `examples/specification_tests.rs`: Complete demonstration
  - White test detecting heteroskedasticity
  - RESET test detecting functional form issues
  - Breusch-Godfrey detecting autocorrelation
  - Goldfeld-Quandt comparing group variances
  - Remedies: robust SE, HAC SE, polynomial terms

### Stata/R/Python Equivalents
- **Stata**: `estat hettest` (White), `estat ovtest` (RESET), `estat bgodfrey`
- **R**: `lmtest::bptest()`, `lmtest::resettest()`, `lmtest::bgtest()`
- **Python**: `statsmodels.stats.diagnostic.het_white()`, `acorr_breusch_godfrey()`

### Migration to 1.0.0
- All previous features (v0.1-v0.9) remain fully compatible
- No breaking API changes from v0.9.0
- New specification tests are additive, not replacing existing diagnostics

## [0.9.0] - 2025-01-XX

### Added
- **Model Selection Utilities**
  - `ModelSelection::compare_models(models)` - Compare multiple models by AIC/BIC
  - `ModelSelection::akaike_weights(aic_values)` - Calculate Akaike weights for model averaging
  - `ModelSelection::print_comparison(comparison)` - Pretty-print model comparison table
  - Automatic ranking and sorting by information criteria
  - Δ_AIC interpretation guidelines (< 2: substantial, 4-7: less support, > 10: no support)

- **Panel Diagnostics Tests**
  - `PanelDiagnostics::breusch_pagan_lm(residuals, entity_ids)` - LM test for random effects
    - Tests H₀: σ²_u = 0 (no panel effect) vs H₁: σ²_u > 0 (random effects needed)
    - Uses chi-squared distribution with 1 degree of freedom
    - Essential for choosing between pooled OLS and panel data models
  - `PanelDiagnostics::f_test_fixed_effects(ssr_pooled, ssr_fe, n, n_entities, k)` - F-test for fixed effects
    - Tests H₀: All entity effects are zero vs H₁: Entity effects exist
    - Compares SSR from pooled OLS and fixed effects models
    - Critical for model selection in panel data analysis

- **Summary Statistics Helper**
  - `SummaryStats::describe(data)` - Comprehensive descriptive statistics
  - Returns (mean, std, min, Q25, median, Q75, max, n_obs)
  - `SummaryStats::print_summary(stats)` - Pretty-printed summary table
  - Essential for initial data exploration

### Documentation
- Added comprehensive panel diagnostics and model selection example
- Demonstrates all v0.9.0 features with realistic firm investment panel data
- Comparison with Stata/R/Python equivalents
- Decision tree for panel data model selection

### Examples
- `examples/panel_model_selection.rs`: Complete demonstration
  - Descriptive statistics for panel data
  - Multiple pooled OLS specifications
  - Model comparison by AIC/BIC and Akaike weights
  - Breusch-Pagan LM test for panel effects
  - Panel data model selection workflow

### Stata/R/Python Equivalents
- **Stata**: `estat ic` (AIC/BIC), `xttest0` (BP LM test), `testparm` (F-test)
- **R**: `AIC()`, `BIC()`, `plm::plmtest()`, `plm::pFtest()`
- **Python**: `linearmodels.panel` diagnostics, `statsmodels` information criteria

## [0.8.0] - 2025-01-XX

### Added
- **Bootstrap Methods for Inference**
  - `Bootstrap::pairs_bootstrap(y, x, n_replications)` - Pairs bootstrap with replacement
  - `Bootstrap::bootstrap_se(boot_coefs)` - Bootstrap standard errors
  - `Bootstrap::percentile_ci(boot_coefs, alpha)` - Percentile confidence intervals
  - Robust to non-normality, heteroscedasticity, and small samples
  - Essential for finite-sample inference and asymptotic skepticism

- **Hypothesis Testing Framework**
  - `HypothesisTest::wald_test(beta, cov, R, q)` - Wald test for linear restrictions
  - `HypothesisTest::joint_significance(beta, cov, has_intercept)` - Test all slopes = 0
  - `HypothesisTest::f_test_nested(ssr_r, ssr_f, n, k_f, k_r)` - F-test for nested OLS models
  - Flexible restriction matrices for complex hypotheses

### Documentation
- Added comprehensive bootstrap and hypothesis testing example
- Demonstrated all three testing approaches with wage data
- Comparison of asymptotic vs bootstrap inference

### Examples
- `examples/bootstrap_hypothesis.rs`: Complete demonstration
  - Bootstrap SE and confidence intervals
  - Wald tests (joint and specific restrictions)
  - F-test for model selection

## [0.7.0] - 2025-01-XX

### Added
- **Confidence Intervals for Marginal Effects**
  - `BinaryModelResult::ame_confidence_intervals(x, alpha)` - 95% CI for AME
  - Uses delta method with numerical approximation
  - Returns (lower_bounds, upper_bounds) for each marginal effect
  - Essential for hypothesis testing on marginal effects

- **Model Comparison Methods**
  - `BinaryModelResult::model_stats()` - Returns (AIC, BIC, LogLik, Pseudo R²)
  - `OlsResult::model_stats()` - Returns (AIC, BIC, LogLik, Adj R²)
  - Easy model comparison across specifications

- **Partial R² for OLS**
  - `OlsResult::partial_r_squared(indices, y, x)` - Contribution of variable subset
  - Measures incremental explanatory power
  - Useful for nested model comparisons

### Documentation
- Enhanced marginal effects documentation with CI examples
- Added model comparison examples

## [0.6.0] - 2025-01-XX

### Added
- **HC4 Covariance Estimator** (Cribari-Neto, 2004)
  - `CovarianceType::HC4` - Refined jackknife with adaptive power adjustment
  - Formula: σ²_i / (1 - h_i)^δᵢ where δᵢ = min(4, n·h_i/k)
  - Best small-sample performance, especially with influential observations
  - More refined than HC3 for datasets with high-leverage points
  - Implemented for both OLS and IV/2SLS

- **Predictions for IV/2SLS Models**
  - `IvResult::predict(x_new)` - Out-of-sample predictions
  - `IvResult::fitted_values(x)` - In-sample fitted values
  - `IvResult::residuals(y, x)` - Calculate residuals
  - Same API as OLS predictions for consistency
  - Essential for model validation and forecasting

### Changed
- Enhanced IV module with post-estimation methods
- Completed HC series (HC1, HC2, HC3, HC4) for comprehensive robustness options

### Documentation
- Updated examples with HC4 usage
- Added IV prediction examples

## [0.5.0] - 2025-01-XX

### Added
- **Two-Way Clustered Standard Errors** (Cameron-Gelbach-Miller, 2011)
  - `CovarianceType::ClusteredTwoWay(firm_ids, time_ids)`
  - Essential for panel data with two-dimensional dependence
  - Formula: V = V₁ + V₂ - V₁₂ (corrects for both firm and time clustering)
  - Implemented for both OLS and IV/2SLS
  - More robust than one-way clustering for panel data
  - See `examples/two_way_clustering.rs` for comprehensive demonstration

- **Marginal Effects for Binary Choice Models**: Post-estimation analysis for Logit/Probit
  - `average_marginal_effects(x)` - Average Marginal Effects (AME) - RECOMMENDED
    - Formula (Logit): AME_j = (1/n) Σ_i [β_j × exp(x'β)/(1+exp(x'β))²]
    - Formula (Probit): AME_j = (1/n) Σ_i [β_j × φ(x'β)] where φ is normal PDF
    - Averages marginal effects across all observations
    - Accounts for sample heterogeneity
    - Most robust estimator for policy/substantive interpretation
  - `marginal_effects_at_means(x)` - Marginal Effects at Means (MEM)
    - Evaluates marginal effect at sample means x̄
    - Faster but less robust than AME
    - Can give misleading results with dummy variables
  - `predict_proba(x)` - Predicted probabilities for new observations
    - Returns P(y=1|x) for Logit/Probit models
    - Essential for out-of-sample predictions and model validation
  - All methods work seamlessly with formula API
  - See `examples/marginal_effects.rs` for comprehensive usage

### Changed
- Enhanced `BinaryModelResult` struct to store design matrix for marginal effects
- Updated `Logit::fit()` and `Probit::fit()` to cache X data for post-estimation
- Marginal effects automatically detect model type (Logit vs Probit) for correct formula

### Documentation
- Added comprehensive marginal effects example (`examples/marginal_effects.rs`)
  - College admission example with GPA, SAT, legacy status
  - Demonstrates AME vs MEM comparison
  - Logit vs Probit comparison
  - Interpretation guidelines
  - Stata/R/Python equivalents
- Updated README.md with v0.5.0 marginal effects section
- Enhanced CHANGELOG.md with detailed formulas and usage notes

### Examples
- `examples/marginal_effects.rs`: Complete demonstration of marginal effects
  - Real-world college admission dataset
  - Both Logit and Probit estimation
  - AME and MEM calculation
  - Model comparison and predictions
  - Comprehensive interpretation guide

### Technical Details
- AME is computed by averaging marginal effects across all observations
- MEM is computed at the vector of sample means
- For continuous variables: ME_j = ∂P(y=1|x)/∂x_j
- For binary variables: ME_j represents discrete change from 0 to 1
- Marginal effects are in probability units (easy to interpret vs log-odds)

### Why Marginal Effects Matter
- Coefficients in Logit/Probit are on log-odds/z-score scale (hard to interpret)
- Marginal effects show PROBABILITY changes (intuitive for policy/research)
- Essential for:
  - Policy analysis ("What's the effect of X on outcomes?")
  - Treatment effect heterogeneity
  - Economic significance vs statistical significance
  - Comparing effects across different models

### Stata/R/Python Equivalents
- **Stata**: `margins, dydx(*)` for AME, `margins, dydx(*) atmeans` for MEM
- **R**: `mfx::logitmfx()`, `margins::margins()`
- **Python**: `statsmodels.discrete.discrete_model.*.get_margeff()`

## [0.4.0] - 2025-01-XX

### Added
- **Categorical Variable Encoding**: Automatic dummy variable creation with `C(var)` syntax
  - R/Python-style syntax: `y ~ x + C(region)`
  - Automatically detects unique categories from numerical values
  - Creates K-1 dummy variables (drops first category as reference)
  - Follows same convention as R's `factor()` and Python's `pd.get_dummies(drop_first=True)`
  - Essential for: regions, industries, treatment groups, demographic categories
  - Works seamlessly with all estimators and can be combined with interactions
  - See `examples/categorical_polynomial.rs` for comprehensive examples

- **Polynomial Terms**: Non-linear relationship modeling with `I(expr)` syntax
  - Supports `I(x^2)`, `I(x^3)`, ..., `I(x^n)` for any integer power
  - Alternative Python-style syntax: `I(x**2)`
  - Captures diminishing returns, growth curves, and non-linear effects
  - Use cases: production functions, wage-experience curves, growth models
  - Combine with categorical and interactions for rich specifications
  - Automatic computation of polynomial columns in design matrix

### Changed
- Enhanced `DataFrame::to_design_matrix()` to process `C()` and `I()` terms
- Added `DataFrame::count_design_matrix_cols()` to handle variable column expansion
- Formula parser now recognizes and preserves `C()` and `I()` expressions
- Design matrix allocation now accounts for categorical variable expansion

### Documentation
- Added comprehensive categorical and polynomial examples
- Updated README.md with v0.4.0 feature sections
- Enhanced formula documentation with new syntax

### Examples
- `examples/categorical_polynomial.rs`: Complete demonstration of C() and I() features
  - Categorical encoding with regional sales data
  - Polynomial modeling with production function
  - Model comparison (linear vs quadratic vs cubic)
  - Combined categorical + polynomial specifications

## [0.3.0] - 2025-01-XX

### Added
- **Interaction Terms in Formulas**: Full R/Python-style interaction syntax
  - `y ~ x1 * x2` expands to x1 + x2 + x1:x2 (full interaction)
  - `y ~ x1 : x2` adds only the interaction term (product)
  - Essential for modeling differential effects and treatment heterogeneity
  - Works seamlessly with all estimators (OLS, IV, Panel Data, etc.)
  - See `examples/v0_3_features.rs` for comprehensive examples

- **Additional Robust Covariance Estimators**: HC2 and HC3
  - `CovarianceType::HC2`: Leverage-adjusted heteroscedasticity-robust SE
    - Formula: σ²_i / (1 - h_i)
    - More efficient than HC1 with small samples
  - `CovarianceType::HC3`: Jackknife heteroscedasticity-robust SE
    - Formula: σ²_i / (1 - h_i)²
    - Most robust for small samples (MacKinnon & White, 1985)
    - **Recommended as default robust SE estimator**
  - Implemented for both OLS and IV/2SLS estimators

- **Post-Estimation Methods**: Predictions and residual analysis
  - `OlsResult::predict(&x_new)`: Out-of-sample predictions for new data
  - `OlsResult::fitted_values(&x)`: In-sample fitted values
  - `OlsResult::residuals(&y, &x)`: Calculate residuals
  - Essential for model validation and forecasting

### Changed
- Updated formula parser to handle interaction operators (`*` and `:`)
- Enhanced DataFrame::to_design_matrix to compute interaction columns automatically
- Display output now shows HC2 and HC3 covariance types

### Documentation
- Added comprehensive v0.3.0 features example
- Updated README.md with interaction, HC2/HC3, and prediction examples
- Updated FORMULA_API.md with interaction syntax documentation

### Examples
- `examples/v0_3_features.rs`: Complete demo of all v0.3.0 features

## [0.2.0] - 2025-01-XX

### Added
- **Clustered Standard Errors**: New `CovarianceType::Clustered(Vec<usize>)` variant for panel data and hierarchical structures
  - Implements cluster-robust variance-covariance matrix estimation
  - Corrects for within-cluster correlation in observations
  - Critical for panel data, experiments, and grouped observations
  - Supports both OLS and IV/2SLS estimators
  - See `examples/clustered_se_example.rs` for usage

- **Advanced Diagnostics Module**: Four new diagnostic methods in `Diagnostics`
  - `vif()`: Variance Inflation Factor for detecting multicollinearity per predictor
  - `condition_number()`: Overall multicollinearity assessment via SVD
  - `leverage()`: Hat values to identify high-leverage observations
  - `cooks_distance()`: Cook's D to detect influential observations
  - See `examples/diagnostics_example.rs` for comprehensive usage

### Changed
- Updated `OLS::fit()` to support clustered standard errors
- Updated `IV::fit()` to support clustered standard errors
- Enhanced display output to show cluster count when using clustered SE

### Documentation
- Updated README.md with v0.2.0 features
- Updated FORMULA_API.md with clustered SE examples and guidelines
- Added comprehensive examples demonstrating new features
- Added CHANGELOG.md for version tracking

### Examples
- `examples/clustered_se_example.rs`: Demonstrates clustered SE with panel data
- `examples/diagnostics_example.rs`: Comprehensive diagnostics workflow

## [0.1.2] - 2024-XX-XX

### Added
- CSV file support via `DataFrame::from_csv()`
- Formula API for R/Python-style model specification
- Support for all estimators: OLS, WLS, DiD, IV/2SLS, Logit/Probit, Quantile, Panel Data

### Changed
- Improved error messages for formula parsing
- Enhanced DataFrame functionality

## [0.1.1] - 2024-XX-XX

### Added
- Initial release with core econometric estimators
- OLS, IV/2SLS, Panel Data (FE/RE), Time Series (VAR/VECM)
- Difference-in-Differences, Logit/Probit, Quantile Regression
- GMM, SUR, 3SLS, Arellano-Bond
- Basic diagnostics: Jarque-Bera, Breusch-Pagan, Durbin-Watson

## [2.0.0] - 2025-01-29 🎉 MAJOR RELEASE: Complete Data Handling & Time Series

### Summary
**Greeners v2.0.0** represents a **major milestone** with comprehensive data handling capabilities matching pandas/polars and essential time series operations for econometric analysis. This release completes the core DataFrame API and time series toolkit.

### Added - Three Major Feature Sets

#### v1.7.0: String Column Support
- **`DataFrame::add_string(name, values)`** - Create string columns for free text
  - Store names, emails, addresses, comments, descriptions
  - Variable-length text without encoding (unlike Categorical)
  - Full integration with all DataFrame operations
- **`DataFrame::get_string(name)`** - Access string data
- **String vs Categorical distinction:**
  - String: Free text, unique values, variable length (names, addresses)
  - Categorical: Repeated categories, encoded as integers (regions, groups)
- **Operations:** concat, filter, select, head/tail preserve strings
- **Export:** to_csv() and to_json() preserve string columns
- **Missing data:** Empty strings represent missing (no NaN concept)
- See `examples/string_features.rs` for comprehensive demonstration

#### v1.8.0: Missing Data & Null Support
- **`DataFrame::isna(column)`** - Boolean mask for missing values
- **`DataFrame::notna(column)`** - Boolean mask for non-missing values
- **`DataFrame::dropna()`** - Remove rows with any NaN
- **`DataFrame::dropna_subset(cols)`** - Remove rows with NaN in specific columns
- **`DataFrame::fillna(column, value)`** - Fill missing with constant
- **`DataFrame::fillna_ffill(column)`** - Forward fill (carry forward)
- **`DataFrame::fillna_bfill(column)`** - Backward fill (carry backward)
- **`DataFrame::interpolate(column)`** - Linear interpolation for gaps
- **Comprehensive missing data workflow:**
  - Detect: isna/notna for investigation
  - Handle: dropna for complete-case analysis
  - Impute: fillna/ffill/bfill/interpolate for missing value treatment
- **Type-aware:** Works with Float, Int, Bool, DateTime columns
- **Bool columns:** Returns false for count_na() (no missing concept)
- See `examples/missing_data_features.rs` for complete demonstration

#### v1.9.0: Time Series Operations
- **`DataFrame::lag(column, periods)`** - Create lagged variables (y_{t-n})
  - Essential for autoregressive models: AR(p), ARIMA
  - Returns NaN for first `periods` observations
  - Example: `df.lag("price", 1)` creates `price_lag_1`
- **`DataFrame::lead(column, periods)`** - Create lead variables (y_{t+n})
  - Forward-looking analysis and causality testing
  - Returns NaN for last `periods` observations
  - Example: `df.lead("sales", 1)` creates `sales_lead_1`
- **`DataFrame::diff(column, periods)`** - First differences (y_t - y_{t-n})
  - Essential for achieving stationarity
  - Used in unit root tests and differenced models
  - Example: `df.diff("gdp", 1)` creates `gdp_diff_1`
- **`DataFrame::pct_change(column, periods)`** - Percentage changes
  - Standard for financial returns: (y_t - y_{t-n}) / y_{t-n}
  - Division by zero returns NaN
  - Example: `df.pct_change("close", 1)` creates `close_pct_1`
- **Mathematical relationships:**
  - lag(x, n)[t] = x[t-n]
  - lead(x, n)[t] = x[t+n]
  - diff(x, n)[t] = x[t] - x[t-n]
  - pct_change(x, n)[t] = diff(x, n)[t] / lag(x, n)[t]
- **Error handling:**
  - InvalidOperation if periods = 0
  - VariableNotFound if column doesn't exist
- **Use cases:**
  - Finance: Returns, momentum strategies
  - Econometrics: AR models, stationarity testing, GDP growth
  - Machine Learning: Time series feature engineering
- See `examples/time_series_features.rs` for 11 practical examples

### Changed
- **Cargo.toml**: Version bumped to 2.0.0
- **Error types**: Added `InvalidOperation` variant to `GreenersError`
- **Column types**: String and enhanced missing data handling
- **Test coverage**: 102 total tests (added 17 new time series tests)

### Documentation
- **README.md**: Updated with v2.0.0 features and examples
- **CHANGELOG.md**: Complete documentation of v1.7.0, v1.8.0, v1.9.0
- **Examples:**
  - `examples/string_features.rs` (467 lines) - String column demonstration
  - `examples/missing_data_features.rs` - Missing data workflow
  - `examples/time_series_features.rs` (467 lines) - Time series operations

### Migration Guide from v1.0.2 to v2.0.0
All v1.0.2 features remain **fully compatible** - no breaking changes!

**New capabilities:**
```rust
// 1. String columns
let df = DataFrame::builder()
    .add_string("name", vec!["Alice".to_string(), "Bob".to_string()])
    .add_column("score", vec![85.0, 92.0])
    .build()?;

// 2. Missing data handling
let clean = df.dropna();  // Remove rows with NaN
let filled = df.fillna_ffill("score")?;  // Forward fill
let interpolated = df.interpolate("price")?;  // Linear interpolation

// 3. Time series operations
let with_lag = df.lag("price", 1)?;  // Previous period
let returns = df.pct_change("price", 1)?;  // Percentage change
let stationary = df.diff("gdp", 1)?;  // First difference
```

### Performance
- String columns: Variable memory per value (use Categorical for repeated values)
- Missing data ops: O(n) time complexity
- Time series ops: O(n) with NaN-safe operations
- All operations leverage ndarray for vectorized performance

### Quality Metrics
- **Test coverage**: 102/102 tests passing
- **Code quality**: Clippy clean, idiomatic Rust
- **Documentation**: 3 comprehensive examples (1,400+ lines total)
- **API stability**: v2.0.0 marks stable DataFrame API

### Comparison with Other Libraries
**Greeners v2.0.0 now offers:**
- ✅ pandas-like data handling (string columns, missing data, time series)
- ✅ statsmodels-like econometric methods (all existing estimators)
- ✅ Rust type safety and performance
- ✅ Production-ready memory safety

**What's unique:**
- Only Rust library with complete econometric + DataFrame capabilities
- Type-safe time series operations (compile-time guarantees)
- Zero-copy design matrix creation from formulas
- BLAS/LAPACK backend for maximum performance

### Stata/R/Python Equivalents

**String columns:**
- Python: `df['name'] = ['Alice', 'Bob']` (pandas Series with dtype='object')
- R: `df$name <- c('Alice', 'Bob')` (character vector)
- Stata: `gen str name = "Alice"` (string variable)

**Missing data:**
- Python: `df.isna()`, `df.dropna()`, `df.fillna()`, `df.interpolate()`
- R: `is.na()`, `na.omit()`, `na.fill()`, `na.approx()`
- Stata: `misstable`, `drop if missing()`, `replace x = ...`

**Time series:**
- Python: `df['x'].shift(1)`, `df['x'].diff()`, `df['x'].pct_change()`
- R: `lag(x, 1)`, `diff(x)`, `Delt(x)`
- Stata: `L.x`, `D.x`, (x - L.x)/L.x`

### Future Roadmap (v2.1.0+)
- ✅ Core DataFrame operations (COMPLETE in v2.0.0)
- ✅ Time series basics (COMPLETE in v2.0.0)
- ⏳ Advanced time series: rolling window statistics
- ⏳ Group-by operations: aggregations by categorical variables
- ⏳ Merge/join operations: combining multiple DataFrames
- ⏳ Reshaping: pivot, melt, stack/unstack

## [Unreleased]

### Planned Features
- Rolling window aggregations (mean, sum, std, min, max)
- Group-by operations with aggregations
- DataFrame merge/join operations
- Pivot tables and data reshaping
