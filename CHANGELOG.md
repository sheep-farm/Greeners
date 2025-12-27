# Changelog

All notable changes to the Greeners project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]

### Planned Features
- Two-way clustering (clustered by both dimensions)
- Interaction terms in formulas (e.g., `y ~ x1 * x2`)
- Categorical variables and factor encoding (e.g., `y ~ C(category)`)
- Polynomial terms (e.g., `y ~ x + I(x^2)`)
- More covariance estimators (HC2, HC3, HC4)
- Wild bootstrap for hypothesis testing
- Post-estimation predictions and marginal effects
