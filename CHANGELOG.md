# Changelog

All notable changes to the Greeners project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
