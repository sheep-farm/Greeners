# Greeners: High-Performance Econometrics in Rust

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-0.2.0-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)

**Greeners** is a lightning-fast, type-safe econometrics library written in pure Rust. It provides a comprehensive suite of estimators for Cross-Sectional, Time-Series, and Panel Data analysis, leveraging linear algebra backends (LAPACK/BLAS) for maximum performance.

Designed for academic research, heavy simulations, and production-grade economic modeling.

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

```bash
sudo apt-get update

sudo apt-get install gfortran libopenblas-dev liblapack-dev pkg-config build-essential
```

### Fedora / RHEL / CentOS:

```bash
sudo dnf install gcc-gfortran openblas-devel lapack-devel pkg-config
```

### Arch Linux / Manjaro:

```bash
sudo pacman -S gcc-fortran openblas lapack base-devel
```

### macOS:

```bash
brew install openblas lapack
```

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
greeners = "0.1.1"
ndarray = "0.15"
# Note: You must have a BLAS/LAPACK provider installed on your system
ndarray-linalg = { version = "0.14", features = ["openblas"] }
```

## 🎯 Quick Start

### Loading Data from CSV (NEW!)

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
  - `csv_formula_example.rs` - **Load CSV files and run regressions**
  - `formula_example.rs` - General formula API demonstration
  - `did_formula_example.rs` - Difference-in-Differences with formulas
  - `quickstart_formula.rs` - Quick start example

Run examples:
```bash
cargo run --example csv_formula_example
cargo run --example formula_example
cargo run --example did_formula_example
```

## 🎯 Why Greeners?

1. **Familiar Syntax:** R/Python-style formulas make transition seamless
2. **Type Safety:** Rust's type system catches errors at compile time
3. **Performance:** Native speed with BLAS/LAPACK backends
4. **Comprehensive:** Full suite of econometric estimators
5. **Production Ready:** Memory safe, no garbage collection pauses