# Formula API - R/Python-style Syntax for Greeners

Greeners now supports model specification using R/Python-style formulas (similar to Python's `statsmodels` and R's `lm()`).

## Basic Syntax

```rust
use greeners::{OLS, DataFrame, Formula, CovarianceType};
use ndarray::Array1;
use std::collections::HashMap;

// Create data
let mut data = HashMap::new();
data.insert("y".to_string(), Array1::from(vec![...]));
data.insert("x1".to_string(), Array1::from(vec![...]));
data.insert("x2".to_string(), Array1::from(vec![...]));

let df = DataFrame::new(data)?;

// Specify formula (R/Python syntax)
let formula = Formula::parse("y ~ x1 + x2")?;

// Estimate model
let result = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;
```

## Python (statsmodels) Equivalents

### Basic OLS

**Python:**
```python
import statsmodels.formula.api as smf
model = smf.ols('y ~ x1 + x2', data=df).fit()
```

**Rust (Greeners):**
```rust
let formula = Formula::parse("y ~ x1 + x2")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust)?;
```

### OLS with Robust Standard Errors (HC1)

**Python:**
```python
model = smf.ols('y ~ x1 + x2', data=df).fit(cov_type='HC1')
```

**Rust (Greeners):**
```rust
let formula = Formula::parse("y ~ x1 + x2")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;
```

### WLS (Weighted Least Squares)

**Python:**
```python
model = smf.wls('fte ~ treated + t + effect', data=df, weights=weights).fit(cov_type='HC1')
```

**Rust (Greeners):**
```rust
let formula = Formula::parse("fte ~ treated + t + effect")?;
let result = FGLS::wls_from_formula(&formula, &df, &weights)?;
```

### No Intercept

**Python:**
```python
model = smf.ols('y ~ x1 + x2 - 1', data=df).fit()
# Or: model = smf.ols('y ~ 0 + x1 + x2', data=df).fit()
```

**Rust (Greeners):**
```rust
let formula = Formula::parse("y ~ x1 + x2 - 1")?;
// Or: let formula = Formula::parse("y ~ 0 + x1 + x2")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::NonRobust)?;
```

## Formula Syntax

### General Format
```
dependent ~ independent1 + independent2 + ... + independentN
```

### Intercept Control

- **With intercept (default):**
  - `y ~ x1 + x2`
  - `y ~ 1 + x1 + x2` (explicit)

- **Without intercept:**
  - `y ~ x1 + x2 - 1`
  - `y ~ 0 + x1 + x2`

- **Intercept only:**
  - `y ~ 1`

## DataFrame Structure

The `DataFrame` is a simple structure for storing tabular data:

```rust
use greeners::DataFrame;
use ndarray::Array1;
use std::collections::HashMap;

let mut data = HashMap::new();
data.insert("column1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
data.insert("column2".to_string(), Array1::from(vec![4.0, 5.0, 6.0]));

let df = DataFrame::new(data)?;

// Access columns
let col = df.get("column1")?;

// Information
println!("Rows: {}", df.n_rows());
println!("Columns: {}", df.n_cols());
println!("Names: {:?}", df.column_names());
```

## Available Methods

### OLS
- `OLS::from_formula(formula, data, cov_type)` - Estimate OLS using formula

### FGLS/WLS
- `FGLS::wls_from_formula(formula, data, weights)` - WLS using formula
- `FGLS::cochrane_orcutt_from_formula(formula, data)` - Cochrane-Orcutt AR(1) using formula

### Difference-in-Differences (DiD)
- `DiffInDiff::from_formula(formula, data, treated_var, post_var, cov_type)` - DiD using formula

### IV/2SLS
- `IV::from_formula(endog_formula, instrument_formula, data, cov_type)` - IV/2SLS using formulas

### Discrete Choice Models
- `Logit::from_formula(formula, data)` - Logit using formula
- `Probit::from_formula(formula, data)` - Probit using formula

### Quantile Regression
- `QuantileReg::from_formula(formula, data, tau, n_boot)` - Quantile regression using formula

### Panel Data
- `FixedEffects::from_formula(formula, data, entity_ids)` - Fixed Effects using formula
- `RandomEffects::from_formula(formula, data, entity_ids)` - Random Effects using formula
- `BetweenEstimator::from_formula(formula, data, entity_ids)` - Between Estimator using formula

## Complete Examples

See examples in the `examples/` folder:
- `formula_example.rs` - General formula API demonstration
- `did_formula_example.rs` - Difference-in-Differences using formulas
- `quickstart_formula.rs` - Quick start example from README

Run with:
```bash
cargo run --example formula_example
cargo run --example did_formula_example
cargo run --example quickstart_formula
```

## Supported Covariance Types

```rust
pub enum CovarianceType {
    /// Standard errors (homoscedasticity assumed)
    NonRobust,

    /// White's robust errors (HC1) - heteroscedasticity only
    HC1,

    /// Newey-West (HAC) - heteroscedasticity + autocorrelation
    /// The usize parameter is the number of lags
    NeweyWest(usize),
}
```

## Current Limitations

1. **Interactions:** Not yet supported (e.g., `y ~ x1 * x2`)
2. **Functions:** Not yet supported (e.g., `y ~ log(x1)`)
3. **Categorical factors:** Not yet supported (e.g., `y ~ C(category)`)
4. **I():** Not yet supported (e.g., `y ~ I(x1**2)`)

These features may be added in future versions.

## Advantages

1. **Familiar syntax:** If you come from R or Python, the syntax is immediately recognizable
2. **Less verbose:** No need to manually construct matrices
3. **Type-safe:** Errors caught at compile time whenever possible
4. **Performance:** Native Rust speed with memory safety
5. **Integration:** Works with all existing estimators

## Performance

The formula API adds minimal overhead:
1. Formula parsing: O(k) where k = number of variables
2. Design matrix construction: O(n*k) where n = observations

For repeated analyses with the same structure, you can parse the formula once and reuse it.
