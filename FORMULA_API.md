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

### Loading from CSV (Recommended!)

The easiest way to work with data is to load it from a CSV file with headers:

```rust
use greeners::DataFrame;

// Load data from CSV file (just like pandas.read_csv!)
let df = DataFrame::from_csv("data.csv")?;

println!("Loaded {} rows and {} columns", df.n_rows(), df.n_cols());
println!("Column names: {:?}", df.column_names());

// Access individual columns
let y_column = df.get("y")?;
```

### Creating Manually

You can also create DataFrames manually:

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
- **`csv_formula_example.rs`** - Load CSV files and run regressions (RECOMMENDED!)
- `formula_example.rs` - General formula API demonstration
- `did_formula_example.rs` - Difference-in-Differences using formulas
- `quickstart_formula.rs` - Quick start example from README

Run with:
```bash
cargo run --example csv_formula_example
cargo run --example formula_example
cargo run --example did_formula_example
cargo run --example quickstart_formula
```

### CSV File Format

Your CSV file should have headers in the first row:

```csv
y,x1,x2,x3
10.5,1.2,2.3,0.5
12.3,2.1,3.1,0.7
15.2,3.5,4.2,0.9
...
```

Then load and use with formulas:

```rust
let df = DataFrame::from_csv("data.csv")?;
let formula = Formula::parse("y ~ x1 + x2")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::HC1)?;
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

    /// Clustered standard errors - for panel/grouped data
    /// The Vec<usize> contains cluster IDs for each observation
    /// Critical for panel data and hierarchical structures
    Clustered(Vec<usize>),
}
```

### Using Clustered Standard Errors (NEW in v0.2.0)

Clustered standard errors account for within-cluster correlation:

```rust
use greeners::{OLS, DataFrame, Formula, CovarianceType};

// Panel data example: 3 firms × 5 time periods = 15 observations
let cluster_ids = vec![
    0,0,0,0,0,  // Firm 0 (5 time periods)
    1,1,1,1,1,  // Firm 1 (5 time periods)
    2,2,2,2,2,  // Firm 2 (5 time periods)
];

let formula = Formula::parse("profit ~ advertising + rd_spending")?;
let result = OLS::from_formula(&formula, &df, CovarianceType::Clustered(cluster_ids))?;

println!("{}", result);  // Shows "Clustered (3 clusters)" in output
```

**When to use clustered SE:**
- Panel data (repeated observations per entity: firms, individuals, countries)
- Hierarchical data (students in schools, patients in hospitals)
- Experiments with cluster randomization (villages, classrooms)
- Geographic clustering (cities, regions, countries)
- Any situation where observations within groups are likely correlated

**Why it matters:**
- Standard OLS SE assume independence → underestimated standard errors
- Clustered SE correct for within-cluster correlation → proper inference
- Ignoring clustering leads to over-rejection of null hypotheses (false discoveries)

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
