# greeners-ols

Classical and related regression estimators for the Greeners workspace.

## Estimators

- **OLS** — ordinary least squares.
- **WLS** — weighted least squares.
- **FGLS / GLSAR** — feasible and AR-corrected generalized least squares.
- **IV** — instrumental variables / 2SLS.
- **GMM** — generalized method of moments.
- **NLS** — non-linear least squares.
- **Heckman / Tobit** — sample selection and censored regression.
- **QuantileReg** — quantile regression.
- **RLM** — robust linear models.
- **SUR / ThreeSLS** — seemingly unrelated regressions and three-stage LS.
- **FMOLS** — fully modified OLS for cointegrated panels.
- **RollingOLS / RollingWLS / RecursiveLS** — rolling and recursive estimation.
- **RegPath** — regularized regression paths.
- **EventStudy** — event-study estimation.

## Usage

```toml
[dependencies]
greeners-ols = "2.0"
```

```rust
use greeners_ols::{OLS, OlsResult};
use ndarray::Array2;

let y = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
let result = OLS::fit(&y, &x, Some(vec!["x1".to_string(), "x2".to_string()])).unwrap();
```

## Design notes

Unique public items are re-exported at the crate root. Estimators with
duplicated names remain under `greeners_ols::<module>`.
