# greeners-imputation

Missing-data imputation methods.

## Methods

- **MICE** — multiple imputation by chained equations.
- **MiceChained** — chained-equations imputation.
- **BayesGaussMI** — Bayesian Gaussian multiple imputation.

## Usage

```toml
[dependencies]
greeners-imputation = "2.0"
```

```rust
use greeners_imputation::{MICE, BayesGaussMI};

let imputed = MICE::impute(&df, 5, 10).unwrap();
```

## Design notes

Imputation methods and results are re-exported at the crate root.
