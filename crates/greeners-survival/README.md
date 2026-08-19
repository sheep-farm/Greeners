# greeners-survival

Survival and duration analysis.

## Estimators

- **KaplanMeier** — non-parametric survival function.
- **CoxPH** — Cox proportional hazards.

## Usage

```toml
[dependencies]
greeners-survival = "2.0"
```

```rust
use greeners_survival::{CoxPH, KaplanMeier};

let cox = CoxPH::fit_with_names(&time, &status, &x, None, Some(cols)).unwrap();
```

## Design notes

All public estimators and results are re-exported at the crate root.
