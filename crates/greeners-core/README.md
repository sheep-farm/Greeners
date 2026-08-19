# greeners-core

Foundational data structures, utilities and numerical building blocks for the
Greeners econometrics workspace.

## Scope

- **Data structures:** `DataFrame`, `Column`, `DataType`, `Formula`.
- **Linear algebra helpers:** `LinalgInverse`, `LinalgSVD`, `LinalgQR`, etc.
- **Statistical utilities:** `Stats`, `DescrStatsW`, `MultipleTests`,
  `ProportionTests`, `Bootstrap`.
- **Distributions:** `chi2_pvalue`, `t_pvalue_two`, `norm_pdf`, etc.
- **Multivariate methods:** `PCA`, `MANOVA`, `FactorAnalysis`, `CanCorr`.
- **Non-parametric tools:** `KDEUnivariate`, `KernelReg`, `Lowess`.
- **I/O and predicates:** `Datasets`, `RowPredicate`.
- **Common error type:** `GreenersError`.

## Usage

```toml
[dependencies]
greeners-core = "2.0"
```

```rust
use greeners_core::{DataFrame, Formula, Stats};

let df = DataFrame::from_columns(vec![
    ("y".to_string(), vec![1.0, 2.0, 3.0]),
    ("x".to_string(), vec![1.0, 2.0, 3.0]),
]).unwrap();

let formula = Formula::parse("y ~ x").unwrap();
let stats = Stats::new(&df);
```

## Design notes

- Items with unique names within the crate are re-exported at the crate root.
- Items whose names appear in more than one module remain namespaced under
  `greeners_core::<module>`.
- This crate is the foundation of the workspace: all other Greeners crates
depend on it.
