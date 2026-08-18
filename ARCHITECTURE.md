# Greeners Workspace Architecture

This document describes the 2.0 workspace refactor and the rules for placing
new estimators, tests and dependencies.

## Crate overview

| crate | responsibility | typical content |
|-------|---------------|-----------------|
| `greeners-core` | Foundational data structures and utilities | `DataFrame`, `Formula`, `Column`, `CovarianceType`, error types, `linalg`, `distributions`, `datasets`, `export` I/O |
| `greeners-ols` | Classical linear and related regression | `OLS`, `WLS`, `FGLS`, `IV`, `GLSAR`, `Heckman`, `Tobit`, `Quantile`, `SUR`, `GMM`, `NLS`, `RLM` helpers |
| `greeners-glm` | Generalized linear models and discrete choice | `GLM`, `Logit`, `Probit`, `Poisson`, `NegBin`, `BetaModel`, `GEE`, `GLMGam` |
| `greeners-panel` | Panel / longitudinal data | fixed/random effects, `PanelVAR`, `PSTR`, dynamic panel, `SystemGmm` |
| `greeners-timeseries` | Time series models | `VAR`, `VARMA`, `VECM`, `ARIMA`, `GARCH`, `MarkovSwitching`, `LocalLevel`, `KalmanFilter` |
| `greeners-bayesian` | Bayesian econometrics | `BVAR`, `FAVAR`, `MFVAR`, Bayesian shrinkage |
| `greeners-causal` | Causal inference | `DMLCrossfit`, `DiD`, `PSM`, `SyntheticControl`, `TMLE`, `RD` |
| `greeners-ml` | Machine learning | `RandomForest`, `GradientBoosting`, `MLP`, `KMeans`, `t-SNE` |
| `greeners-spatial` | Spatial econometrics | `Spatial`, `SpatialDurbin`, `SpatialPanel` |
| `greeners-survival` | Survival analysis | `KaplanMeier`, `CoxPH` |
| `greeners-imputation` | Missing data | `Mice`, imputation helpers |
| `greeners-diagnostics` | Model diagnostics and selection | `Diagnostics`, `SpecificationTests`, `ModelSelection` |
| `greeners` | Facade | re-exports all modules and their public items; contains cross-cutting `export` I/O and smoke tests |

## Dependency rules

1. **No crate may depend on `greeners` (the facade).** The facade is only a
   re-export surface; other crates depend on concrete sub-crates.
2. **Down-only dependencies.** Crates closer to the foundation should not depend
   on crates higher in the stack. `greeners-core` is the foundation.
3. **No test-only production dependencies.** If a test in crate A needs a type
   from crate B, either move the test to crate B, place it in the facade tests,
   or generate synthetic data without crossing domains.
4. **Workspace inheritance is required.** Shared package metadata and dependency
   versions are declared in the root `Cargo.toml` and inherited via
   `workspace = true`.

## Where to add new estimators

Ask these questions:

1. **What is the primary domain?** Put it in the thematic crate (`ols`, `glm`,
   `timeseries`, `panel`, `ml`, etc.).
2. **Does it need `DataFrame`/`Formula`?** It will live in a crate above
   `greeners-core`.
3. **Is it a tiny utility used by many crates?** It probably belongs in
   `greeners-core`.
4. **Does it re-export many result types?** Use `greeners-core::export` (now in
   the facade) for generating tabular/CSV/JSON output, not for the estimator
   itself.

## Fachada `greeners`

The `greeners` crate intentionally re-exports every public module and its
public items:

```rust
pub use greeners_ols::ols;
pub use greeners_ols::ols::*;
```

This allows users to write either:

```rust
use greeners::OLS;
```

or the namespaced version:

```rust
use greeners_ols::ols::OLS;
```

The facade contains only:

- `pub mod export;` for the cross-cutting I/O utility.
- re-exports from sub-crates.
- a small set of smoke / cross-cutting tests in `greeners/tests/`.

## Committing changes

Before committing, run:

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
cargo doc
cargo deny check
cargo build --release
```

Keep this file updated when adding, removing or renaming crates.
