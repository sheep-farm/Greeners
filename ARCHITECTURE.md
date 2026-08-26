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

The `greeners` crate is a curated facade. It re-exports every public module
from the sub-crates and a selected set of the most common items at the root.
There are no glob re-exports (`pub use ...::*`) and no
`#[allow(ambiguous_glob_reexports)]`.

Whenever a sub-crate re-exports an item at its own root, the facade imports
from the crate root:

```rust
pub use greeners_ols::OLS;
```

otherwise it falls back to the module path:

```rust
pub use greeners_ols::ols::OLS;
```

This allows users to write either:

```rust
use greeners::OLS;
```

or the namespaced version:

```rust
use greeners_ols::OLS;
```

The facade contains:

- `pub mod export;` for the cross-cutting I/O utility.
- `pub use crate_name::module;` for every module of every sub-crate.
- `pub use crate_name::{Item};` or `pub use crate_name::module::{Item};` for
  curated top-level items.
- a small set of smoke / cross-cutting tests in `greeners/tests/`.

Each sub-crate also re-exports public items that are unique within the crate
at its own root `lib.rs`. Items whose names appear in more than one module
remain namespaced.

### Keeping the facade in sync

`scripts/check_facade.py` regenerates the facade from the public items used by
`greeners/tests/` and `greeners/examples/` plus a base list of common types.
Run it before committing facade changes:

```bash
python3 scripts/check_facade.py
```

If the script reports the facade is out of date, update the `BASE` set in the
script and/or adjust the facade imports, then rerun the check until it passes.

## Committing changes

Before committing, run:

```bash
cargo fmt --check
cargo clippy -- -D warnings
python3 scripts/check_facade.py
cargo doc
cargo test
cargo deny check
cargo build --release
```

A GitHub Actions workflow in `.github/workflows/ci.yml` runs the same checks on
push and pull request. A local pre-commit hook is also available in
`.githooks/pre-commit`.

## Known architectural trade-offs

- `greeners-core` is still the largest crate. It mixes data structures
  (`DataFrame`, `Column`, `Formula`), error types, I/O, and numerical
  utilities (`linalg`, `stats`, `multivariate`, `nonparametric`). A future step
  could split it into `greeners-data`, `greeners-linalg`, `greeners-stats`, and
  `greeners-ml-core`, but this requires untangling the `linalg`/`GreenersError`
  dependency cycle first.
- `greeners-timeseries` contains 39 modules. It could become a small sub-workspace
  (`greeners-arima`, `greeners-garch`, `greeners-var`, etc.) once the core crate
  is further split.
- The facade root re-exports ~100 items. This is a compromise between
  convenience and namespace pollution. New common items should be added to
  `BASE` in `scripts/check_facade.py` explicitly.

Keep this file updated when adding, removing or renaming crates.
