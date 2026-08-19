# greeners

Curated public facade for the Greeners econometrics workspace.

This crate re-exports every public module from the thematic sub-crates and a
selected set of common items at the crate root. It is the intended entry point
for downstream consumers such as the Hayashi interpreter.

## Usage

Add this crate to your `Cargo.toml`:

```toml
[dependencies]
greeners = "2.0"
```

Then import the most common estimators and types directly:

```rust
use greeners::{OLS, DataFrame, Formula, OlsResult};
```

For less common items, use the module path:

```rust
use greeners::autoreg::ARDL;
use greeners::gmm_clustering::GmmResult;
```

## What is re-exported

- All modules from `greeners-core`, `greeners-ols`, `greeners-glm`,
  `greeners-panel`, `greeners-timeseries`, `greeners-bayesian`,
  `greeners-causal`, `greeners-ml`, `greeners-spatial`, `greeners-survival`,
  `greeners-imputation` and `greeners-diagnostics`.
- A curated list of frequently used types and estimators at the crate root
  (e.g. `OLS`, `DataFrame`, `VAR`, `GLM`, `CoxPH`).

## Maintaining the facade

`scripts/check_facade.py` in the workspace root regenerates and verifies the
root re-exports from sub-crate public items. Run it after adding or renaming
estimators:

```bash
python3 scripts/check_facade.py
```

See the workspace `ARCHITECTURE.md` for dependency rules and the facade policy.
