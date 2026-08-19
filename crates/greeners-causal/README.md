# greeners-causal

Causal inference and treatment-effect estimators.

## Estimators

- **RD** — sharp and fuzzy regression discontinuity.
- **PSM** — propensity score matching.
- **DiffInDiff** — difference-in-differences.
- **SyntheticControl / SyntheticDiD** — synthetic control and DiD.
- **DoubleML / DML** — double/debiased machine learning.
- **DMLCrossfit** — cross-fitted DML.
- **CUPED** — controlled-experiment using pre-experiment data.
- **CausalImpact** — Bayesian structural time-series impact.
- **ConformalPrediction** — conformal inference.
- **CausalForest / DRLearner** — causal forest and DR-learner.
- **LpDid** — linear-programming DiD.
- **TMLE** — targeted maximum likelihood estimation.

## Usage

```toml
[dependencies]
greeners-causal = "2.0"
```

```rust
use greeners_causal::{RD, PSM};

let rd = RD::fit(&y, &x, &cutoff, Some(RdKernel::Triangular)).unwrap();
```

## Design notes

Common causal estimators are re-exported at the crate root. Result types stay
under their module when names collide.
