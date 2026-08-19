# greeners-bayesian

Bayesian and shrinkage econometric models.

## Estimators

- **BVAR** — Bayesian vector autoregression.
- **FAVAR** — factor-augmented vector autoregression.
- **MFVAR** — mixed-frequency VAR.
- **BayesianLinear** — Bayesian linear regression.
- **BayesianSC** — Bayesian synthetic control.
- **BayesianSFA** — Bayesian stochastic frontier analysis.
- **MixedLM / BayesMixedGLM** — Bayesian / mixed linear and generalized linear
  models.

## Usage

```toml
[dependencies]
greeners-bayesian = "2.0"
```

```rust
use greeners_bayesian::{BVAR, FAVAR};

let bvar = BVAR::fit(&data, 2, Some(0.5)).unwrap();
```

## Design notes

Common Bayesian estimators are re-exported at the crate root.
