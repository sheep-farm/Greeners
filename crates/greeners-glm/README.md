# greeners-glm

Generalized linear models, discrete choice and related extensions.

## Estimators / families

- **GLM** — generalized linear models with `Family` and `Link`.
- **Logit / Probit** — binary choice models.
- **MNLogit** — multinomial logit.
- **OrderedLogit / OrderedProbit** — ordinal models.
- **Poisson / NegBin / NegBinP / GenPoisson** — count models.
- **ZINB / ZIP** — zero-inflated count models.
- **BetaModel** — beta regression.
- **GLMGam** — generalized additive models.
- **GEE** — generalized estimating equations.
- **ConditionalLogit / ConditionalMNLogit / ConditionalPoisson** — conditional
  fixed-effects models.

## Usage

```toml
[dependencies]
greeners-glm = "2.0"
```

```rust
use greeners_glm::{GLM, Family, Link};

let result = GLM::fit_with_link(
    &y, &x,
    Family::Gaussian,
    Link::Identity,
    Some(vec!["x1".to_string()]),
).unwrap();
```

## Design notes

Common types such as `Family`, `Link`, `GLM` and `Poisson` are re-exported at
the crate root. Module-specific result types stay under `greeners_glm::<module>`.
