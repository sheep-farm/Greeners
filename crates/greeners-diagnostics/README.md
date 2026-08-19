# greeners-diagnostics

Model diagnostics, specification tests and selection helpers.

## Tools

- **Diagnostics** — residual diagnostics (Durbin-Watson, Ljung-Box, ARCH,
  Breusch-Pagan, VIF, etc.).
- **SpecificationTests** — RESET, White, Breusch-Godfrey.
- **Influence / CUSUMTest** — influence and structural-break diagnostics.
- **BinaryDiagnostics** — binary-outfit diagnostics (ROC, Hosmer-Lemeshow,
  link test).
- **ModelSelection** — AIC/BIC, likelihood-ratio tests, panel diagnostics,
  summary statistics.
- **FamaMacBeth** — Fama-MacBeth regression.

## Usage

```toml
[dependencies]
greeners-diagnostics = "2.0"
```

```rust
use greeners_diagnostics::{Diagnostics, SpecificationTests};

let dw = Diagnostics::durbin_watson(&resids).unwrap();
```

## Design notes

Diagnostics structs and tests are re-exported at the crate root. Classification
and test result types are module-scoped when names overlap.
