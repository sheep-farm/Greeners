# greeners-panel

Panel-data and longitudinal estimators.

## Estimators

- **FixedEffects / RandomEffects** — fixed and random effects models.
- **BetweenEstimator / PCSE / PanelGLS** — between, panel-corrected and GLS
  estimators.
- **FE2SLS / PanelIvResult** — panel instrumental variables.
- **ArellanoBond / SystemGmm** — dynamic panel GMM.
- **PanelVAR** — panel vector autoregression.
- **PSTR** — panel smooth transition regression.
- **PanelThreshold** — threshold regression for panels.
- **PanelQuantile** — panel quantile regression.
- **PanelHeckman / PanelTobit** — panel sample selection and censored models.
- **FAPanel** — factor-augmented panel.
- **RobustHausman / RobustFTest** — robust specification tests.

## Usage

```toml
[dependencies]
greeners-panel = "2.0"
```

```rust
use greeners_panel::{FixedEffects, BetweenEstimator};

let fe = FixedEffects::from_formula(...).unwrap();
```

## Design notes

Common panel types are re-exported at the crate root. Result types with
colliding names remain namespaced.
