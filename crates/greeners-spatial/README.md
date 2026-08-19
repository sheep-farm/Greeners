# greeners-spatial

Spatial econometric models.

## Estimators

- **Spatial** — spatial autoregressive (SAR) and error (SEM) models.
- **SpatialDurbin** — spatial Durbin model.
- **SpatialDurbinError** — spatial Durbin error model.
- **SpatialPanel** — spatial panel models.

## Usage

```toml
[dependencies]
greeners-spatial = "2.0"
```

```rust
use greeners_spatial::{Spatial, SpatialDurbin};

let sar = Spatial::fit_sar(&y, &x, &w, None).unwrap();
```

## Design notes

All public estimators and results are re-exported at the crate root.
