# greeners-ml

Machine-learning estimators for econometric tasks.

## Estimators

- **RandomForest / GradientBoosting / XGBoost** — tree ensembles.
- **MLP / LSTM / Transformer** — neural networks.
- **KMeans / DBSCAN / HierarchicalClustering / SpectralClustering** — clustering.
- **TSNE / UMAP** — dimensionality reduction.
- **GaussianProcess** — Gaussian process regression.
- **BART** — Bayesian additive regression trees.
- **GRF / QRF / QrfInference / OrthogonalForest** — generalized, quantile and
  causal random forests.

## Usage

```toml
[dependencies]
greeners-ml = "2.0"
```

```rust
use greeners_ml::{RandomForest, KMeans};

let rf = RandomForest::fit(&y, &x, 100, 10, None).unwrap();
```

## Design notes

ML estimators are re-exported at the crate root. Result types are
module-scoped when names conflict.
