# greeners-timeseries

Time-series models, filters and forecasts.

## Families

- **ARIMA / SARIMAX** — auto-regressive integrated moving averages.
- **VAR / VARMA / VECM / SVAR** — vector autoregression and cointegration.
- **GARCH / EGARCH / GJRGARCH / DCCGARCH** — volatility models.
- **AutoReg / ARDL / NARDL / SETAR / TVAR / TVP** — univariate and threshold
  models.
- **MarkovSwitching / MSVAR / MarkovAutoregression** — regime-switching models.
- **ETS / MSTL / ExponentialSmoothing** — exponential smoothing and
  decomposition.
- **StateSpaceModel / KalmanFilter / KalmanSmoother** — state-space and
  unobserved components.
- **DFM / DynamicFactor** — dynamic factor models.
- **LSTM / Transformer** — neural and sequence models.
- **Spectral / MODWT / Wavelet** — spectral and wavelet analysis.
- **Hawkes / TvCopula / Midas** — point processes, time-varying copulas and
  mixed-frequency data.

## Usage

```toml
[dependencies]
greeners-timeseries = "2.0"
```

```rust
use greeners_timeseries::{VAR, ARIMA};

let var = VAR::fit(&data, 2, None).unwrap();
```

## Design notes

This is the largest crate in the workspace. A future refactor may split it
into smaller thematic crates (`greeners-arima`, `greeners-garch`, etc.).
