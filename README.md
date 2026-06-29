
# Greeners

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-1.4.9-blue.svg)](Cargo.toml)
[![Examples](https://img.shields.io/badge/examples-59-green.svg)](examples/)

> High-performance econometrics in Rust — statsmodels-grade coverage,
> compiled-language speed, and Rust's type safety guarantees.
> Named in honor of William H. Greene.

---

## Why Greeners?

Every serious econometrics tool makes a trade-off:

| Tool | Strength | Weakness |
|------|----------|----------|
| R (`plm`, `AER`, `sandwich`) | Deep econometrics, mature | Slow on large panels, no type safety |
| Python (`statsmodels`) | Broad coverage, readable | GIL, interpreter overhead, silent type errors |
| Julia (`FixedEffectModels`) | Fast | Small ecosystem, GC pauses |
| C++ | Maximum speed | No econometrics library exists |

**Greeners fills the gap**: the estimator coverage of statsmodels, the
performance of compiled code, and Rust's guarantee that type errors and
data races are caught at compile time — not at 2 AM in production.

No Python or R runtime. No system BLAS or LAPACK. Built on
[faer](https://github.com/sarah-ek/faer-rs) — pure-Rust linear algebra
with performance competitive with OpenBLAS, and a single `cargo add greeners`
to install.

Beyond parity, Greeners includes methods statsmodels does not: panel fixed and
random effects, Arellano-Bond GMM, IV/2SLS, Difference-in-Differences, and
automatic binary variable detection — all in a self-contained library with
built-in DataFrame, formula parser, and estimators.

---

## Quick start

```rust
use greeners::{OLS, DataFrame, Formula, CovarianceType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = DataFrame::from_csv("data.csv")?;
    let formula = Formula::parse("wage ~ education + experience + C(region)")?;
    let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;
    println!("{}", result);
    Ok(())
}
```

---

## Installation

```toml
[dependencies]
greeners = "1.4"
```

No system dependencies required.

---

## What's covered

### Linear models

| Estimator | Module | Key features |
|---|---|---|
| OLS | `ols` | HC1–HC4, NeweyWest, Clustered, ClusteredTwoWay, StudentT/Normal inference |
| WLS | `wls` | Weighted least squares |
| FGLS | `gls` | Feasible GLS, Cochrane-Orcutt AR(1) |
| GLSAR | `glsar` | GLS with AR errors |
| IV/2SLS | `iv` | Instrumental variables, all covariance types |
| Quantile | `quantile` | IRLS with bootstrap SE, any quantile |
| RLM | `rlm` | M-estimators: Huber, Tukey, Hampel, Andrews |

### Generalized linear models

| Estimator | Module | Key features |
|---|---|---|
| GLM | `glm` | Gaussian, Binomial, Poisson, Gamma, InverseGaussian, Tweedie, NegBin; 9 link functions |
| GLM-GAM | `glmgam` | B-spline basis with ridge penalty |
| Beta | `beta_model` | Beta regression for (0,1) outcomes |

### Discrete choice & count models

| Estimator | Module | Key features |
|---|---|---|
| Logit / Probit | `discrete` | MLE, AME/MEM marginal effects |
| MNLogit | `mnlogit` | Multinomial logit |
| OrderedLogit / OrderedProbit | `ordered` | Ordered outcomes |
| Poisson | `poisson` | Count data |
| NegBin / NegBinP / GenPoisson | `negbin` | Overdispersed counts |
| ZIP / ZINB | `zero_inflated` | Zero-inflated models |
| ConditionalLogit / MNLogit / Poisson | `conditional` | Conditional fixed-effects |

### Panel data

| Estimator | Module | Key features |
|---|---|---|
| Fixed Effects | `panel` | Within transformation, entity clustering |
| Random Effects | `panel` | Swamy-Arora GLS |
| Between | `panel` | Group-mean regression |
| Arellano-Bond | `dynamic_panel` | Difference GMM, Sargan test, AR tests |
| Panel Threshold | `threshold` | Hansen (1999), bootstrap CI |

### Time series

| Estimator | Module | Key features |
|---|---|---|
| ARIMA / SARIMAX | `arima` | Hannan-Rissanen, exogenous regressors, forecast intervals |
| AutoReg / ARDL | `autoreg` | Autoregressive distributed lag |
| ETS | `ets` | Additive/multiplicative error, trend, seasonality; damped trend |
| VAR | `var` | IRF (Cholesky), FEVD, AIC/BIC |
| VECM | `vecm` | Johansen cointegration, rank selection |
| VARMA | `varma` | State-space MLE |
| SVAR | `svar` | Cholesky, short-run, long-run, sign restrictions |
| Markov Switching | `markov`, `markov_autoreg` | Regime switching, autoregressive |
| GARCH / EGARCH / GJR-GARCH | `garch` | Normal + Student-t, BFGS MLE |

### State space & decomposition

| Estimator | Module | Key features |
|---|---|---|
| Kalman Filter / Smoother | `statespace` | Forward + RTS backward pass |
| Unobserved Components | `unobserved_components` | Local level, trends, seasonal, cycles |
| Dynamic Factor | `dynamic_factor` | DFM |
| MSTL | `mstl` | Multiple seasonal-trend decomposition |
| Classical decomposition | `decomposition` | Additive/multiplicative |

### System estimation

| Estimator | Module | Key features |
|---|---|---|
| SUR | `sur` | Seemingly Unrelated Regressions (Zellner FGLS) |
| 3SLS | `three_sls` | Three-Stage Least Squares |

### Mixed & multilevel models

| Estimator | Module | Key features |
|---|---|---|
| MixedLM | `mixed` | REML, random intercepts/slopes |
| BayesMixedGLM | `mixed` | Bayesian mixed GLM (MCMC) |
| GEE | `gee` | Independence, exchangeable, AR(1), unstructured |
| NominalGEE / OrdinalGEE | `gee` | Categorical outcomes |

### Causal & robust methods

| Estimator | Module | Key features |
|---|---|---|
| DiD | `did` | ATT, parallel trends test, event study |
| GMM | `gmm` | Optimal weighting, J-test |
| Bootstrap | `bootstrap` | Pairs, residual, block; hypothesis testing |

### Survival analysis

| Estimator | Module | Key features |
|---|---|---|
| Cox PH | `survival` | Partial likelihood, concordance index |
| Kaplan-Meier | `survival` | Survival curves |

### Multivariate & nonparametric

| Estimator | Module | Key features |
|---|---|---|
| PCA | `multivariate` | Principal components |
| Factor Analysis | `multivariate` | Varimax, Quartimax, Equamax rotations |
| MANOVA | `multivariate` | Multivariate ANOVA |
| Canonical Correlation | `multivariate` | CCA |
| KDE | `nonparametric` | Univariate + multivariate kernel density |
| Kernel Regression | `nonparametric` | Nadaraya-Watson |
| Lowess | `nonparametric` | Locally weighted regression |

### Rolling & recursive

| Estimator | Module | Key features |
|---|---|---|
| RollingOLS / RollingWLS | `rolling` | Moving-window estimation |
| RecursiveLS | `rolling` | Expanding-window least squares |

---

## Diagnostics & tests

### Regression diagnostics (`diagnostics`)

Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF, condition number, leverage, Cook's distance, omnibus, Harvey-Collier, Anderson-Darling.

### Specification tests (`specification_tests`)

White test, RESET, Breusch-Godfrey, Goldfeld-Quandt.

### Influence & stability (`influence`)

Influence measures, CUSUM test.

### Time series tests (`timeseries`)

ADF, KPSS, Phillips-Perron, Zivot-Andrews (structural break), Ljung-Box, ACF/PACF.

### Model selection (`model_selection`)

AIC/BIC comparison, Akaike weights, panel diagnostics (BP LM, F-test, Hausman).

### Statistical tests (`stats`)

One-way and two-way ANOVA, Tukey HSD, Bonferroni post-hoc, regression F-tests.

### Multiple testing (`multipletests`)

Bonferroni, Holm, Hochberg, Hommel, Benjamini-Hochberg, Benjamini-Yekutieli.

### Other (`proportion`, `descrstatsw`, `hausman`)

Proportion tests (one/two-sample, equivalence), weighted descriptive statistics, Hausman test.

---

## Inference

All linear models support two inference distributions via `.with_inference()`:

```rust
use greeners::InferenceType;

let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;
// Default: Student's t (exact, finite-sample)
let result_z = result.with_inference(InferenceType::Normal)?;
// Normal/z (asymptotic, statsmodels-compatible)
```

---

## Covariance types

```rust
CovarianceType::NonRobust               // Classical
CovarianceType::HC1                     // White (1980)
CovarianceType::HC2                     // Leverage-adjusted
CovarianceType::HC3                     // Jackknife (recommended for small samples)
CovarianceType::HC4                     // Cribari-Neto (2004)
CovarianceType::NeweyWest(lags)         // HAC
CovarianceType::Clustered(ids)          // One-way clustering
CovarianceType::ClusteredTwoWay(a, b)   // Two-way clustering (Cameron-Gelbach-Miller)
```

---

## Formula syntax

```rust
// Basic
"y ~ x1 + x2"                    // with intercept
"y ~ x1 + x2 - 1"                // no intercept

// Categoricals
"y ~ x1 + C(region)"             // auto dummy encoding (K-1)

// Transforms
"y ~ x + I(x^2)"                 // polynomial
"y ~ log(x1) + sqrt(x2)"         // functions

// Interactions
"y ~ x1 * x2"                    // full: x1 + x2 + x1:x2
"y ~ x1 : x2"                    // interaction only

// Splines
"y ~ bs(x, df=5)"                // B-splines
```

---

## DataFrame

```rust
// Load from CSV, JSON, URL, or builder
let df = DataFrame::from_csv("data.csv")?;
let df = DataFrame::from_csv_url("https://example.com/data.csv")?;
let df = DataFrame::from_json("data.json")?;
let df = DataFrame::builder()
    .add_column("y", vec![1.0, 2.0, 3.0])
    .add_column("x", vec![4.0, 5.0, 6.0])
    .build()?;

// Missing data
let clean   = df.dropna()?;
let filled  = df.fillna("x", 0.0)?;
let forward = df.fillna_ffill("x")?;
let interp  = df.interpolate("x")?;

// Time series ops
let lagged  = df.lag("price", 1)?;
let diffed  = df.diff("price", 1)?;
let returns = df.pct_change("price", 1)?;
```

Column types are auto-detected: Float, Int, Bool (including binary detection
from any two-value column), DateTime, Categorical, String.

---

## Imputation

```rust
use greeners::{MICE, BayesGaussMI};

let imputed = MICE::impute(&df, 10)?;         // Multiple imputation by chained equations
let imputed = BayesGaussMI::impute(&df, 10)?; // Bayesian Gaussian MI
```

---

## Examples

59 examples in `examples/`. Run any of them:

```sh
cargo run --example formula_example
cargo run --example marginal_effects
cargo run --example specification_tests
cargo run --example panel_model_selection
```

---

## About the name

William H. Greene is the author of *Econometric Analysis* — the reference text
used in graduate econometrics programs worldwide. Greeners is named in his honor:
the goal is to make the methods he systematized available to anyone writing
production systems in Rust.

---

## Roadmap

Active development targets full statsmodels parity. See [ROADMAP.md](ROADMAP.md)
for the complete feature matrix with implementation status.

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Before opening a PR:
- Add at least one example to `examples/` for new estimators
- Verify numerical output against statsmodels or R on a reference dataset
- Run `cargo test` and `cargo clippy -- -D warnings`

---

## License

[MIT](LICENSE)
