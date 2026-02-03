
# Greeners

**High-performance econometrics in Rust.**

![Version](https://img.shields.io/badge/version-1.4.0-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

A self-contained econometrics library with R/Python-style formulas, robust inference, and comprehensive coverage of cross-sectional, time series, panel data, and system estimation methods. No external statistical dependencies — DataFrame, formula parser, and all estimators are built in.

## Installation

### System dependencies

```sh
# Debian/Ubuntu
sudo apt-get install gfortran libopenblas-dev liblapack-dev pkg-config

# Fedora/RHEL
sudo dnf install gcc-gfortran openblas-devel lapack-devel pkg-config

# Arch
sudo pacman -S gcc-fortran openblas lapack base-devel

# macOS
brew install openblas lapack
```

### Cargo.toml

```toml
[dependencies]
greeners = "1.4.0"
ndarray = "0.17"
ndarray-linalg = { version = "0.18", features = ["openblas-system"] }
```

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

## Inference

All linear models support two inference distributions via `.with_inference()`:

```rust
use greeners::InferenceType;

let result = OLS::from_formula(&formula, &df, CovarianceType::HC3)?;
// Default: Student's t (exact, finite-sample)
let result_z = result.with_inference(InferenceType::Normal)?;
// Normal/z (asymptotic, statsmodels-compatible)
```

## Covariance types

```rust
CovarianceType::NonRobust           // Classical
CovarianceType::HC1                 // White (1980)
CovarianceType::HC2                 // Leverage-adjusted
CovarianceType::HC3                 // Jackknife (recommended for small samples)
CovarianceType::HC4                 // Cribari-Neto (2004)
CovarianceType::NeweyWest(lags)     // HAC
CovarianceType::Clustered(ids)      // One-way clustering
CovarianceType::ClusteredTwoWay(a, b) // Two-way clustering (Cameron-Gelbach-Miller)
```

## Formula syntax

```rust
// Basic
"y ~ x1 + x2"                    // with intercept
"y ~ x1 + x2 - 1"                // no intercept

// Categoricals
"y ~ x1 + C(region)"             // auto dummy encoding (K-1)

// Transforms
"y ~ x + I(x^2)"                 // polynomial
"y ~ log(x1) + sqrt(x2)"        // functions

// Interactions
"y ~ x1 * x2"                    // full: x1 + x2 + x1:x2
"y ~ x1 : x2"                    // interaction only

// Splines
"y ~ bs(x, df=5)"                // B-splines
```

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
let clean = df.dropna()?;
let filled = df.fillna("x", 0.0)?;
let forward = df.fillna_ffill("x")?;
let interp = df.interpolate("x")?;

// Time series ops
let lagged = df.lag("price", 1)?;
let diffed = df.diff("price", 1)?;
let returns = df.pct_change("price", 1)?;
```

Column types are auto-detected: Float, Int, Bool (including binary detection from any two-value column), DateTime, Categorical, String.

## Imputation

```rust
use greeners::{MICE, BayesGaussMI};

let imputed = MICE::impute(&df, 10)?;           // Multiple imputation by chained equations
let imputed = BayesGaussMI::impute(&df, 10)?;   // Bayesian Gaussian MI
```

## Examples

59 examples in `examples/`. Run any of them:

```sh
cargo run --example formula_example
cargo run --example marginal_effects
cargo run --example specification_tests
cargo run --example panel_model_selection
```

## License

GPL-3.0-or-later

