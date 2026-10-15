# greeners-ols

Classical and related regression estimators for the Greeners workspace.

## Estimators

- **OLS** — ordinary least squares.
- **WLS** — weighted least squares.
- **FGLS / GLSAR** — feasible and AR-corrected generalized least squares.
- **IV** — instrumental variables / 2SLS.
- **GMM** — generalized method of moments.
- **NLS** — non-linear least squares.
- **Heckman / Tobit** — sample selection and censored regression.
- **QuantileReg** — quantile regression.
- **RLM** — robust linear models.
- **SUR / ThreeSLS** — seemingly unrelated regressions and three-stage LS.
- **FMOLS** — Phillips-Hansen fully modified OLS for a cointegrating time-series regression.
- **RollingOLS / RollingWLS / RecursiveLS** — rolling and recursive estimation.
- **RegPath** — regularized regression paths.
- **EventStudy** — event-study estimation.

The corrected coefficient/SE calculation matches the declared native references.
That agreement does not establish uncertainty calibration. A prespecified
2,000-fit diagnostic found nominal 95% slope coverage of 90.0% (N=300) and
91.8% (N=1,000) under endogenous AR errors; both Monte Carlo intervals exclude
95%. Keep the estimator experimental and assess inference for the intended DGP.
The reproducible study is `examples/fmols_simulation.py`, using the actual Rust
`fmols_qualification` example. It is separate from routine unit tests.

## Usage

```toml
[dependencies]
greeners-ols = "2.0"
```

```rust
use greeners_ols::{OLS, OlsResult};
use ndarray::Array2;

let y = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
let result = OLS::fit(&y, &x, Some(vec!["x1".to_string(), "x2".to_string()])).unwrap();
```

## Design notes

Unique public items are re-exported at the crate root. Estimators with
duplicated names remain under `greeners_ols::<module>`.

## FMOLS contract (experimental)

`FMOLS::fit` estimates an intercept and one or more I(1) slopes, assuming
cointegration and suitable weak dependence. It does not test these assumptions.
Initial OLS uses all N observations; the fully modified regression uses rows
2..N, so `n_obs=N-1`. The Bartlett bandwidth is
`max(1,floor(4*(N/100)^(2/9)))`, without covariance centring or degrees-of-freedom
adjustment. The one-sided bias multiplier is N-1; uncertainty uses the
conditional long-run variance and two-sided Normal probabilities. `beta_t`
retains its legacy name but contains Normal test statistics. `r_squared` remains
the initial OLS quantity, rather than a goodness-of-fit measure of transformed y.

An exact unit intercept column may appear anywhere. Names must describe all
supplied columns or, with an explicit intercept, only the retained slopes.
Malformed names, other constants, non-finite input, deficient ranks or unresolved
positive uncertainty return errors. There are no hidden ridge terms. At least
one slope and N >= k+5 are required. Centring/scaling preserves predictor units
and shifts, including the original intercept uncertainty. Numerical rank and
residual budgets use 64 times machine epsilon times the problem dimension;
these are precision rejection rules, not statistical significance rules.

The native-reference fixtures in `tests/fixtures/fmols` compare coefficients,
standard errors and long-run covariance against arch 8.0.0 and document the
cointReg 0.2.0 N-versus-N-1 bias convention separately. Normal tails are checked
independently. Agreement on these inputs does not establish general interval
coverage, robustness or causal identification.
