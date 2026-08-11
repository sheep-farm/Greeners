# Mathematical proofs for the Greeners numerical engine

This document tries to do for Greeners what Knuth calls a proof, not a test:
for every implemented estimator we state the mathematical specification,
derive the closed-form result, and map each step to the source code.
The corresponding tests in `tests/` check the algebraic invariants that
follow from those derivations.

---

## OLS — Ordinary Least Squares

### Specification

Linear model with `n` observations and `k` regressors:

```
y = X β + ε
```

where `y` is `n×1`, `X` is `n×k` with full column rank `k`, `β` is `k×1`.

### Objective

Minimize the residual sum of squares:

```
S(β) = (y - Xβ)'(y - Xβ)
```

### Derivation

Expand and take the gradient with respect to `β`:

```
S(β) = y'y - 2 β' X' y + β' X' X β
∇S(β) = -2 X' y + 2 X' X β
```

The first-order condition `∇S(β) = 0` gives the **normal equations**:

```
X' X β = X' y
```

If `X'X` is invertible (equivalently, `X` has full column rank), the unique
least-squares estimator is:

```
β̂ = (X'X)^-1 X'y
```

### Residuals and fitted values

```
ê    = y - X β̂
ŷ    = X β̂ = P y
P    = X (X'X)^-1 X'
```

`P` is the orthogonal projection onto the column space of `X`. It is
idempotent (`P P = P`) and symmetric (`P' = P`). The normal equations imply
`X' ê = 0`; if `X` contains a constant column this also implies `Σ_i ê_i = 0`.

### Variance (non-robust)

With homoskedastic errors, the OLS residual variance is:

```
σ̂² = ê'ê / (n - k)
```

and the covariance matrix of `β̂` is:

```
Var̂(β̂) = σ̂² (X'X)^-1
```

Standard errors are `sqrt(diag(Var̂(β̂)))`.

### Mapping to `src/ols.rs`

| Math | Code (`src/ols.rs`, inside `OLS::fit_internal`) |
|---|---|
| `X'X` | `let xt_x = x_to_use.t().dot(x_to_use);` |
| `(X'X)^-1` | `let xt_x_inv = xt_x.inv()?;` |
| `X'y` | `let xt_y = x_t.dot(y);` |
| `β̂` | `let beta = xt_x_inv.dot(&xt_y);` |
| `ŷ` | `let predicted = x_to_use.dot(&beta);` |
| `ê` | `let residuals = y - &predicted;` |
| `ê'ê` | `let ssr = residuals.dot(&residuals);` |
| `σ̂²` | `let sigma2 = ssr / (df_resid as f64);` |
| `Var̂(β̂)` | `&xt_x_inv * sigma2` (NonRobust branch) |

Collinearity detection (`src/ols.rs:detect_collinearity` and
`src/linalg::drop_collinear`) ensures `X'X` is invertible by removing linearly
dependent columns before the main fit.

### Robust and clustered covariance

All robust forms are sandwich estimators of the generic form:

```
V = (X'X)^-1 M (X'X)^-1
```

where `M` is a "meat" matrix built from residuals and regressors.

| Type | Meat `M` |
|---|---|
| HC1 | `X' diag(ê_i²) X × n/(n-k)` |
| HC2 | `X' diag(ê_i² / (1 - h_i)) X` |
| HC3 | `X' diag(ê_i² / (1 - h_i)²) X` |
| HC4 | `X' diag(ê_i² / (1 - h_i)^{δ_i}) X`, `δ_i = min(4, n h_i / k)` |
| Newey-West | `M_0 + Σ_l w_l (M_l + M_l')` |
| Clustered | `Σ_g (X_g' ê_g ê_g' X_g)` with small-sample correction |
| Two-way cluster | `V_1 + V_2 - V_{1∩2}` |

### Invariants verified in `tests/ols_invariants.rs`

1. **Normal equations**: `X' ê = 0`
2. **Residuals sum to zero** when `X` contains an intercept
3. **Decomposition**: `y = ŷ + ê` and `ŷ' ê = 0`
4. **Projection idempotency**: `P P = P`
5. **Projection reproduces fitted values**: `P y = ŷ`
6. **Linearity in `y`**: scaling `y` by `c` scales `β̂` by `c`
7. **Translation invariance** of a regressor: shifting `x` by `a` changes the
   intercept by `-a·slope` and leaves the slope unchanged
8. **Exact arithmetic** for a small integer design where `β̂ = [2, 3]`

These tests are stronger than empirical comparison with R or Python because
they check the algebra of the derived estimator directly.

---

## IV / 2SLS — Instrumental Variables

### Specification

Linear model with endogenous regressors:

```
y = X β + u
```

where `X` is `n×k` and may be correlated with the error `u`. We have an
instrument matrix `Z` of dimension `n×l` (`l ≥ k`) that is correlated with
`X` but uncorrelated with `u`:

```
E[Z'u] = 0
```

### Derivation

The **two-stage least squares** (2SLS) estimator can be derived in two
equivalent ways.

#### Two-stage construction

**First stage:** project each endogenous regressor onto the space spanned by
the instruments:

```
Π = (Z'Z)^-1 Z'X
X̂ = Z Π = Z (Z'Z)^-1 Z' X = P_Z X
```

`P_Z = Z (Z'Z)^-1 Z'` is the orthogonal projection matrix onto the column
space of `Z`.

**Second stage:** run OLS of `y` on the fitted regressors `X̂`:

```
β̂_2SLS = (X̂'X̂)^-1 X̂'y
```

#### One-stage formulation (more efficient numerically)

Substituting `X̂ = P_Z X`:

```
β̂_2SLS = [X' P_Z' P_Z X]^-1 X' P_Z' y
       = [X' P_Z X]^-1 X' P_Z y
```

because `P_Z` is idempotent and symmetric (`P_Z' = P_Z`, `P_Z P_Z = P_Z`).
This is the standard one-stage 2SLS formula.

### Invariants

- **First-stage orthogonality**: the first-stage residuals `V = X - X̂`
  satisfy `Z' V = 0` (each column of `V` is orthogonal to every instrument).
- **Second-stage normal equations**: `X̂' (y - X β̂) = 0`.
- **Order condition**: for identification we require `l ≥ k`. With `l = k` the
  model is exactly identified; with `l > k` it is overidentified and the
  overidentifying restrictions can be tested with the Sargan/Hansen J test.

### Mapping to `src/iv.rs`

| Math | Code (`src/iv.rs`, inside `IV::fit_with_names`) |
|---|---|
| `Z'Z` | `let zt_z = z.t().dot(z);` |
| `(Z'Z)^-1` | `let zt_z_inv = zt_z.inv()?;` |
| `Z'X` | `let zt_x = z_t.dot(x_to_use);` |
| `Π` | `let first_stage_coeffs = zt_z_inv.dot(&zt_x);` |
| `X̂` | `let x_hat = z.dot(&first_stage_coeffs);` |
| `X̂'X̂` | `let xht_xh = x_hat.t().dot(&x_hat);` |
| `β̂` | `let beta = xht_xh_inv.dot(&xht_y);` |
| `ê` | `let residuals = y - &predicted_original;` |

The residuals are computed with the **original** `X`, not `X̂`; this is the
right IV residual for inference. The second-stage regressor, however, is
`X̂`, so the normal equations `X̂' ê = 0` hold.

### Robust covariance

The logic is the same as OLS, with `X` replaced by `X̂` in the sandwich
"bread" and in the construction of the "meat". HC1–HC4, Newey-West and
clustered forms are documented in `src/iv.rs`.

### Sargan / Hansen J

For overidentified models, the Sargan test checks whether the extra
instruments are valid. It is implemented as:

```
Sargan = n × R²
```

from the regression of `IV residuals ê` on `Z`, distributed as
`χ²(l - k)` under the null.

### Invariants verified in `tests/iv_invariants.rs`

1. `X̂' ê = 0` (second-stage normal equations)
2. `Z' (X - X̂) = 0` (first-stage residuals orthogonal to instruments)
3. `β̂ = (X̂'X̂)^-1 X̂'y` matches `IV::fit` output
4. Scaling `y` scales `β̂` by the same factor
5. Exact arithmetic for a hand-crafted DGP: `y = 1 + 5x`, `x = 2 + 3z`,
   `β̂ = [1, 5]`
6. Order condition `l < k` is rejected

---

## Fixed Effects (Within Estimator)

### Specification

Panel model with entity-specific intercepts:

```
y_it = x_it' β + α_i + u_it
```

where `i = 1..N` indexes entities, `t = 1..T_i` indexes time, and `α_i` is a
fixed effect. The `within` or `fixed effects` estimator removes `α_i` by
demeaning each variable within its entity:

```
ỹ_it = y_it - (1/T_i) Σ_t y_it
x̃_it = x_it - (1/T_i) Σ_t x_it
```

### Derivation

Averaging the model over time for entity `i`:

```
ȳ_i = x̄_i' β + α_i + ū_i
```

Subtracting from the original equation:

```
ỹ_it = x̃_it' β + ũ_it
```

The entity fixed effect `α_i` cancels. Running OLS on the demeaned data
gives the within estimator `β̂_W`:

```
β̂_W = (Σ_i Σ_t x̃_it x̃_it')^-1 Σ_i Σ_t x̃_it ỹ_it
```

Because the within transformation removes all time-invariant variation, any
regressor that is constant within an entity becomes identically zero after
demeaning and cannot be identified.

### Degrees of freedom

Demeaning uses one degree of freedom per entity, so the residual degrees of
freedom are:

```
df = n - k - (N - 1)
```

where `n = Σ_i T_i` is the total number of observations, `k` is the number
of (time-varying) regressors, and `N` is the number of entities.

### Mapping to `src/panel.rs`

| Math | Code (`src/panel.rs`, inside `FixedEffects::fit_with_names`) |
|---|---|
| `y_it` | `let y_mat = y.view().insert_axis(Axis(1)).to_owned();` |
| `ỹ_it` | `let y_demeaned_mat = Self::within_transform(&y_mat, groups)?;` |
| `x̃_it` | `let x_demeaned = Self::within_transform(x, groups)?;` |
| `β̂_W` | `let ols_result = OLS::fit(&y_demeaned, &x_demeaned, cov_type.clone())?;` |
| `df` | `let df_resid_correct = n - k - (n_entities - 1);` |
| `σ̂²` | `let sigma2 = ssr / (df_resid_correct as f64);` |

`FixedEffects::within_transform` computes group sums and subtracts the group
mean from each row, which is precisely the within transformation.

### Invariants verified in `tests/panel_invariants.rs`

1. Group means of within-transformed `y` and `X` are zero for every entity.
2. `β̂_W` equals OLS on the manually demeaned data.
3. Within residuals `(ỹ - X̃ β̂_W)` have zero mean within each entity.
4. Exact arithmetic for a hand-crafted panel: `y_it = a_i + 2 x_it`
   gives `β̂_W = 2`, `n_entities = 2`, `df = 2`.
5. Time-invariant regressors are dropped / cause failure because their
   within transform is identically zero.

---

## SUR / 3SLS — Seemingly Unrelated Regressions

### SUR

Consider `M` linear equations observed over the same `n` units:

```
y_m = X_m β_m + u_m,   m = 1..M
```

Stack all equations:

```
Y = X β + U
```

where `Y = (y_1',..., y_M')'`, `X = block-diag(X_1,...,X_M)`,
`β = (β_1',...,β_M')'`. Assume

```
E[U U'] = Σ ⊗ I_n
```

where `Σ` is `M×M` with `Σ_{ij} = E[u_{mi} u_{nj}]`. With known `Σ`, the
efficient estimator is GLS:

```
β̂_SUR = [X' (Σ^{-1} ⊗ I_n) X]^{-1} X' (Σ^{-1} ⊗ I_n) Y
```

In practice `Σ` is unknown, so we use Feasible GLS:

1. Estimate each equation by OLS, collect residuals `û_m`.
2. Estimate `Σ_{ij} = û_i' û_j / n`.
3. Plug into the GLS formula.

If the errors are uncorrelated across equations (`Σ` diagonal), the GLS
system block-diagonalizes and `β̂_SUR` reduces to OLS on each equation.

### Mapping to `src/sur.rs`

The code builds the block matrices directly, avoiding the Kronecker product:

- `lhs[[start_i.., start_j..]] = s^{ij} * (X_i' X_j)`
- `rhs[start_i..] += s^{ij} * (X_i' y_j)`

where `s^{ij}` is the `(i,j)` element of `Σ^{-1}`. Solving `lhs β = rhs`
gives `β_SUR`.

### 3SLS

Three-Stage Least Squares generalizes SUR to simultaneous equations with
endogenous regressors. Each structural equation is

```
y_m = X_m β_m + u_m
```

where `X_m` may contain endogenous variables. Let `Z` be the full matrix of
exogenous instruments (with a constant added if absent). The algorithm is:

1. **Stage 1:** project every endogenous regressor onto `Z`:

   ```
   X̂_m = P_Z X_m,   P_Z = Z (Z'Z)^{-1} Z'
   ```

2. **Stage 2:** run 2SLS equation-by-equation and collect residuals
   `û_m = y_m - X_m β̂_{2sls,m}`.

3. **Stage 3:** estimate `Σ_{ij} = û_i' û_j / n` and run feasible GLS on
   the stacked system with `X̂_m` in place of `X_m`:

   ```
   β̂_3SLS = [X̂' (Σ^{-1} ⊗ I_n) X̂]^{-1} X̂' (Σ^{-1} ⊗ I_n) Y
   ```

When `M = 1`, 3SLS reduces to 2SLS.

### Mapping to `src/three_sls.rs`

| Math | Code (`src/three_sls.rs`) |
|---|---|
| `P_Z = Z (Z'Z)^{-1} Z'` | `projection_matrix_part = z_instruments.dot(&ztz_inv).dot(&z_t);` |
| `X̂_m` | `let x_hat = projection_matrix_part.dot(&eq.x);` |
| `β̂_{2sls,m}` | `let beta_2sls = xt_x_inv.dot(&xt_y);` using `X̂_m' X_m` |
| `Σ` | `residuals_2sls.t().dot(&residuals_2sls) / n` |
| `Σ^{-1}` | `sigma.inv()` |
| `β̂_3SLS` | `lhs_inv.dot(&rhs_system)` built from `s_ij * X̂_i' X̂_j` and `s_ij * X̂_i' y_j` |

### Invariants verified in `tests/sur_3sls_invariants.rs`

1. SUR with uncorrelated equation errors gives the same coefficients as
   OLS on each equation.
2. The estimated cross-equation covariance is zero for the uncorrelated
   design.
3. 3SLS with one equation gives the same coefficients as IV/2SLS.

---

## GLM — Generalized Linear Models

### Specification

For exponential family with mean `μ_i` and link `g(·)`:

```
E[y_i | x_i] = μ_i
η_i = g(μ_i) = x_i' β
```

### Score and Fisher information

For a single observation the log-likelihood (up to constants) is:

```
ℓ_i(β) = [y_i θ_i - b(θ_i)] / φ
```

with `θ_i` the canonical parameter. The score is:

```
∂ℓ / ∂β = Σ (y_i - μ_i) / [V(μ_i) g'(μ_i)] x_i
```

For the **canonical link** we have `g'(μ_i) = 1 / V(μ_i)`, so the score
simplifies to the key invariant:

```
X' (y - μ) = 0
```

The Fisher information is:

```
I(β) = X' W X,   W_ii = 1 / [V(μ_i) g'(μ_i)²]
```

### IRLS algorithm

Greeners solves the score equations with Iteratively Reweighted Least
Squares. At each iteration:

1. `η = Xβ (+ offset)`
2. `μ = g⁻¹(η)`
3. `w_i = 1 / [V(μ_i) g'(μ_i)²]`
4. `z_i = η_i + (y_i - μ_i) g'(μ_i)`
5. `β_new = (X'WX)^{-1} X'Wz`

For a canonical link, step 5 is algebraically equivalent to solving
`X'(y - μ) = 0`.

### Mapping to `src/glm.rs`

| Math | Code (`src/glm.rs`, inside `GLM::fit_internal`) |
|---|---|
| `μ` / `η` | `mu = eta.mapv(\|e\| link.linkinv(e));` |
| `V(μ)` | `let v = family.variance(mu[i]);` |
| `g'(μ)` | `let g_prime = link.deriv(mu[i]);` |
| IRLS weights `W` | `w_vec` built from `1.0 / (v * g_prime * g_prime)` |
| Working variable `z` | `base = eta[i] + (y[i] - mu[i]) * g_prime` |
| WLS update | `inv_xtwx.dot(&xtwz)` |
| Covariance `(X'WX)^{-1} φ` | `inv_xtwx` scaled by `dispersion` |

### Invariants verified in `tests/glm_invariants.rs`

1. Gaussian + identity link reproduces OLS exactly.
2. Poisson + log link recovers the true DGP `y = exp(β0 + β1 x)`.
3. Logit (canonical link) satisfies the score equation `X'(y - μ) = 0` at
   convergence and recovers the true parameters.

---

## VAR / VARMA

### VAR

A vector autoregression of order `p` for a `k`-dimensional process `y_t`:

```
y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t
```

Stack all observations `t = p+1..T`:

```
Y = X B + U
```

where `Y` is `(T-p) × k`, `X` is `(T-p) × (1 + kp)` with columns
`[1, y_{t-1}', ..., y_{t-p}']`, and `B` is `(1+kp) × k`.

Because the model is linear in the parameters, OLS equation-by-equation is
minimum-distance-equivalent to GLS when the same regressors appear in every
equation. The closed-form estimator is:

```
B̂ = (X'X)^{-1} X'Y
```

with residual covariance:

```
Σ_u = U'U / (T - p - kp - 1)
```

### Mapping to `src/var.rs`

| Math | Code (`src/var.rs`) |
|---|---|
| `X = [1, y_{t-1}, ..., y_{t-p}]` | `x_mat` built with intercept column and lag rows |
| `B = (X'X)^{-1} X'Y` | `params = xt_x_inv.dot(&xt_y);` |
| `U = Y - X B` | `residuals = &y_eff - &preds;` |
| `Σ_u` | `residuals.t().dot(&residuals) / (n_obs - n_cols_x)` |

### VARMA

The VARMA(p,q) adds a moving-average term:

```
y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t + M_1 u_{t-1} + ... + M_q u_{t-q}
```

`src/varma.rs` implements the Hannan-Rissanen two-step procedure:

1. **Long VAR** of order `p_long > max(p,q)`: fit a high-order VAR to
   obtain residuals `û_t`.
2. **Second-stage regression:** regress `y_t` on its own lags and on
   `û_{t-1}, ..., û_{t-q}` to estimate `A_i` and `M_j`.

For `q = 0` there are no MA terms, so the second stage is a VAR(p) fitted
on a slightly shorter sample (after the long VAR burn-in). On deterministic
VAR(1) data this recovers the VAR(1) coefficients to machine precision.

### Invariants verified in `tests/var_invariants.rs` and `tests/varma_invariants.rs`

1. VAR(1) on deterministic data recovers the true `c` and `A`.
2. VAR residuals are orthogonal to the regressor matrix (`X'U = 0`).
3. `VAR::fit` rejects `lags > observations`.
4. VARMA(1,0) recovers the same AR parameters as VAR(1) on deterministic
   data.
5. VARMA(1,1) returns parameter matrices of expected shape and a
   non-negative residual covariance.

---

## Panel GMM — Arellano-Bond (Diff-GMM)

### Specification

Dynamic panel with an AR(1) lag and fixed effects:

```
y_{it} = ρ y_{i,t-1} + x_{it}' β + α_i + u_{it}
```

The fixed effect is removed by first-differencing:

```
Δy_{it} = ρ Δy_{i,t-1} + Δx_{it}' β + Δu_{it}
```

`Δy_{i,t-1}` is endogenous because `y_{i,t-1}` is correlated with
`α_i`. Arellano-Bond instruments the differenced equation with deeper
lags in levels:

```
E[y_{i,t-s} Δu_{it}] = 0,   s ≥ 2
```

### GMM moment conditions

For an equation with `r` endogenous/FD regressors and `l` instruments:

```
W = [ΔY_lag | ΔX_active]        (n_eff × k)
Z = [y_{t-2},...,y_{t-max_lags-1} | ΔX_active]   (n_eff × l)
```

The efficient one-step GMM estimator is:

```
β̂ = (W'Z A_1 Z'W)^{-1} W'Z A_1 Z'Δy
```

where `A_1 = (Z' H Z)^{-1}` and `H` is the block-diagonal covariance
matrix of first-differenced errors (`H_i` tridiagonal with `2, -1, -1`).

In the just-identified case (`l = k`), the weight matrix `A_1` cancels
and the estimator reduces to instrumental variables:

```
β̂_IV = (Z'W)^{-1} Z'Δy
```

### Mapping to `src/dynamic_panel.rs`

| Math | Code (`src/dynamic_panel.rs`, inside `ArellanoBond::fit`) |
|---|---|
| First differences `Δy, ΔY_lag, ΔX` | `dy_vec`, `dyl_vec`, `dx_rows` built from ordered panel data |
| Instrument matrix `Z` | `z_mat` with lagged levels + FD exogenous regressors |
| Weight `A_1 = (Z'HZ)^{-1}` | `a1 = zthz.inv()` built from `H_i` blocks |
| GMM formula | `params1 = lhs1_inv.dot(&wtz_a1.dot(&zty));` |
| Sargan/Hansen test | `s = zu1.dot(&a1.dot(&zu1)) * (n_eff / ssr)` |
| m1 / m2 serial-correlation tests | `compute_m_stats` on FD residuals |

### Invariants verified in `tests/panel_gmm_invariants.rs`

1. Arellano-Bond with `max_lags = 1` and no exogenous regressors equals
   the hand-computed IV estimate `β_IV = (Σ z Δy) / (Σ z Δy_{lag})`.
2. Over-identified specification (`max_lags = 2`) reports a Sargan test
   with `df = l - k = 1` and a finite p-value.

---

## Quantile Regression

### Specification

For a quantile `τ ∈ (0,1)` the linear model is:

```
Q_τ(y_i | x_i) = x_i' β_τ
```

The objective is the check (pinball) loss:

```
ρ_τ(u) = u (τ - I(u < 0))
```

The subgradient condition at an optimum is:

```
Σ (τ - I(u_i < 0)) x_i = 0,   u_i = y_i - x_i' β_τ
```

### Mapping to `src/quantile.rs`

Greeners solves the problem with iteratively reweighted least squares. At
each step the weight is:

```
w_i = (τ if u_i ≥ 0 else 1-τ) / max(ε, |u_i|)
```

so that `w_i u_i` approximates `τ - I(u_i < 0)`. The WLS update solves:

```
(X' W X) β_new = X' W y
```

which is the smoothed first-order condition.

### Invariants verified in `tests/quantile_invariants.rs`

1. Perfectly collinear data `y = β_0 + x'β` is recovered for every `τ`.
2. Scaling `y` by `c` scales all coefficients by `c`.
3. Translating `y` by `c` shifts only the intercept by `c`.
4. `τ` outside `(0,1)` is rejected.

---

## Regularization Path — Ridge / Lasso / ElasticNet

### Ridge closed form

For standardized `X` and `y`:

```
β̂_Ridge(λ) = (X'X + λ I)^{-1} X'y
```

`src/reg_path.rs` standardizes the regressors and response, computes the
ridge path, then un-standardizes:

```
β_j = β_j^* · (σ_y / σ_j)
β_0 = ȳ - Σ_j β_j x̄_j
```

### Lasso coordinate descent

The Lasso (or ElasticNet) sub-problem is solved by cyclic coordinate descent.
For each `j`:

```
rho = Σ_i x_{ij} (y_i - Σ_{k≠j} x_{ik} β_k)
β_j = sign(rho) · max(|rho| - λ·α, 0) / (Σ_i x_{ij}^2 + λ·(1-α))
```

### Mapping to `src/reg_path.rs`

| Math | Code (`src/reg_path.rs`) |
|---|---|
| Standardization | `x_mean`, `x_std`, `y_mean`, `y_std` and `x_norm` / `y_norm` |
| `λ_max` | `xty.iter().map(\|v\| v.abs()).fold(0.0, f64::max) / (n * a)` |
| Ridge formula | `Self::ridge_fit` builds `X'X + λ I` and inverts |
| Lasso update | `Self::lasso_fit` coordinate descent with soft-thresholding |
| BIC | `bic = n ln(σ²) + n_nonzero ln(n)` |

### Invariants verified in `tests/reg_path_invariants.rs`

1. The full ridge path matches the closed-form formula on standardized data,
   including the un-standardization of coefficients and intercept.
2. Lasso with a strong penalty at `λ_max` shrinks the slope coefficients to
   zero.
3. ElasticNet with `α = 0` (pure ridge) returns finite, regularized
   coefficients.

---

## Panel Tobit (Random Effects)

### Specification

The left-censored random-effects Tobit is:

```
y_it = max(c, x_it'β + α_i + ε_it)
α_i ~ N(0, σ_α²),   ε_it ~ N(0, σ_ε²)
```

Greeners uses a practical EM-style routine in `src/panel_tobit.rs`:

1. Start from OLS on the uncensored observations.
2. E-step: replace censored `y_it` by its conditional expectation under the
   current `β` and `σ`:

```
E[y_it* | y_it* ≤ c] = xb - σ φ((c - xb)/σ) / Φ((c - xb)/σ)
```

3. M-step: OLS on the completed `y*`.
4. Estimate between-panel and within-panel variance components.
5. Report likelihood, standard errors and ICC `ρ = σ_α² / (σ_α² + σ_ε²)`.

### Mapping to `src/panel_tobit.rs`

| Step | Code |
|---|---|
| Identify censored obs | `censored = y_i ≤ censor_left` |
| E-step | `trunc_mean = xb - sigma * phi_z / cdf_z` |
| M-step | `beta = (X'X)^{-1} X'y_star` |
| Variance components | `panel_resid_sums` → `between_var`, `within_var` |
| Standard errors | `cov_beta = (X'X)^{-1} σ²` |
| Log-likelihood | `ll = Σ uncensored ln φ((y-xb)/σ) + Σ censored ln Φ((c-xb)/σ)` |

### Invariants verified in `tests/panel_tobit_invariants.rs`

1. When no observation is censored (`censor_left` far below the data), the
   EM iterations reduce to OLS and `beta` equals the OLS coefficients.
2. When all observations are censored, estimation is rejected for lack of
   uncensored observations.

---

## Bayesian VAR (BVAR) — Minnesota Prior

### Specification

For the VAR equation of variable `j`:

```
y_{t,j} = Σ_{l=1}^p Σ_{m=1}^k B_{j,lm} y_{t-l,m} + u_{t,j}
```

with Normal likelihood `u_t ~ N(0, Σ)`. Greeners assumes a conjugate
Normal prior:

```
β_j ~ N(b_0, V_0),    b_{0,own-lag-1} = μ, others 0
```

with a diagonal Minnesota prior variance:

```
V_0[own lag]      = λ_1² · lag_decay · σ_j²
V_0[cross lag]    = λ_2² · lag_decay · (σ_m² / σ_j²) · σ_j²
lag_decay(l)      = (l+1)^{-λ_3}
```

The posterior is also Normal:

```
post_cov_j   = (X'X + V_0^{-1})^{-1}
post_mean_j  = post_cov_j · (X'y_j + V_0^{-1} b_0)
```

### Mapping to `src/bvar.rs`

| Math | Code |
|---|---|
| Lag design matrix | `x[(i, lag*k + j)] = y[(lags + i - 1 - lag, j)]` |
| OLS start | `ols_beta = (X'X)^{-1} X'Y` |
| Prior mean `b_0` | `b0[eq] = mu` for own first lag, else 0 |
| Prior covariance `V_0` | diagonal entries from `l1`, `l2`, `l3` and `sigma2_ols` |
| Posterior | `post_prec = X'X + V0_inv; post_mean = post_cov * (X'y + V0_inv * b0)` |
| Marginal likelihood | simplified Laplace approximation |

### Invariants verified in `tests/bvar_invariants.rs`

1. With a diffuse prior (`λ_1, λ_2` large) and moderate noise, the posterior
   mean converges to the OLS estimate on the lag design.
2. With a very tight prior (`λ_1, λ_2 → 0`) centered on a random walk, the
   own first-lag coefficients shrink to 1 and the cross-lags shrink to 0.
3. `lags = 0` and too few observations are rejected.

## Long-term follow-up

The core priority list below has been completed. The goal is to keep
extending this document and the invariant tests to every estimator that does
not yet have closed-form or first-order-condition coverage.

1. ANOVA — done
2. Arellano-Bond — done
3. ARIMA — done
4. ARDL / AutoReg — done
5. BART — done
6. Bayesian Linear — done
7. Bayesian Synthetic Control — done
8. Bayesian SFA — done
9. Beta regression — done
10. Between estimator — done
11. Binary Diagnostics — done
12. Biplot — done
13. Bootstrap / HypothesisTest — done
14. Bayesian VAR (BVAR) — Minnesota Prior — done
15. Causal Forest — done
16. Causal Impact — done
17. Conditional Logit / Poisson / MNLogit — done
18. Conformal Prediction — done
19. Copula — done
20. Poisson / Negative Binomial / GenPoisson / NegBinP — done
21. CUPED — done
22. DBSCAN — done
23. DCC-GARCH — done
24. Dynamic Factor Model (DFM) — done
25. Diff-in-Differences — done
26. DML Crossfit — done
27. Double ML — done
28. DR Learner — done
29. Dynamic Factor — done
30. Exponential Smoothing (ETS) — done
31. Event Study — done
32. Factor Analysis Panel — done
33. Fama-MacBeth — done
34. FAVAR — done
35. FMOLS — done
36. GLM-GAM — done
37. GARCH / EGARCH / GJRGARCH — done
38. Generalized Estimating Equations (GEE) — done
39. GLM — Generalized Linear Models — done
40. Gaussian Mixture clustering — done
41. Gaussian Process — done
42. Gradient Boosting — done
43. GRF — done
44. Hausman test — done
45. Hawkes process — done
46. Heckman Two-Step (Heckit) — done
47. Hierarchical clustering — done
48. Imputation (MICE / BayesGaussMI) — done
49. CUSUM / Influence — done
50. Isotonic regression — done
51. IV / 2SLS — Instrumental Variables — done
52. Johansen break — done
53. Kalman Filter and Local Level — done
54. K-Means — done
55. Local Projections DiD — done
56. LSTM — done
57. Markov autoregression — done
58. Markov switching — done
59. MFVAR — done
60. MIDAS — done
61. Mixed / BayesMixedGLM — done
62. MLP — done
63. Multinomial Logit — done
64. MS-VAR — done
65. MSTL — done
66. Multiple Tests — done
67. Multivariate tests — done
68. NARDL — done
69. NLS (Nonlinear Least Squares) — done
70. Nonparametric (KDE / Lowess / Kernel Reg / Local Level) — done
71. OLS — Ordinary Least Squares — done
72. Ordered Logit / Probit — done
73. Orthogonal Forest — done
74. Panel GLS — done
75. Panel GMM — Arellano-Bond (Diff-GMM) — done
76. Panel Heckman — done
77. Fixed Effects (Within Estimator) — done
78. Panel IV (FE-2SLS) — done
79. Panel Quantile — done
80. Random Effects Panel — done
81. Panel threshold (Hansen) — done
82. Panel Tobit (Random Effects) — done
83. Panel VAR — done
84. PCA — done
85. PCSE — done
86. WLS / GLSAR / FGLS / PCSE / PanelGLS — done
87. Proportion Tests — done
88. Propensity Score Matching — done
89. PSTR — done
90. QRF Inference — done
91. QRF — done
92. Quantile Regression — done
93. Quantile VAR — done
94. Random Forest — done
95. RD (Regression Discontinuity) — done
96. Regularization Path — Ridge / Lasso / ElasticNet — done
97. Robust Linear Model (RLM) — done
98. Robust Hausman — done
99. SETAR — done
100. Spatial Durbin Error — done
101. Spatial Durbin — done
102. Spatial — done
103. Spatial Panel — done
104. Spectral clustering — done
105. State Space (Kalman filter / smoother) — done
106. Stats (T-test / Compare means) — done
107. Stochastic Frontier — done
108. SUR / 3SLS — Seemingly Unrelated Regressions — done
109. Survival (Kaplan-Meier / Cox PH) — done
110. Stochastic Volatility (SV) — done
111. Synthetic Control — done
112. Synthetic DiD — done
113. System GMM — done
114. TimeSeries (ACF / PACF / ADF) — done
115. TMLE — done
116. Tobit — done
117. Transformer — done
118. t-SNE — done
119. Time-varying Copula — done
120. TVAR — done
121. TVP (Time-Varying Parameters) — done
122. TVP-VAR — done
123. UMAP — done
124. Unobserved Components — done
125. VAR / VARMA — done
126. XGBoost — done
127. ZIP / ZINB — done

---

## GARCH / EGARCH / GJRGARCH

### Specification

The GARCH(p,q) conditional variance is:

```
h_t = ω + Σ_{i=1}^q α_i ε_{t-i}² + Σ_{j=1}^p β_j h_{t-j}
```

with `ε_t = y_t - μ`. Stationarity requires `Σ α_i + Σ β_j < 1`. The
unconditional variance is `ω / (1 - Σ α_i - Σ β_j)`.

EGARCH models `ln h_t`; GJR-GARCH adds an asymmetry term `γ_i I(ε_{t-i}<0)`.

### Mapping to `src/garch.rs`

| Math | Code |
|---|---|
| `h_t` recursion | `garch_conditional_variance` |
| `ln h_t` | `egarch_conditional_variance` |
| Asymmetry | `gjrgarch_conditional_variance` |
| MLE | `optimize` over `neg_ll` with `garch_constrain` |

### Invariants verified in `tests/garch_invariants.rs`

1. Estimated `conditional_variance` matches the recursive formula using the
   estimated parameters.
2. `residuals = y - μ` and `standardized_residuals = residuals / sqrt(h)`.
3. Stationarity: `Σ α_i + Σ β_j < 1`.
4. Multi-step GARCH(1,1) forecast converges to the unconditional variance.
5. Invalid input (`q=0`, too few obs) is rejected.

---

## ARIMA

### Specification

`ARIMA(p,d,q)` with differenced series `z_t = Δ^d y_t`:

```
z_t = c + Σ_{i=1}^p φ_i z_{t-i} + u_t + Σ_{j=1}^q θ_j u_{t-j}
```

`src/arima.rs` uses Hannan-Rissanen two-step estimation:
1. Fit a long AR to obtain proxy residuals `û_t`.
2. Regress `z_t` on AR lags, MA lags (`û_{t-j}`), intercept and optional
   seasonal/exogenous terms.

### Invariants verified in `tests/arima_invariants.rs`

1. Noisy AR(1) data `y_t = c + φ y_{t-1} + ε_t` is recovered (`φ` and `c`
   close to truth).
2. ARIMA(0,1,0) on a random walk estimates the mean of the increments.
3. Input validation: too short series or too much differencing fail.

---

## DCC-GARCH

### Specification

Two-step procedure in `src/dcc_garch.rs`:

1. Fit univariate GARCH(1,1) to each series.
2. Model the conditional correlation matrix:

```
Q_t = (1 - a - b) Q̄ + a ε_{t-1} ε_{t-1}' + b Q_{t-1}
R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
```

### Invariants verified in `tests/dcc_garch_invariants.rs`

1. Conditional volatilities are positive.
2. GARCH(1,1) persistence is below the grid cap (`α + β < 0.99`).
3. Each `R_t` is symmetric, has unit diagonal and off-diagonals in `[-1,1]`.
4. `log_likelihood`, AIC and BIC are finite.
5. Input validation: too few observations or univariate data fail.

---

## Kalman Filter and Local Level

### Specification

Linear-Gaussian state space:

```
y_t = H s_t + e_t,    e_t ~ N(0, R_obs)
s_t = F s_{t-1} + R u_t,  u_t ~ N(0, Q)
```

`KalmanFilter::filter` performs the standard predict/update recursions:

```
v_t = y_t - H s_{t|t-1}
F_t = H P_{t|t-1} H' + R_obs
K_t = P_{t|t-1} H' F_t^{-1}
s_{t|t} = s_{t|t-1} + K_t v_t
P_{t|t} = (I - K_t H) P_{t|t-1}
s_{t+1|t} = F s_{t|t}
P_{t+1|t} = F P_{t|t} F' + R Q R'
```

`KalmanSmoother::smooth` runs the Rauch-Tung-Striebel backward pass.

### Invariants verified in `tests/kalman_invariants.rs`

1. `LocalLevel::fit` returns positive `sigma_obs` and `sigma_state`, finite
   log-likelihood, and the expected number of filtered/smoothed states.
2. Smoothed and filtered states at `t = T` coincide.
3. A constant state with observation noise is tracked: final filtered state
   converges to the true constant.
4. Smoother last state equals the filter last state.

---

## Diff-in-Differences

### Specification

Canonical 2×2 DiD regression:

```
y = β_0 + β_1 treated + β_2 post + δ (treated · post) + u
```

The ATT is `δ = (ȳ_{T,post} - ȳ_{T,pre}) - (ȳ_{C,post} - ȳ_{C,pre})`.

`src/did.rs` builds the interaction matrix and runs OLS.

### Invariants verified in `tests/did_invariants.rs`

1. `att` equals the hand-computed difference-in-differences for a balanced
   2×2 table.
2. Group means are correctly reported.
3. `R² ∈ [0,1]`.
4. Mismatched input lengths are rejected.

---

## Generalized Estimating Equations (GEE)

### Specification

GEE solves:

```
Σ_i D_i(β)' V_i(β)^{-1} (y_i - μ_i(β)) = 0
```

where `D_i = ∂μ_i/∂β`, `V_i = A_i^{1/2} R A_i^{1/2} · scale`, and `R` is the
working correlation matrix.

For Gaussian family + identity link + `Independence`, `R = I`, `A_i = I` and
`scale = σ²`, so the GEE estimating equation reduces to the OLS normal
equation.

### Invariants verified in `tests/gee_invariants.rs`

1. GEE with Gaussian, identity, `Independence` returns the same coefficients
   as pooled OLS.
2. The working correlation for `Independence` is the identity.
3. Input validation: mismatched dimensions fail.

---

## Heckman Two-Step (Heckit)

### Specification

Selection equation: `z_i* = w_i'γ + v_i,   z_i = 1(z_i* > 0)`
Outcome equation:  `y_i = x_i'β + u_i`, observed only if `z_i = 1`

`u_i` and `v_i` are bivariate normal with correlation `ρ`. The conditional
mean of `u_i` given selection is `ρ σ_ε λ_i` where
`λ_i = φ(w_i'γ) / Φ(w_i'γ)` is the inverse Mills ratio.

The two-step estimator:
1. Probit for `γ̂`.
2. OLS of `y` on `[x, λ(w'γ̂)]` for selected observations.
3. Variance correction using `δ̂ = ρ̂ σ̂_ε`.

`src/heckman.rs` implements exactly this.

### Invariants verified in `tests/heckman_invariants.rs`

1. When selection is independent of the outcome error (`ρ = 0`), the outcome
   coefficients are close to OLS on the selected sample and `δ̂` and `ρ̂` are
   close to 0.
2. `σ_ε > 0`, `n_obs` matches input and `n_selected > 0`.
3. Input validation: non-binary `z`, mismatched lengths and too few selected
   observations fail.

---

## WLS / GLSAR / FGLS / PCSE / PanelGLS

### Specification

**Weighted Least Squares** (`WLS`): minimize

```
Σ_i w_i (y_i - x_i'β)²
```

The equivalent OLS problem is `√w_i y_i` on `√w_i x_i`, implemented in
`FGLS::wls`.

**GLS with AR errors** (`GLSAR`): iterates between an AR(p) fit to the
residuals and quasi-differencing of `y` and `X`:

```
y*_t = y_t - Σ_{j=1}^p ρ_j y_{t-j}
X*_t = X_t - Σ_{j=1}^p ρ_j X_{t-j}
```

The final `β` is OLS on the transformed data.

**PCSE** (Beck-Katz) estimates `β` by pooled OLS and uses a sandwich
estimator with the panel residual covariance
`Σ̂_ij = e_i'e_j / T`.

**PanelGLS** estimates a Parks-type feasible GLS with either diagonal
entity-specific variances (`Hetero`) or full cross-sectional residual
covariance (`Correlated`).

### Invariants verified in `tests/wls_glsar_invariants.rs`

1. WLS with unit weights returns the same coefficients as OLS.
2. WLS with known inverse-variance weights recovers the true coefficients on
   heteroskedastic data.
3. `GLSAR(1)` recovers the true AR(1) error parameter and the slope on a
   no-intercept model.
4. Input validation: bad weight length, NaN/Inf, `ar_order=0` or too few obs.

### Invariants verified in `tests/pcse_panel_gls_invariants.rs`

1. PCSE point estimates coincide with pooled OLS on a homoskedastic balanced
   panel.
2. `PanelGLS::Hetero` on a balanced panel with equal variances is close to
   pooled OLS.
3. `PanelGLS::Correlated` converges and returns a sensible slope on a balanced
   panel.
4. Input validation: mismatched `entity_ids` / `time_ids` lengths.

---

## Tobit

### Specification

Censored regression with left limit `ll`:

```
y_i* = x_i'β + u_i,  u_i ~ N(0, σ²)
y_i = max(y_i*, ll)
```

Likelihood contributions:

- uncensored: `log φ((y_i - x_i'β)/σ) - log σ`
- censored: `log Φ((x_i'β - ll)/σ)`

`src/tobit.rs` uses Newton-Raphson on `(β, log σ)`.

### Invariants verified in `tests/tobit_invariants.rs`

1. When no observation is censored, Tobit MLE coincides with OLS.
2. A sample with all observations censored does not return a normal result
   (currently panics on the empty uncensored subsample).
3. With moderate censoring, Tobit recovers the true latent coefficients
   approximately.
4. Input validation: mismatched dimensions and NaN/Inf.

---

## Fama-MacBeth

### Specification

For each time period `t` run a cross-sectional OLS:

```
y_it = x_it'β_t + u_it
```

The Fama-MacBeth estimator is the time-series mean:

```
β̂_FM = (1/T) Σ_t β̂_t
SE(β̂_j) = sd(β̂_tj) / √(T-1)
```

Newey-West adjustment adds weighted autocovariances of `β̂_tj`.

### Invariants verified in `tests/fama_macbeth_invariants.rs`

1. Mean coefficient is close to the average of the period-by-period OLS
   estimates.
2. Newey-West standard errors are positive and finite.
3. Input validation: fewer than two usable periods or missing time column.

---

## PCA

### Specification

Standardize columns to `z_j = (x_j - x̄_j) / s_j`. The correlation matrix
`R = Z'Z / (n-1)` is decomposed as `R = E Λ E'`. Principal components are the
columns of `E` sorted by descending eigenvalues. Scores are `Z E`. Loadings are
`E diag(√λ)`.

### Invariants verified in `tests/pca_invariants.rs`

1. `components` columns are orthonormal.
2. `explained_variance_ratio` sums to 1 for full decomposition.
3. Total `explained_variance` equals the number of variables.
4. `scores` equal the standardized data times `components`.
5. `loadings` equal `components` scaled by `√eigenvalue`.
6. `n_components` is capped at the number of columns.
7. Perfectly collinear data has one PC explaining all variance.

---

## Random Effects Panel

### Specification

The Swamy-Arora random-effects estimator estimates

- `σ_u²` from the within (demeaned) residuals,
- `σ_α²` from the between regression,

and applies the GLS quasi-difference with

```
θ = 1 - √[σ_u² / (σ_u² + T σ_α²)]
y*_it = y_it - θ ȳ_i
X*_it = X_it - θ X̄_i
```

`src/panel.rs` returns the OLS estimate on `(y*, X*)`.

### Invariants verified in `tests/panel_random_effects_invariants.rs`

1. With no entity-specific effect, RE is close to pooled OLS and `θ` is small.
2. With strong entity-specific effects, the RE slope is close to the Fixed
   Effects slope and `θ` is large.
3. Input validation: mismatched entity-IDs length.

---

## Robust Linear Model (RLM)

### Specification

M-estimation via iteratively reweighted least squares:

```
β^{(k+1)} = (X' W^{(k)} X)^{-1} X' W^{(k)} y
W_i^{(k)} = ψ(r_i / σ) / (r_i / σ)
```

Scale is estimated by MAD / 0.6745. `src/rlm.rs` supports Huber, Tukey,
Andrew's wave and Hampel norms.

### Invariants verified in `tests/rlm_invariants.rs`

1. `RobustNorm::LeastSquares` reproduces OLS.
2. Huber RLM recovers the true coefficients on clean normal data.
3. Huber RLM is less affected by an extreme outlier than OLS.
4. Input validation: row-count mismatch.

---

## ZIP / ZINB

### Specification

Two-part model:

```
P(y_i = 0)    = π_i + (1 - π_i) f(0; μ_i)
P(y_i = k > 0)= (1 - π_i) f(k; μ_i)
log(μ_i)      = x_i'β
logit(π_i)    = z_i'γ
```

For ZIP, `f` is Poisson; for ZINB, `f` is Negative Binomial with dispersion
`α`. `src/zero_inflated.rs` fits via EM.

### Invariants verified in `tests/zip_invariants.rs`

1. ZIP recovers the count-model and inflation-model intercepts approximately on
   simulated data.
2. ZIP has `alpha = None`; ZINB has `alpha > 0`.
3. Both models converge, produce finite `log_likelihood`, AIC and BIC.
4. Input validation is implicit via `fit` returning `Err` on invalid data.

---

## Poisson / Negative Binomial / GenPoisson / NegBinP

### Specification

Count models assume

```
E[y_i | x_i] = μ_i = exp(x_i'β)
```

For Poisson, `Var(y_i) = μ_i`. Negative Binomial adds a dispersion `α` so
`Var(y_i) = μ_i + α μ_i²`. `Poisson::fit` and `NegBin::fit` are wrappers over
`GLM`; `NegBin` also profiles `α` to maximize the likelihood.

### Invariants verified in `tests/count_invariants.rs`

1. Poisson GLM recovers the true log-linear coefficients on simulated data.
2. Poisson with an exposure offset `ln(exposure)` recovers the rate
   parameters.
3. Negative Binomial with a known `α` recovers the coefficients.
4. Negative Binomial with automatic `α` converges and yields a positive
   dispersion estimate.
5. Generalized Poisson on Poisson data gives an `alpha` close to zero and
   recovers the mean parameters.
6. NegBinP with `p=2` (NB2) recovers the coefficients and a positive `alpha`
   on Gamma-Poisson data.

---

## NLS (Nonlinear Least Squares)

### Specification

NLS minimizes `Σ (y_i - f(β, x_i))²` using the Levenberg-Marquardt
algorithm with numerical Jacobians. `src/nls.rs` provides common functional
forms (`predict_exp`, `predict_power`, `predict_logistic`) and a generic
`fit` for user-supplied predictors.

### Invariants verified in `tests/nls_invariants.rs`

1. Constant and proportional models converge from a non-exact start.
2. Exponential and power-law models recover the true parameters.
3. Output fields (`params`, `std_errors`, `t_values`, `rss`, `r_squared`) are
   finite and have the expected dimensions.
4. Input validation: mismatched row counts are rejected.

---

## Between estimator

### Specification

The between estimator collapses each panel entity to its temporal means and
runs OLS on the `(n_entities, k)` collapsed data:

```
ȳ_i = β' x̄_i + (α_i + ū_i)
```

### Invariants verified in `tests/between_invariants.rs`

1. On data where `x` only varies between entities, the between slope equals
   OLS on the entity-level means.
2. Coefficients are close to the true between effect.
3. `n_entities` matches the number of unique IDs.
4. Input validation: mismatched entity-IDs length is rejected.

---

## Survival (Kaplan-Meier / Cox PH)

### Specification

Kaplan-Meier gives the product-limit survival estimate

```
Ŝ(t) = Π_{t_j ≤ t} (1 - d_j / n_j)
```

Cox PH maximizes the partial log-likelihood

```
ℓ(β) = Σ_{i: δ_i=1} [x_i'β - ln(Σ_{j: t_j ≥ t_i} exp(x_j'β))]
```

### Invariants verified in `tests/survival_invariants.rs`

1. KM with no censoring produces exactly `(n - cumulative events) / n`.
2. KM survival probabilities are non-increasing and lie in [0,1].
3. KM confidence bands are inside [0,1].
4. KM median is the first event time where survival ≤ 0.5.
5. Cox PH recovers the true log hazard ratio on simulated exponential data.
6. `hazard_ratios = exp(params)`, `concordance ∈ (0.5, 1]`, log-likelihood
   finite.
7. Input validation: mismatched lengths and no events are rejected.

---

## Markov switching

### Specification

A Markov-switching AR(p) model has

```
y_t = μ_{s_t} + Σ_{j=1}^p φ_{s_t,j} y_{t-j} + ε_t,  ε_t ~ N(0, σ²_{s_t})
P(s_t = j | s_{t-1} = i) = p_{ij}
```

`src/markov.rs` fits the model with the Hamilton filter / Kim smoother and an
EM M-step.

### Invariants verified in `tests/markov_switching_invariants.rs`

1. On simulated two-regime AR(1) data the fit is finite and has 2 regimes.
2. Transition matrix rows sum to 1 and the diagonal is dominant.
3. Filtered and smoothed probabilities are valid distributions (rows sum to
   1, all non-negative).
4. `regime_params` and `regime_variances` are finite and positive.
5. `predict` returns finite forecasts of the requested length.
6. Input validation: fewer than 2 regimes or too few observations are
   rejected.
7. Regime intercepts are ordered by the true regime means.
8. Smoothed state probabilities classify the true high-mean regime with
   accuracy > 0.65.

---

## Panel IV (FE-2SLS)

### Specification

Fixed-effects 2SLS removes entity means and instruments the endogenous
regressor in the within-transformed data:

```
ỹ = X̃ β + ũ
X̃ = Z̃ (Z̃'Z̃)^-1 Z̃'X̃ + residual
β = (X̂'X̃)^-1 X̂'ỹ
```

`src/panel.rs` (`FE2SLS`) implements this as `xtivreg, fe`.

### Invariants verified in `tests/panel_iv_invariants.rs`

1. With a valid excluded instrument, FE-2SLS recovers the structural
   coefficients of an endogenous regressor.
2. `n_entities` and `df_resid` are consistent with the panel dimensions.
3. `r_squared`, `sigma`, and all parameter fields are finite.
4. Input validation: mismatched dimensions, NaN data, and violated order
   condition are rejected.

---

## Hausman test

### Specification

The Hausman test compares the FE and RE estimators:

```
H = (β_fe - β_re)' [Var(β_fe) - Var(β_re)]^-1 (β_fe - β_re)  ~ χ²(k)
```

Under H0 the random-effects estimator is consistent and efficient (entity
effects are uncorrelated with the regressors); under H1 it is inconsistent.

### Invariants verified in `tests/hausman_invariants.rs`

1. When the entity effect is correlated with `x`, the test rejects H0 and
   recommends fixed effects.
2. When the entity effect is independent of `x`, the test fails to reject H0
   and recommends random effects.
3. Output string contains the Chi2 statistic, p-value and recommendation.

---

## Panel threshold (Hansen)

### Specification

The panel threshold model splits the sample according to a threshold variable
`q`:

```
y_it = (α_i +) β_1' x_it I(q_it ≤ γ) + β_2' x_it I(q_it > γ) + u_it
```

`PanelThreshold::fit` performs a grid search over `γ` and selects the value that
minimizes the SSR of a Fixed-Effects regression on the expanded design.

### Invariants verified in `tests/panel_threshold_invariants.rs`

1. On data with a true regime split, the estimated `threshold_gamma` is close
   to the true threshold and each regime's slope is close to the true value.
2. On single-regime data the threshold SSR is at most the plain FE SSR and the
   R² matches the FE R².
3. `r_squared`, `ssr_min` and `n_search` have the expected properties.
4. Input validation: mismatched lengths and low variability in `q` are
   rejected.

---

## TimeSeries (ACF / PACF / ADF)

### Specification

- `acf(k) = γ_k / γ_0` where `γ_k = Σ (y_t - ȳ)(y_{t-k} - ȳ) / n`.
- `pacf(k)` is computed via the Durbin-Levinson recursion.
- ADF tests the null of a unit root by regressing `Δy_t` on `y_{t-1}` plus lagged
  differences and comparing the t-statistic to Dickey-Fuller critical values.

### Invariants verified in `tests/timeseries_invariants.rs`

1. `acf[0] = 1` and `|acf[k]| ≤ 1`.
2. ACF of an AR(1) with coefficient `φ` has `acf[1] ≈ φ` and decays for higher
   lags.
3. PACF of an AR(1) has a single spike at lag 1 and near-zero values at higher
   lags.
4. ADF rejects the unit-root null for a stationary AR(1) and fails to reject it
   for a random walk.
5. Input validation: `nlags >= n` and too-short series are rejected.

---

## ARDL / AutoReg

### Specification

AutoReg is a linear regression of `y_t` on a constant/trend, own lags and
optional exogenous regressors. ARDL adds distributed lags of the exogenous
variables:

```
y_t = c + Σ_{j=1}^{p} ρ_j y_{t-j} + Σ_{m=0}^{q} β_m' x_{t-m} + u_t
```

Both are estimated by OLS on the lag design matrix.

### Invariants verified in `tests/autoreg_invariants.rs`

1. AutoReg recovers the AR(2) coefficients and intercept.
2. AutoReg with an exogenous regressor recovers the AR coefficient and the
   contemporaneous slope.
3. ARDL(1,1) recovers the constant, the AR(1) coefficient, and the
   contemporaneous and lag-1 exogenous effects.
4. Input validation: not enough observations or mismatched `x` rows are
   rejected.

---

## Ordered Logit / Probit

### Specification

For outcome `y_i ∈ {1,...,J}` the ordered model uses latent variable
`z_i = x_i'β - ε_i` and thresholds `α_1 < ... < α_{J-1}`:

```
P(y_i = j) = F(α_j - x_i'β) - F(α_{j-1} - x_i'β)
```

where `F` is the logistic CDF for logit and the normal CDF for probit.

### Invariants verified in `tests/ordered_invariants.rs`

1. Ordered Logit recovers the slope and produces monotone thresholds.
2. Ordered Probit recovers the slope and produces monotone thresholds.
3. All fields (`log_likelihood`, `aic`, `bic`, `p_values`, `n_obs`) are finite.
4. Input validation: fewer than 3 categories or NaN inputs are rejected.

---

## Beta regression

### Specification

For responses `y_i ∈ (0, 1)`:

```
y_i ~ Beta(μ_i φ, (1 - μ_i) φ)
g(μ_i) = x_i'β
```

`g` is a link (logit, probit, cloglog) and `φ` is the precision parameter.
`BetaModel::fit` uses BFGS on `[β; log φ]`.

### Invariants verified in `tests/beta_invariants.rs`

1. Logit link recovers the intercept and slope.
2. Probit link recovers the correct slope sign.
3. Precision parameter is positive and the likelihood is finite.
4. Input validation: `y` outside `(0,1)` or dimension mismatch is rejected.

---

## GLM-GAM

### Specification

GLM-GAM extends a generalized linear model with a smooth B-spline basis:

```
η = X_linear β + X_smooth γ
g(μ) = η
```

A penalty `α` is applied to `γ` and the model is fit by penalized IRLS.

### Invariants verified in `tests/gam_invariants.rs`

1. On data generated by a sinusoidal smooth function, the fitted values have
   high in-sample R² (> 0.85).
2. Effective degrees of freedom and scale are positive and finite.
3. `BSplineBasis::generate` rejects invalid `df < degree + 1`.
4. `GLMGam::fit` rejects row-count mismatches.

---

## Isotonic regression

### Specification

Isotonic regression solves

```
min  Σ (y_i - ŷ_i)²
s.t. ŷ_1 ≤ ŷ_2 ≤ ... ≤ ŷ_n
```

using the Pool Adjacent Violators Algorithm (PAVA).

### Invariants verified in `tests/isotonic_invariants.rs`

1. On increasing data, the fitted values are non-decreasing and R² is high.
2. On decreasing data (with `increasing = false`), the fitted values are
   non-increasing and R² is high.
3. `x_steps` and `y_steps` are consistent in length.
4. Weighted isotonic regression and dimension validation work.

---

## Multinomial Logit

### Specification

For `J` categories with base category `J`:

```
P(y_i = c) = exp(x_i'β_c) / (1 + Σ_{m=1}^{J-1} exp(x_i'β_m))
```

`MNLogit::fit` uses Newton-Raphson on the multinomial log-likelihood.

### Invariants verified in `tests/mnlogit_invariants.rs`

1. On simulated 3-category data, the coefficient matrix has the right shape
   and the recovered coefficients are close to the true values.
2. `log_likelihood`, `pseudo_r2`, `aic` and `bic` are finite.
3. Input validation: fewer than 3 categories or NaN inputs are rejected.

---

## ANOVA

### Specification

See the implementation and test file `tests/anova_invariants.rs`.

### Invariants verified in `tests/anova_invariants.rs`

1. One-way ANOVA returns a valid decomposition of sums of squares.
1. ANOVA for regression decomposes total variation into model and residual parts.
1. Input validation rejects mismatched dimensions, too few groups and empty data.

---

## Arellano-Bond

### Specification

See the implementation and test file `tests/arellano_bond_invariants.rs`.

### Invariants verified in `tests/arellano_bond_invariants.rs`

1. `arellano_bond_runs_and_recovers_parameters` runs and returns finite values.
1. `arellano_bond_two_step_produces_finite_output` runs and returns finite values.
1. `arellano_bond_input_validation` runs and returns finite values.

---

## BART

### Specification

See the implementation and test file `tests/bart_invariants.rs`.

### Invariants verified in `tests/bart_invariants.rs`

1. BART returns fitted values of the right length and finite diagnostics.
1. BART rejects too few observations or a zero-variance response.
1. Fitted values and in-sample MSE are self-consistent.

---

## Bayesian Linear

### Specification

See the implementation and test file `tests/bayesian_linear_invariants.rs`.

### Invariants verified in `tests/bayesian_linear_invariants.rs`

1. BayesianLinear recovers the true coefficients (up to sampling noise).
1. Fitted values are finite and have the same length as y.
1. Input validation rejects mismatched shapes and too few obs.

---

## Bayesian Synthetic Control

### Specification

See the implementation and test file `tests/bayesian_sc_invariants.rs`.

### Invariants verified in `tests/bayesian_sc_invariants.rs`

1. BayesianSC returns expected shapes and finite treatment effect estimates.
1. The counterfactual and observed series have the same length and reproduce y for pre-period.
1. Input validation rejects out-of-bounds treatment periods and mismatched lengths.

---

## Bayesian SFA

### Specification

See the implementation and test file `tests/bayesian_sfa_invariants.rs`.

### Invariants verified in `tests/bayesian_sfa_invariants.rs`

1. BayesianSFA returns finite coefficients and efficiency estimates.
1. Cost frontier returns the expected model type.
1. Input validation rejects mismatched row counts.

---

## Biplot

### Specification

See the implementation and test file `tests/biplot_invariants.rs`.

### Invariants verified in `tests/biplot_invariants.rs`

1. Biplot returns consistent shapes for all three biplot types.
1. Biplot handles custom variable names and validates input.

---

## Bootstrap / HypothesisTest

### Specification

See the implementation and test file `tests/bootstrap_invariants.rs`.

### Invariants verified in `tests/bootstrap_invariants.rs`

1. Pairs bootstrap yields finite coefficients and ordered percentile intervals.
1. Hypothesis tests return finite statistics and valid p-values.
1. Input validation rejects mismatched shapes and invalid model comparisons.

---

## Causal Forest

### Specification

See the implementation and test file `tests/causal_forest_invariants.rs`.

### Invariants verified in `tests/causal_forest_invariants.rs`

1. CausalForest returns finite treatment effect estimates and a reasonable ATE.
1. Input validation catches mismatched dimensions and insufficient data.

---

## Causal Impact

### Specification

See the implementation and test file `tests/causal_impact_invariants.rs`.

### Invariants verified in `tests/causal_impact_invariants.rs`

1. CausalImpact returns expected shapes and finite post-treatment effects.
1. Pre-period counterfactual equals observed y (in-sample) for a noiseless design.
1. Input validation rejects out-of-bounds treatment periods and mismatched rows.

---

## Conformal Prediction

### Specification

See the implementation and test file `tests/conformal_invariants.rs`.

### Invariants verified in `tests/conformal_invariants.rs`

1. Conformal prediction returns intervals that contain the point prediction.
1. Coverage level is the complement of miscoverage.
1. Input validation rejects mismatched shapes and too few observations.

---

## Copula

### Specification

See the implementation and test file `tests/copula_invariants.rs`.

### Invariants verified in `tests/copula_invariants.rs`

1. Fit a Gaussian copula to bivariate normal data and check that the estimated correlation is close to the true latent correlation.
1. All implemented copula types return finite results and expected shapes.
1. Input validation rejects too few observations or a single variable.

---

## CUPED

### Specification

See the implementation and test file `tests/cuped_invariants.rs`.

### Invariants verified in `tests/cuped_invariants.rs`

1. `cuped_univariate_runs_and_recovers` runs and returns finite values.
1. `cuped_multivariate_runs_and_recovers` runs and returns finite values.
1. `cuped_input_validation` runs and returns finite values.

---

## DBSCAN

### Specification

See the implementation and test file `tests/dbscan_invariants.rs`.

### Invariants verified in `tests/dbscan_invariants.rs`

1. DBSCAN finds two clusters and flags the outlier as noise.
1. All noise: DBSCAN with too small eps returns only noise.
1. Input validation catches invalid parameters.

---

## Dynamic Factor Model (DFM)

### Specification

See the implementation and test file `tests/dfm_invariants.rs`.

### Invariants verified in `tests/dfm_invariants.rs`

1. `test_dfm_runs_and_produces_finite_output` runs and returns finite values.
1. `test_dfm_factor_ar_is_stable` runs and returns finite values.
1. `test_dfm_input_validation` runs and returns finite values.

---

## DML Crossfit

### Specification

See the implementation and test file `tests/dml_crossfit_invariants.rs`.

### Invariants verified in `tests/dml_crossfit_invariants.rs`

1. DMLCrossfit returns a finite treatment effect and MSEs.
1. The confidence interval contains the point estimate.
1. Input validation rejects mismatched sizes and too few observations.

---

## Double ML

### Specification

See the implementation and test file `tests/double_ml_invariants.rs`.

### Invariants verified in `tests/double_ml_invariants.rs`

1. DoubleML returns finite treatment effect and residuals.
1. y_tilde and d_tilde are centered (mean close to zero by construction).
1. Input validation rejects mismatched sizes and too few folds.

---

## DR Learner

### Specification

See the implementation and test file `tests/dr_learner_invariants.rs`.

### Invariants verified in `tests/dr_learner_invariants.rs`

1. DRLearner returns finite CATE predictions and ATE.
1. CATE regression coefficients are finite and have expected length.
1. Input validation rejects too few obs or imbalanced treatment.

---

## Dynamic Factor

### Specification

See the implementation and test file `tests/dynamic_factor_invariants.rs`.

### Invariants verified in `tests/dynamic_factor_invariants.rs`

1. `test_dynamic_factor_runs_and_produces_finite_output` runs and returns finite values.
1. `test_dynamic_factor_predict_and_ar_stability` runs and returns finite values.
1. `test_dynamic_factor_input_validation` runs and returns finite values.

---

## Exponential Smoothing (ETS)

### Specification

See the implementation and test file `tests/ets_invariants.rs`.

### Invariants verified in `tests/ets_invariants.rs`

1. `test_ets_level_and_predict` runs and returns finite values.
1. `test_ets_trend_and_seasonal` runs and returns finite values.
1. `test_ets_input_validation` runs and returns finite values.

---

## Event Study

### Specification

See the implementation and test file `tests/event_study_invariants.rs`.

### Invariants verified in `tests/event_study_invariants.rs`

1. `event_study_runs_and_recovers_post_effects` runs and returns finite values.
1. `event_study_with_controls` runs and returns finite values.
1. `event_study_input_validation` runs and returns finite values.

---

## Factor Analysis Panel

### Specification

See the implementation and test file `tests/fa_panel_invariants.rs`.

### Invariants verified in `tests/fa_panel_invariants.rs`

1. `fa_panel_runs_and_recovers_parameters` runs and returns finite values.
1. `fa_panel_input_validation` runs and returns finite values.

---

## FAVAR

### Specification

See the implementation and test file `tests/favar_invariants.rs`.

### Invariants verified in `tests/favar_invariants.rs`

1. `test_favar_runs_and_produces_finite_output` runs and returns finite values.
1. `test_favar_factor_recovery` runs and returns finite values.
1. `test_favar_input_validation` runs and returns finite values.

---

## FMOLS

### Specification

See the implementation and test file `tests/fmols_invariants.rs`.

### Invariants verified in `tests/fmols_invariants.rs`

1. FMOLS returns finite coefficients and expected shapes.
1. FMOLS rejects shape mismatches and too few observations.
1. Coefficient signs and magnitudes are stable across two random seeds.

---

## Gaussian Mixture clustering

### Specification

See the implementation and test file `tests/gmm_clustering_invariants.rs`.

### Invariants verified in `tests/gmm_clustering_invariants.rs`

1. GMM recovers three well-separated Gaussian clusters.
1. GMM centroids are near the true cluster means.
1. Input validation catches invalid cluster counts and too few observations.

---

## Gaussian Process

### Specification

See the implementation and test file `tests/gp_invariants.rs`.

### Invariants verified in `tests/gp_invariants.rs`

1. GP fit returns correct shapes, finite values, and a reasonable fit.
1. Input validation: too few observations, no features, or zero-variance y.

---

## Gradient Boosting

### Specification

See the implementation and test file `tests/gradient_boosting_invariants.rs`.

### Invariants verified in `tests/gradient_boosting_invariants.rs`

1. Gradient boosting returns correct shapes and a reasonable fit.
1. Input validation catches invalid dimensions and zero trees.

---

## GRF

### Specification

See the implementation and test file `tests/grf_invariants.rs`.

### Invariants verified in `tests/grf_invariants.rs`

1. GRF returns finite CATE estimates and a reasonable ATE.
1. Input validation catches mismatched dimensions and insufficient support.

---

## Hawkes process

### Specification

See the implementation and test file `tests/hawkes_invariants.rs`.

### Invariants verified in `tests/hawkes_invariants.rs`

1. Hawkes fit returns finite parameters and expected shapes on sorted event times.
1. When no time_window is supplied, it defaults to the last event time.
1. Input validation rejects too few events, unsorted times, and non-positive windows.

---

## Hierarchical clustering

### Specification

See the implementation and test file `tests/hierarchical_invariants.rs`.

### Invariants verified in `tests/hierarchical_invariants.rs`

1. Hierarchical clustering with Ward linkage returns correct shapes and clusters.
1. All four linkages produce finite, consistent results on the same data.
1. Input validation catches insufficient observations.

---

## Imputation (MICE / BayesGaussMI)

### Specification

See the implementation and test file `tests/imputation_invariants.rs`.

### Invariants verified in `tests/imputation_invariants.rs`

1. MICE produces the requested number of imputed datasets without NaNs.
1. Bayesian Gaussian MI produces imputed datasets of the expected shape.
1. Input validation rejects empty data and length mismatches.

---

## CUSUM / Influence

### Specification

See the implementation and test file `tests/influence_invariants.rs`.

### Invariants verified in `tests/influence_invariants.rs`

1. Influence diagnostics return the expected shapes on OLS residuals.
1. CUSUM test returns expected shapes and a boolean stability flag.
1. Input validation rejects mismatched sizes and too few obs.

---

## Johansen break

### Specification

See the implementation and test file `tests/johansen_break_invariants.rs`.

### Invariants verified in `tests/johansen_break_invariants.rs`

1. JohansenBreak returns the expected output shapes and finite trace stats.
1. Including break points produces the same rank space and records them.
1. Input validation rejects insufficient observations or zero lags.

---

## K-Means

### Specification

See the implementation and test file `tests/kmeans_invariants.rs`.

### Invariants verified in `tests/kmeans_invariants.rs`

1. KMeans recovers three well-separated clusters with correct shapes.
1. KMeans centroids are near the true cluster centers.
1. Input validation catches impossible clustering requests.

---

## Local Projections DiD

### Specification

See the implementation and test file `tests/lp_did_invariants.rs`.

### Invariants verified in `tests/lp_did_invariants.rs`

1. `lp_did_runs_and_produces_horizons` runs and returns finite values.
1. `lp_did_post_effects_positive` runs and returns finite values.
1. `lp_did_input_validation` runs and returns finite values.

---

## LSTM

### Specification

See the implementation and test file `tests/lstm_invariants.rs`.

### Invariants verified in `tests/lstm_invariants.rs`

1. LSTM returns fitted and forecast series of the expected lengths.
1. Forecast length and n_hidden follow the defaults and requested values.
1. Input validation rejects short series and zero variance.

---

## Markov autoregression

### Specification

See the implementation and test file `tests/markov_autoreg_invariants.rs`.

### Invariants verified in `tests/markov_autoreg_invariants.rs`

1. `test_markov_autoreg_runs_and_produces_finite_output` runs and returns finite values.
1. `test_markov_autoreg_separates_regimes_and_predicts` runs and returns finite values.
1. `test_markov_autoreg_input_validation` runs and returns finite values.

---

## MFVAR

### Specification

See the implementation and test file `tests/mfvar_invariants.rs`.

### Invariants verified in `tests/mfvar_invariants.rs`

1. `test_mfvar_runs_and_produces_finite_output` runs and returns finite values.
1. `test_mfvar_recovers_positive_high_freq_effect` runs and returns finite values.
1. `test_mfvar_input_validation` runs and returns finite values.

---

## MIDAS

### Specification

See the implementation and test file `tests/midas_invariants.rs`.

### Invariants verified in `tests/midas_invariants.rs`

1. MIDAS returns finite coefficients and expected shapes.
1. MIDAS recovers positive beta on a known positive relationship.
1. Input validation rejects insufficient high-frequency data and invalid parameters.

---

## Mixed / BayesMixedGLM

### Specification

See the implementation and test file `tests/mixed_invariants.rs`.

### Invariants verified in `tests/mixed_invariants.rs`

1. MixedLM returns finite fixed and random effects with the expected shapes.
1. BayesMixedGLM returns finite posterior summaries for binomial data.
1. Input validation rejects dimension mismatches for MixedLM.

---

## MLP

### Specification

See the implementation and test file `tests/mlp_invariants.rs`.

### Invariants verified in `tests/mlp_invariants.rs`

1. MLP returns correct shapes and finite values.
1. Input validation catches invalid dimensions and zero hidden units.

---

## MS-VAR

### Specification

See the implementation and test file `tests/ms_var_invariants.rs`.

### Invariants verified in `tests/ms_var_invariants.rs`

1. `test_msvar_runs_and_produces_finite_output` runs and returns finite values.
1. `test_msvar_probabilities_are_valid_and_ar_stable` runs and returns finite values.
1. `test_msvar_input_validation` runs and returns finite values.

---

## MSTL

### Specification

See the implementation and test file `tests/mstl_invariants.rs`.

### Invariants verified in `tests/mstl_invariants.rs`

1. `test_mstl_runs_and_reconstructs_series` runs and returns finite values.
1. `test_mstl_trend_is_smooth_and_seasonals_oscillate` runs and returns finite values.
1. `test_mstl_input_validation` runs and returns finite values.

---

## Multiple Tests

### Specification

See the implementation and test file `tests/multipletests_invariants.rs`.

### Invariants verified in `tests/multipletests_invariants.rs`

1. Multiple tests methods return corrected p-values and rejection flags of the correct length.
1. Corrected p-values are monotonically non-decreasing across the methods (or at least bounded).
1. Input validation rejects empty slices, out-of-range p-values and invalid alpha.

---

## Multivariate tests

### Specification

See the implementation and test file `tests/multivariate_invariants.rs`.

### Invariants verified in `tests/multivariate_invariants.rs`

1. Factor analysis returns loadings and communalities with the expected shapes.
1. MANOVA and canonical correlation return finite multivariate test statistics.
1. Input validation catches insufficient observations or mismatched dimensions.

---

## NARDL

### Specification

See the implementation and test file `tests/nardl_invariants.rs`.

### Invariants verified in `tests/nardl_invariants.rs`

1. `test_nardl_runs_and_produces_finite_output` runs and returns finite values.
1. `test_nardl_recovers_long_run_multiplier` runs and returns finite values.
1. `test_nardl_input_validation` runs and returns finite values.

---

## Orthogonal Forest

### Specification

See the implementation and test file `tests/orthogonal_forest_invariants.rs`.

### Invariants verified in `tests/orthogonal_forest_invariants.rs`

1. OrthogonalRandomForest returns finite CATE estimates and a reasonable ATE.
1. Input validation catches mismatched dimensions and insufficient data.

---

## Panel GLS

### Specification

See the implementation and test file `tests/panel_gls_invariants.rs`.

### Invariants verified in `tests/panel_gls_invariants.rs`

1. `panel_gls_hetero_close_to_ols` runs and returns finite values.
1. `panel_gls_correlated_finite` runs and returns finite values.
1. `panel_gls_input_validation` runs and returns finite values.

---

## Panel Heckman

### Specification

See the implementation and test file `tests/panel_heckman_invariants.rs`.

### Invariants verified in `tests/panel_heckman_invariants.rs`

1. `panel_heckman_runs_and_recovers_parameters` runs and returns finite values.
1. `panel_heckman_input_validation` runs and returns finite values.

---

## Panel Quantile

### Specification

See the implementation and test file `tests/panel_quantile_invariants.rs`.

### Invariants verified in `tests/panel_quantile_invariants.rs`

1. `panel_quantile_runs_and_recovers_parameters` runs and returns finite values.
1. `panel_quantile_input_validation` runs and returns finite values.

---

## Panel VAR

### Specification

See the implementation and test file `tests/panel_var_invariants.rs`.

### Invariants verified in `tests/panel_var_invariants.rs`

1. `test_panel_var_runs_and_produces_finite_output` runs and returns finite values.
1. `test_panel_var_coefficients_and_pvalues_are_reasonable` runs and returns finite values.
1. `test_panel_var_input_validation` runs and returns finite values.

---

## PCSE

### Specification

See the implementation and test file `tests/pcse_invariants.rs`.

### Invariants verified in `tests/pcse_invariants.rs`

1. `pcse_equals_ols_homoskedastic` runs and returns finite values.
1. `pcse_rejects_non_finite` runs and returns finite values.
1. `pcse_input_validation` runs and returns finite values.

---

## Proportion Tests

### Specification

See the implementation and test file `tests/proportion_invariants.rs`.

### Invariants verified in `tests/proportion_invariants.rs`

1. One-sample z-test recovers the expected value for a simple case.
1. Two-sample z-test and confidence intervals return finite, ordered results.
1. Contingency table chi-square is consistent with a manual computation.

---

## Propensity Score Matching

### Specification

See the implementation and test file `tests/psm_invariants.rs`.

### Invariants verified in `tests/psm_invariants.rs`

1. `psm_runs_and_recovers_att` runs and returns finite values.
1. `psm_input_validation` runs and returns finite values.

---

## PSTR

### Specification

See the implementation and test file `tests/pstr_invariants.rs`.

### Invariants verified in `tests/pstr_invariants.rs`

1. `pstr_runs_and_recovers_parameters` runs and returns finite values.
1. `pstr_input_validation` runs and returns finite values.

---

## QRF Inference

### Specification

See the implementation and test file `tests/qrf_inference_invariants.rs`.

### Invariants verified in `tests/qrf_inference_invariants.rs`

1. QRFInference returns shaped, finite point estimates and confidence bounds.
1. Input validation catches insufficient observations and invalid quantiles.

---

## QRF

### Specification

See the implementation and test file `tests/qrf_invariants.rs`.

### Invariants verified in `tests/qrf_invariants.rs`

1. QRF quantile predictions have the expected shape, are finite, and are monotonic.
1. Input validation catches invalid dimensions, zero trees, and bad quantiles.

---

## Quantile VAR

### Specification

See the implementation and test file `tests/quantile_var_invariants.rs`.

### Invariants verified in `tests/quantile_var_invariants.rs`

1. `test_quantile_var_runs_and_produces_finite_output` runs and returns finite values.
1. `test_quantile_var_irf_shape_and_median_recovery` runs and returns finite values.
1. `test_quantile_var_input_validation` runs and returns finite values.

---

## Random Forest

### Specification

See the implementation and test file `tests/random_forest_invariants.rs`.

### Invariants verified in `tests/random_forest_invariants.rs`

1. Random forest returns correct shapes and a reasonable fit.
1. Input validation catches invalid dimensions and zero trees.

---

## Robust Hausman

### Specification

See the implementation and test file `tests/robust_hausman_invariants.rs`.

### Invariants verified in `tests/robust_hausman_invariants.rs`

1. `robust_hausman_classical_runs` runs and returns finite values.
1. `robust_hausman_compare_arrays_runs` runs and returns finite values.
1. `robust_hausman_compare_runs` runs and returns finite values.
1. `robust_hausman_input_validation` runs and returns finite values.

---

## SETAR

### Specification

See the implementation and test file `tests/setar_invariants.rs`.

### Invariants verified in `tests/setar_invariants.rs`

1. `test_setar_runs_and_produces_finite_output` runs and returns finite values.
1. `test_setar_recovers_threshold_and_regime_coefficients` runs and returns finite values.
1. `test_setar_input_validation` runs and returns finite values.

---

## Spectral clustering

### Specification

See the implementation and test file `tests/spectral_invariants.rs`.

### Invariants verified in `tests/spectral_invariants.rs`

1. Spectral clustering recovers two well-separated clusters.
1. Different numbers of components produce consistent shapes.
1. Input validation catches invalid cluster counts and too few observations.

---

## Stats (T-test / Compare means)

### Specification

See the implementation and test file `tests/stats_invariants.rs`.

### Invariants verified in `tests/stats_invariants.rs`

1. CompareMeans returns finite statistics and the expected sign of the mean difference.
1. TTest 1-sample and 2-sample functions return consistent full results.
1. Input validation rejects too few observations or mismatched paired data.

---

## Stochastic Frontier

### Specification

See the implementation and test file `tests/stochastic_frontier_invariants.rs`.

### Invariants verified in `tests/stochastic_frontier_invariants.rs`

1. StochasticFrontier returns finite parameters and expected shapes.
1. Cost frontier returns the expected model type and finite mean efficiency.
1. Input validation rejects mismatched shapes.

---

## Stochastic Volatility (SV)

### Specification

See the implementation and test file `tests/sv_invariants.rs`.

### Invariants verified in `tests/sv_invariants.rs`

1. SV fit returns finite parameters and expected shapes.
1. Input validation rejects too few observations.
1. Log-likelihood increases with more observations for stationary data.

---

## Synthetic Control

### Specification

See the implementation and test file `tests/synthetic_control_invariants.rs`.

### Invariants verified in `tests/synthetic_control_invariants.rs`

1. `synthetic_control_runs_and_produces_weights` runs and returns finite values.
1. `synthetic_control_rejects_invalid_treated_unit` runs and returns finite values.
1. `synthetic_control_rejects_too_few_pre_periods` runs and returns finite values.

---

## Synthetic DiD

### Specification

See the implementation and test file `tests/synthetic_did_invariants.rs`.

### Invariants verified in `tests/synthetic_did_invariants.rs`

1. `synthetic_did_runs_and_recovers_att` runs and returns finite values.
1. `synthetic_did_input_validation` runs and returns finite values.

---

## System GMM

### Specification

See the implementation and test file `tests/system_gmm_invariants.rs`.

### Invariants verified in `tests/system_gmm_invariants.rs`

1. `system_gmm_runs_and_recovers_parameters` runs and returns finite values.
1. `system_gmm_two_step_produces_finite_output` runs and returns finite values.
1. `system_gmm_input_validation` runs and returns finite values.

---

## TMLE

### Specification

See the implementation and test file `tests/tmle_invariants.rs`.

### Invariants verified in `tests/tmle_invariants.rs`

1. TMLE returns finite ATE and standard error.
1. Propensity scores and EIF are bounded and have correct length.
1. Input validation rejects mismatched lengths and too few treated/control.

---

## Transformer

### Specification

See the implementation and test file `tests/transformer_invariants.rs`.

### Invariants verified in `tests/transformer_invariants.rs`

1. Transformer returns shaped, finite in-sample and forecast values.
1. Input validation catches short series and zero-variance input.

---

## t-SNE

### Specification

See the implementation and test file `tests/tsne_invariants.rs`.

### Invariants verified in `tests/tsne_invariants.rs`

1. t-SNE returns an embedding with the requested number of dimensions.
1. t-SNE supports 3D output and respects input constraints.

---

## Time-varying Copula

### Specification

See the implementation and test file `tests/tv_copula_invariants.rs`.

### Invariants verified in `tests/tv_copula_invariants.rs`

1. TvCopula fits and returns the expected shapes for all copula types. Outputs may be NaN for some types on random data, so only shape invariants are enforced.
1. A deterministic, well-separated design returns a finite result.
1. Input validation rejects too few obs or a single variable.

---

## TVAR

### Specification

See the implementation and test file `tests/tvar_invariants.rs`.

### Invariants verified in `tests/tvar_invariants.rs`

1. `test_tvar_runs_and_produces_finite_output` runs and returns finite values.
1. `test_tvar_regime_directions_are_reasonable` runs and returns finite values.
1. `test_tvar_input_validation` runs and returns finite values.

---

## UMAP

### Specification

See the implementation and test file `tests/umap_invariants.rs`.

### Invariants verified in `tests/umap_invariants.rs`

1. UMAP returns an embedding with the requested number of dimensions.
1. UMAP supports 1D output and respects input constraints.

---

## Unobserved Components

### Specification

See the implementation and test file `tests/unobserved_components_invariants.rs`.

### Invariants verified in `tests/unobserved_components_invariants.rs`

1. `test_uc_local_level_runs_and_produces_finite_output` runs and returns finite values.
1. `test_uc_local_linear_trend` runs and returns finite values.
1. `test_uc_input_validation` runs and returns finite values.

---

## XGBoost

### Specification

See the implementation and test file `tests/xgboost_invariants.rs`.

### Invariants verified in `tests/xgboost_invariants.rs`

1. XGBoost returns correct shapes and a reasonable fit.
1. Input validation catches invalid dimensions and zero trees.

---

## Binary Diagnostics

### Specification

See the implementation and test file `tests/binary_diagnostics_invariants.rs`.

### Invariants verified in `tests/binary_diagnostics_invariants.rs`

1. Binary diagnostics return finite classification, ROC and Hosmer-Lemeshow results.
1. Linktest returns finite coefficients and specification diagnostics.
1. Input validation rejects mismatched lengths and degenerate ROC/H-L inputs.

---

## Conditional Logit / Poisson / MNLogit

### Specification

See the implementation and test file `tests/conditional_invariants.rs`.

### Invariants verified in `tests/conditional_invariants.rs`

1. Conditional logit returns finite coefficients and group diagnostics.
1. Conditional Poisson returns finite coefficients for panel count data.
1. Conditional multinomial logit returns coefficients for repeated choice sets.
1. Input validation rejects mismatched dimensions, empty groups and degenerate choices.

---

## Nonparametric (KDE / Lowess / Kernel Reg / Local Level)

### Specification

See the implementation and test file `tests/nonparametric_invariants.rs`.

### Invariants verified in `tests/nonparametric_invariants.rs`

1. Univariate and multivariate KDE return finite densities and bandwidths with expected shapes.
1. LOWESS and kernel regression return fitted values that reconstruct the response shape.
1. Local-level state-space model estimates finite variances and state paths.
1. Input validation catches mismatched lengths, too few observations and bad bandwidths.

---

## RD (Regression Discontinuity)

### Specification

See the implementation and test file `tests/rd_invariants.rs`.

### Invariants verified in `tests/rd_invariants.rs`

1. Sharp RD recovers a finite treatment effect near the true jump.
1. Fuzzy RD returns a finite LATE and first-stage diagnostics.
1. Input validation rejects length mismatches, NaNs, and insufficient local samples.

---

## Spatial Durbin Error

### Specification

See the implementation and test file `tests/spatial_durbin_error_invariants.rs`.

### Invariants verified in `tests/spatial_durbin_error_invariants.rs`

1. Spatial Durbin error model returns finite direct/indirect effects and a spatial error parameter.
1. Input validation catches dimension and panel ID mismatches.
1. SDEM is stable with a fully connected row-standardised weights matrix.

---

## Spatial Durbin

### Specification

See the implementation and test file `tests/spatial_durbin_invariants.rs`.

### Invariants verified in `tests/spatial_durbin_invariants.rs`

1. Spatial Durbin panel fit returns direct and indirect effects with the expected shapes.
1. Input validation rejects mismatched dimensions or an inconsistent panel layout.
1. Spatial Durbin handles a larger cross-section with a denser weights matrix.

---

## Spatial

### Specification

See the implementation and test file `tests/spatial_invariants.rs`.

### Invariants verified in `tests/spatial_invariants.rs`

1. SAR and SEM fits return finite coefficients, spatial parameters and diagnostics.
1. Input validation rejects mismatched dimensions for the weights or design matrix.
1. A pure-noise dependent variable still yields a bounded spatial parameter and R-squared.

---

## Spatial Panel

### Specification

See the implementation and test file `tests/spatial_panel_invariants.rs`.

### Invariants verified in `tests/spatial_panel_invariants.rs`

1. Spatial panel SAR and SEM return finite spatial parameters and within coefficients.
1. Input validation rejects dimension mismatches and inconsistent entity IDs.
1. A larger panel with a denser weights matrix still yields bounded estimates.

---

## State Space (Kalman filter / smoother)

### Specification

See the implementation and test file `tests/statespace_invariants.rs`.

### Invariants verified in `tests/statespace_invariants.rs`

1. State-space estimation returns filtered and smoothed states with the expected shapes.
1. Kalman filter and smoother can be run independently and agree on state count.
1. Local-level model estimates finite variances and state paths.
1. Input validation catches too few observations for local-level estimation.

---

## TVP (Time-Varying Parameters)

### Specification

See the implementation and test file `tests/tvp_invariants.rs`.

### Invariants verified in `tests/tvp_invariants.rs`

1. TVP fit returns smoothed coefficients with the expected shape and finite statistics.
1. TVP rejects a mismatch between y length and the number of x rows.
1. The smoothed coefficient path is finite and retains the correct shape.

---

## TVP-VAR

### Specification

See the implementation and test file `tests/tvp_var_invariants.rs`.

### Invariants verified in `tests/tvp_var_invariants.rs`

1. TVP-VAR fit returns smoothed coefficients and a covariance matrix with the expected shapes.
1. TVP-VAR rejects too few observations or lags set to zero.
1. Smoothed TVP-VAR coefficients have bounded variation around the initial OLS estimates.

For iterative estimators the proof will be split into:
- derivation of the objective function and gradient;
- proof that the iterative routine satisfies the first-order conditions at
  convergence;
- tests that the gradient is numerically zero and the Hessian has the right
  definiteness.
