# Greeners Roadmap: Paridade com statsmodels

Este documento mapeia todas as funcionalidades do statsmodels (v0.14) e compara com o estado atual do Greeners. Serve como guia permanente para implementacoes futuras — consulte-o no inicio de cada sessao.

Legenda: ✅ Implementado | 🔶 Parcial | ❌ Ausente

**Ultima atualizacao:** 2026-02-02

---

## 1. Regression (statsmodels.regression)

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| OLS | `sm.OLS` | `OLS` | ✅ | HC1-HC4, NeweyWest, Clustered, TwoWay |
| WLS | `sm.WLS` | `WLS` | ✅ | Interface dedicada com pesos explicitos |
| GLS | `sm.GLS` | `FGLS` | ✅ | |
| GLSAR | `sm.GLSAR` | `GLSAR` | ✅ | GLS com erros AR |
| RecursiveLS | `RecursiveLS` | `RecursiveLS` | ✅ | Minimos quadrados recursivos |
| RollingOLS | `RollingOLS` | `RollingOLS` | ✅ | OLS com janela movel |
| RollingWLS | `RollingWLS` | `RollingWLS` | ✅ | WLS com janela movel |
| QuantReg | `QuantReg` | `QuantileReg` | ✅ | |

## 2. Generalized Linear Models (statsmodels.genmod)

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| GLM | `sm.GLM` | `GLM` | ✅ | Gaussian, Binomial, Poisson, Gamma, InvGaussian, Tweedie, NegBin |
| Links | 10+ | 10 | ✅ | Identity, Log, Logit, Probit, InvPower, InvSq, CLogLog, Power, NegBin, Cauchy |
| GEE | `GEE` | `GEE` | ✅ | Exchangeable, AR(1), unstructured, independence |
| NominalGEE | `NominalGEE` | `NominalGEE` | ✅ | Baseline-category logit |
| OrdinalGEE | `OrdinalGEE` | `OrdinalGEE` | ✅ | Cumulative logit |
| GLMGam | `GLMGam` | `GLMGam` | ✅ | GAM via penalized B-splines |
| BayesMixedGLM | `BinomialBayesMixedGLM` | `BayesMixedGLM` | ✅ | Laplace approximation |

## 3. Discrete Choice (statsmodels.discrete)

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| Logit | `Logit` | `Logit` | ✅ | AME + MEM |
| Probit | `Probit` | `Probit` | ✅ | AME + MEM |
| MNLogit | `MNLogit` | `MNLogit` | ✅ | Softmax, RRR, predict_proba |
| OrderedModel | `OrderedModel` | `OrderedLogit`/`OrderedProbit` | ✅ | Threshold reparametrization |
| Poisson (dedicado) | `Poisson` | `Poisson` | ✅ | Overdispersion test, exposure, AME |
| NegativeBinomial | `NegativeBinomial` | `NegBin` | ✅ | Profile likelihood alpha, LR test |
| NegativeBinomialP | `NegativeBinomialP` | `NegBinP` | ✅ | NB1 (p=1), NB2 (p=2), flexible p |
| GeneralizedPoisson | `GeneralizedPoisson` | `GenPoisson` | ✅ | Newton-Raphson MLE |
| ZeroInflatedPoisson | `ZeroInflatedPoisson` | `ZIP` | ✅ | EM algorithm |
| ZeroInflatedNB | `ZeroInflatedNegativeBinomialP` | `ZINB` | ✅ | EM algorithm |
| ConditionalLogit | `ConditionalLogit` | `ConditionalLogit` | ✅ | Chamberlain conditional MLE |
| ConditionalPoisson | `ConditionalPoisson` | `ConditionalPoisson` | ✅ | Hausman-Hall-Griliches |
| ConditionalMNLogit | `ConditionalMNLogit` | `ConditionalMNLogit` | ✅ | Softmax conditional likelihood |

## 4. Time Series (statsmodels.tsa)

### 4.1 Modelos

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| ARIMA/SARIMAX | `ARIMA`, `SARIMAX` | `ARIMA` | 🔶 | Basico com seasonal; faltam exog completo, simulate, predict com IC |
| AutoReg | `AutoReg` | `AutoReg` | ✅ | |
| ARDL | `ARDL` | `ARDL` | ✅ | |
| ExponentialSmoothing | `ExponentialSmoothing` | `ExponentialSmoothing` | ✅ | Holt-Winters (SES, Holt, HW additive/multiplicative) |
| ETSModel | `ETSModel` | — | ❌ | Error-Trend-Seasonality framework completo |
| VAR | `VAR` | `VAR` | ✅ | IRF, FEVD |
| VARMAX | `VARMAX` | `VARMA` | 🔶 | Falta exog (o X do VARMAX) |
| VECM | `VECM` | `VECM` | ✅ | |
| SVAR | `SVAR` | `SVAR` | ✅ | VAR Estrutural |
| DynamicFactor | `DynamicFactor` | — | ❌ | |
| UnobservedComponents | `UnobservedComponents` | — | ❌ | Local level, tendencia, sazonalidade |
| MarkovRegression | `MarkovRegression` | `MarkovSwitching` | ✅ | Regime switching |
| MarkovAutoregression | `MarkovAutoregression` | — | ❌ | Markov com componente AR |

### 4.2 Volatility Models (arch package)

| Feature | arch | Greeners | Status | Notas |
|---|---|---|---|---|
| GARCH(p,q) | `arch_model(vol='GARCH')` | `GARCH` | ✅ | Normal + Student-t, BFGS optimizer |
| ARCH(q) | `arch_model(vol='ARCH')` | `GARCH` (p=0) | ✅ | Caso especial de GARCH(0,q) |
| EGARCH | `arch_model(vol='EGARCH')` | `EGARCH` | ✅ | Normal + Student-t, log-variance |
| GJR-GARCH | `arch_model(vol='GARCH', o=1)` | `GJRGARCH` | ✅ | Normal + Student-t, leverage effect |
| FIGARCH | `arch_model(vol='FIGARCH')` | — | ❌ | Fractionally integrated GARCH |
| APARCH | `arch_model(vol='APARCH')` | — | ❌ | Asymmetric Power ARCH |
| HARCH | `arch_model(vol='HARCH')` | — | ❌ | Heterogeneous ARCH |
| ConstantVariance | `ConstantVariance` | — | ❌ | Baseline/benchmark |
| SkewStudent | `SkewStudent` | — | ❌ | Distribuicao skew-t |
| GED | `GeneralizedError` | — | ❌ | Distribuicao generalizada de erros |

### 4.3 State Space Framework

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| MLEModel | `MLEModel` | — | ❌ | Framework generico (Kalman filter) — base para DynamicFactor, UC |
| KalmanFilter | `KalmanFilter` | `KalmanFilter` | ✅ | Filtro de Kalman basico |
| KalmanSmoother | `KalmanSmoother` | `KalmanSmoother` | ✅ | RTS smoother |
| StateSpaceModel | `MLEModel` | `StateSpaceModel` | ✅ | Estimacao via MLE |
| Simulation smoother | `SimulationSmoother` | — | ❌ | |

### 4.4 Testes e Ferramentas de Series Temporais

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| ADF | `adfuller` | `TimeSeries::adf` | ✅ | |
| KPSS | `kpss` | `TimeSeries::kpss` | ✅ | |
| Phillips-Perron | `PhillipsPerron` | `TimeSeries::phillips_perron` | ✅ | Z(alpha), Z(t) with Newey-West |
| Zivot-Andrews | `zivot_andrews` | `TimeSeries::zivot_andrews` | ✅ | Structural break unit root |
| Engle-Granger coint | `coint` | `TimeSeries::engle_granger` | ✅ | |
| Johansen coint | `coint_johansen` | `TimeSeries::johansen` | ✅ | |
| Granger causality | `grangercausalitytests` | `TimeSeries::granger_causality` | ✅ | |
| Ljung-Box | `acorr_ljungbox` | `TimeSeries::ljung_box` | ✅ | |
| ACF | `acf` | `TimeSeries::acf` | ✅ | |
| PACF | `pacf` | `TimeSeries::pacf` | ✅ | |
| seasonal_decompose | `seasonal_decompose` | `Decomposition` | ✅ | Additive + multiplicative |
| STL | `STL` | `TimeSeries::stl` | ✅ | LOESS-based decomposition |
| MSTL | `MSTL` | — | ❌ | Multi-seasonal STL |
| HP filter | `hpfilter` | `TimeSeries::hp_filter` | ✅ | Hodrick-Prescott |
| BK filter | `bkfilter` | `TimeSeries::bk_filter` | ✅ | Baxter-King |
| CF filter | `cffilter` | `TimeSeries::cf_filter` | ✅ | Christiano-Fitzgerald |
| IRF | `irf` (no VAR result) | `VarResult::irf` | ✅ | Impulse Response Functions |
| FEVD | `fevd` (no VAR result) | `VarResult::fevd` | ✅ | Forecast Error Variance Decomposition |
| DeterministicProcess | `DeterministicProcess` | `TimeSeries::deterministic_process` | ✅ | Const, trend, seasonal, fourier |
| lagmat | `lagmat` | `TimeSeries::lagmat` | ✅ | Lag matrix construction |

## 5. Robust Regression (statsmodels.robust)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| RLM | `sm.RLM` | `RLM` | ✅ | Huber, Bisquare, Andrews, Hampel |
| Norms (Huber, Tukey, etc.) | `norms` | `RobustNorm` | ✅ | 4 norms implemented |

## 6. Mixed/Multilevel Models

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| MixedLM | `MixedLM` | `MixedLM` | ✅ | Random intercepts, REML |
| BayesMixedGLM | `BayesMixedGLM` | `BayesMixedGLM` | ✅ | Laplace approximation |
| BetaModel | `BetaModel` | `BetaModel` | ✅ | Beta regression |

## 7. Multivariate (statsmodels.multivariate)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| PCA | `PCA` | `PCA` | ✅ | Eigendecomposition, scree, loadings |
| Factor Analysis | `Factor` | `FactorAnalysis` | ✅ | Principal axis factoring, rotation |
| MANOVA | `MANOVA` | `MANOVA` | ✅ | Wilks, Pillai, Hotelling, Roy |
| Canonical Correlation | `CanCorr` | `CanCorr` | ✅ | SVD-based, Wilks' Lambda, F-test |

## 8. Nonparametric (statsmodels.nonparametric)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| KDEUnivariate | `KDEUnivariate` | `KDEUnivariate` | ✅ | Gaussian, Epanechnikov, Silverman/Scott |
| KDEMultivariate | `KDEMultivariate` | `KDEMultivariate` | ✅ | Product kernel, Silverman per-dim |
| KernelReg | `KernelReg` | `KernelReg` | ✅ | Nadaraya-Watson |
| Lowess | `lowess` | `Lowess` | ✅ | Local weighted regression |

## 9. Duration/Survival (statsmodels.duration)

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| Kaplan-Meier | `SurvfuncRight` | `KaplanMeier` | ✅ | Survival function, CI |
| Cox PH | `PHReg` | `CoxPH` | ✅ | Proportional hazards |

## 10. Imputation (statsmodels.imputation)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| MICE | `MICE` | `MICE` | ✅ | Chained equations with OLS |
| BayesGaussMI | `BayesGaussMI` | `BayesGaussMI` | ✅ | Conditional normal imputation |

## 11. Diagnostics & Specification Tests

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| Jarque-Bera | `jarque_bera` | `Diagnostics::jarque_bera` | ✅ |
| Breusch-Pagan | `het_breuschpagan` | `Diagnostics::breusch_pagan` | ✅ |
| Durbin-Watson | `durbin_watson` | `Diagnostics::durbin_watson` | ✅ |
| VIF | `variance_inflation_factor` | `Diagnostics::vif` | ✅ |
| Condition Number | `np.linalg.cond` | `Diagnostics::condition_number` | ✅ |
| Leverage | `OLSInfluence` | `Diagnostics::leverage` | ✅ |
| Cook's D | `OLSInfluence` | `Diagnostics::cooks_distance` | ✅ |
| White test | `het_white` | `SpecificationTests::white_test` | ✅ |
| RESET | `linear_reset` | `SpecificationTests::reset_test` | ✅ |
| Breusch-Godfrey | `acorr_breusch_godfrey` | `SpecificationTests::breusch_godfrey` | ✅ |
| Goldfeld-Quandt | `het_goldfeldquandt` | `SpecificationTests::goldfeld_quandt` | ✅ |
| AIC/BIC | `.aic`, `.bic` | `ModelSelection` | ✅ |
| Ljung-Box | `acorr_ljungbox` | `TimeSeries::ljung_box` | ✅ |
| ARCH test | `het_arch` | `TimeSeries::arch_test` | ✅ |
| Omnibus | `omni_normtest` | `Diagnostics::omnibus` | ✅ |
| Harvey-Collier | `linear_harvey_collier` | `Diagnostics::harvey_collier` | ✅ | t-test on recursive residuals |
| DFBetas | `OLSInfluence` | `Influence::dfbetas` | ✅ | Per-observation influence |
| DFFITS | `OLSInfluence` | `Influence::dffits` | ✅ | Per-observation influence |
| CUSUM | `OLSInfluence` | `CUSUMTest` | ✅ | Recursive CUSUM + bounds |
| Wald test | `.wald_test()` | `OlsResult::wald_test` | ✅ |
| F test | `.f_test()` | `OlsResult::f_test` | ✅ |
| t test (restricoes) | `.t_test()` | `OlsResult::t_test` | ✅ |

## 12. Results & Output

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| `.summary()` | completo | `Display` trait | 🔶 | Falta summary2, LaTeX, HTML |
| `.predict()` | com CI | `predict()` | ✅ | IC via get_prediction |
| `.get_prediction()` | com SE e IC | `get_prediction()` | ✅ | OLS, GLM, Poisson, NegBin |
| `.conf_int()` | em todos os modelos | `conf_int()` | ✅ | OLS, GLM, ARIMA, Poisson, NegBin |
| `summary_col` | tabela comparativa | `SummaryCol` | ✅ | Side-by-side model comparison |
| Export LaTeX | `summary().as_latex()` | — | ❌ | |
| Export HTML | `summary().as_html()` | — | ❌ | |
| Export CSV | `summary().as_csv()` | — | ❌ | |

## 13. Formula System

| Feature | patsy/formulaic | Greeners | Status |
|---|---|---|---|
| `y ~ x1 + x2` | patsy | `Formula` | ✅ |
| `C(var)` categoricals | patsy | `C()` | ✅ |
| `I(x^2)` transforms | patsy | `I()` | ✅ |
| Interactions `x1*x2` | patsy | `*` | ✅ |
| `x1:x2` (sem main) | patsy | — | ❌ |
| `poly(x, 3)` | patsy | — | ❌ |
| `bs(x)` B-splines | patsy | — | ❌ |
| `np.log(x)` transforms | patsy | — | ❌ |

## 14. Stats & Distributions (statsmodels.stats)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| `DescrStatsW` | descriptive stats com pesos | `DescrStatsW` | ✅ | Weighted mean, var, std, CI, t-test |
| `CompareMeans` | testes de medias | `Stats::compare_means` | ✅ | Welch t-test, Cohen's d, CI |
| `anova_lm` | ANOVA | `Stats::anova_oneway` | ✅ | One-way ANOVA + regression ANOVA |
| `multipletests` | correcao de multiplos testes | — | ❌ |
| `diagnostic` (varios) | Anderson-Darling, Lilliefors, etc. | `Diagnostics` | ✅ | Anderson-Darling, Lilliefors |
| `proportion` | testes de proporcao | — | ❌ |
| `weightstats` | estatisticas ponderadas | `DescrStatsW` | ✅ | Via descrstatsw module |
| `moment_helpers` | skew, kurtosis | — | ❌ |
| `sandwich_covariance` | HAC, kernel covariance | parcial | 🔶 | NeweyWest existe em CovarianceType |
| `stattools` | varios testes | parcial | 🔶 | ADF, KPSS, PP, ZA |

## 15. Datasets (statsmodels.datasets)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| Datasets embutidos | ~30 datasets | `Datasets` | ✅ | Longley, Macrodata, Sunspots, etc. |

---

## Funcionalidades do Greeners SEM equivalente no statsmodels

Estas sao vantagens competitivas do Greeners (via linearmodels ou proprias):

| Feature | Greeners | statsmodels |
|---|---|---|
| Panel Fixed Effects | `FixedEffects` | ❌ (usa linearmodels) |
| Panel Random Effects | `RandomEffects` | ❌ |
| Between Estimator | `BetweenEstimator` | ❌ |
| Arellano-Bond | `ArellanoBond` | ❌ |
| Panel Threshold | `PanelThreshold` | ❌ |
| Hausman Test | `HausmanTest` | ❌ |
| SUR | `SUR` | ❌ (parcial em statsmodels) |
| 3SLS | `ThreeSLS` | ❌ |
| IV/2SLS | `IV` | ❌ (usa linearmodels) |
| DiD | `DiffInDiff` | ❌ |
| GMM | `GMM` | ❌ (usa gmm package) |
| Bootstrap | `Bootstrap` | ❌ (manual) |
| Binary auto-detection | `Column` | ❌ |
| Type safety (Rust) | nativo | ❌ |

---

## O que falta (❌) — Prioridades

### Alta prioridade
1. **ARIMA melhorias** — exog completo, simulate, predict com IC (🔶 → ✅)
2. **VARMAX** — adicionar exog ao VARMA (🔶 → ✅)
3. **Export** LaTeX / HTML / CSV de resultados
4. **Formulas avancadas** — `x1:x2`, `poly()`, `bs()`, `log()`

### Media prioridade
5. **ETSModel** — Error-Trend-Seasonality framework
6. **DynamicFactor** — Dynamic Factor models
7. **UnobservedComponents** — Local level, trend, seasonal
8. **MarkovAutoregression** — Markov com AR
9. **MSTL** — Multi-seasonal STL
10. **multipletests** — Bonferroni, FDR, Holm
11. **proportion** — testes de proporcao

### Baixa prioridade (nice-to-have)
12. **MLEModel generico** — Framework state space unificado
13. **SimulationSmoother**
14. **moment_helpers** — skew, kurtosis centralizados
15. **FIGARCH, APARCH, HARCH** — variantes GARCH adicionais
16. **SkewStudent, GED** — distribuicoes adicionais para GARCH
17. **summary2** — LaTeX/HTML summaries avancados
