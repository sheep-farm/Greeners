# Greeners Roadmap: Paridade com statsmodels

Este documento mapeia todas as funcionalidades do statsmodels (v0.14) e compara com o estado atual do Greeners. Serve como guia permanente para implementacoes futuras — consulte-o no inicio de cada sessao.

Legenda: ✅ Implementado | 🔶 Parcial | ❌ Ausente

**Ultima atualizacao:** 2026-02-01

---

## 1. Regression (statsmodels.regression)

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| OLS | `sm.OLS` | `OLS` | ✅ | HC1-HC4, NeweyWest, Clustered, TwoWay |
| WLS | `sm.WLS` | `WLS` | ✅ | Interface dedicada com pesos explicitos |
| GLS | `sm.GLS` | `FGLS` | ✅ | |
| GLSAR | `sm.GLSAR` | — | ❌ | GLS com erros AR |
| RecursiveLS | `RecursiveLS` | — | ❌ | Minimos quadrados recursivos |
| RollingOLS | `RollingOLS` | — | ❌ | OLS com janela movel |
| RollingWLS | `RollingWLS` | — | ❌ | WLS com janela movel |
| QuantReg | `QuantReg` | `QuantileReg` | ✅ | |

## 2. Generalized Linear Models (statsmodels.genmod)

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| GLM | `sm.GLM` | `GLM` | ✅ | Gaussian, Binomial, Poisson, Gamma, InvGaussian, Tweedie, NegBin |
| Links | 10+ | 10 | ✅ | Identity, Log, Logit, Probit, InvPower, InvSq, CLogLog, Power, NegBin, Cauchy |
| GEE | `GEE` | — | ❌ | Generalized Estimating Equations |
| NominalGEE | `NominalGEE` | — | ❌ | |
| OrdinalGEE | `OrdinalGEE` | — | ❌ | |
| GLMGam | `GLMGam` | — | ❌ | GAM via penalized splines |
| BayesMixedGLM | `BinomialBayesMixedGLM` | — | ❌ | |

## 3. Discrete Choice (statsmodels.discrete)

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| Logit | `Logit` | `Logit` | ✅ | AME + MEM |
| Probit | `Probit` | `Probit` | ✅ | AME + MEM |
| MNLogit | `MNLogit` | `MNLogit` | ✅ | Softmax, RRR, predict_proba |
| OrderedModel | `OrderedModel` | `OrderedLogit`/`OrderedProbit` | ✅ | Threshold reparametrization |
| Poisson (dedicado) | `Poisson` | `Poisson` | ✅ | Overdispersion test, exposure, AME |
| NegativeBinomial | `NegativeBinomial` | `NegBin` | ✅ | Profile likelihood alpha, LR test |
| NegativeBinomialP | `NegativeBinomialP` | — | ❌ | |
| GeneralizedPoisson | `GeneralizedPoisson` | — | ❌ | |
| ZeroInflatedPoisson | `ZeroInflatedPoisson` | `ZIP` | ✅ | EM algorithm |
| ZeroInflatedNB | `ZeroInflatedNegativeBinomialP` | `ZINB` | ✅ | EM algorithm |
| ConditionalLogit | `ConditionalLogit` | `ConditionalLogit` | ✅ | Chamberlain conditional MLE |
| ConditionalPoisson | `ConditionalPoisson` | `ConditionalPoisson` | ✅ | Hausman-Hall-Griliches |
| ConditionalMNLogit | `ConditionalMNLogit` | — | ❌ | |

## 4. Time Series (statsmodels.tsa)

### 4.1 Modelos

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| ARIMA/SARIMAX | `ARIMA`, `SARIMAX` | `ARIMA` | 🔶 | Basico com seasonal; faltam exog completo, simulate, predict com IC |
| AutoReg | `AutoReg` | — | ❌ | |
| ARDL | `ARDL` | — | ❌ | |
| ExponentialSmoothing | `ExponentialSmoothing` | — | ❌ | Holt-Winters |
| SimpleExpSmoothing | `SimpleExpSmoothing` | — | ❌ | |
| Holt | `Holt` | — | ❌ | |
| ETSModel | `ETSModel` | — | ❌ | Error-Trend-Seasonality |
| VAR | `VAR` | `VAR` | ✅ | |
| VARMAX | `VARMAX` | `VARMA` | 🔶 | Falta exog (o X do VARMAX) |
| VECM | `VECM` | `VECM` | ✅ | |
| SVAR | `SVAR` | — | ❌ | VAR Estrutural |
| DynamicFactor | `DynamicFactor` | — | ❌ | |
| UnobservedComponents | `UnobservedComponents` | — | ❌ | Local level, tendencia, sazonalidade |
| MarkovRegression | `MarkovRegression` | — | ❌ | Regime switching |
| MarkovAutoregression | `MarkovAutoregression` | — | ❌ | |

### 4.2 State Space Framework

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| MLEModel | `MLEModel` | — | ❌ | Framework generico (Kalman filter) — base para todos os modelos acima |
| KalmanFilter | `KalmanFilter` | — | ❌ | |
| KalmanSmoother | `KalmanSmoother` | — | ❌ | |
| Simulation smoother | `SimulationSmoother` | — | ❌ | |

### 4.3 Testes e Ferramentas de Series Temporais

| Feature | statsmodels | Greeners | Status | Notas |
|---|---|---|---|---|
| ADF | `adfuller` | `TimeSeries::adf` | ✅ | |
| KPSS | `kpss` | `TimeSeries::kpss` | ✅ | |
| Phillips-Perron | `PhillipsPerron` | — | ❌ | |
| Zivot-Andrews | `zivot_andrews` | — | ❌ | |
| Engle-Granger coint | `coint` | `TimeSeries::engle_granger` | ✅ | |
| Johansen coint | `coint_johansen` | `TimeSeries::johansen` | ✅ | |
| Granger causality | `grangercausalitytests` | `TimeSeries::granger_causality` | ✅ | |
| Ljung-Box | `acorr_ljungbox` | `TimeSeries::ljung_box` | ✅ | |
| ACF | `acf` | `TimeSeries::acf` | ✅ | |
| PACF | `pacf` | `TimeSeries::pacf` | ✅ | |
| seasonal_decompose | `seasonal_decompose` | — | ❌ | |
| STL | `STL` | — | ❌ | |
| MSTL | `MSTL` | — | ❌ | |
| HP filter | `hpfilter` | — | ❌ | |
| BK filter | `bkfilter` | — | ❌ | |
| CF filter | `cffilter` | — | ❌ | |
| IRF | `irf` (no VAR result) | `VarResult::irf` | ✅ | Impulse Response Functions |
| FEVD | `fevd` (no VAR result) | `VarResult::fevd` | ✅ | Forecast Error Variance Decomposition |
| DeterministicProcess | `DeterministicProcess` | — | ❌ | Trend, fourier, sazonalidade |
| lagmat | `lagmat` | — | ❌ | Construcao de matrizes de lags |

## 5. Robust Regression (statsmodels.robust)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| RLM | `sm.RLM` | — | ❌ |
| Norms (Huber, Tukey, etc.) | `norms` | — | ❌ |

## 6. Mixed/Multilevel Models

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| MixedLM | `MixedLM` | — | ❌ |
| BetaModel | `BetaModel` | — | ❌ |

## 7. Multivariate (statsmodels.multivariate)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| PCA | `PCA` | — | ❌ |
| Factor Analysis | `Factor` | — | ❌ |
| MANOVA | `MANOVA` | — | ❌ |
| Canonical Correlation | `CanCorr` | — | ❌ |

## 8. Nonparametric (statsmodels.nonparametric)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| KDEUnivariate | `KDEUnivariate` | — | ❌ |
| KDEMultivariate | `KDEMultivariate` | — | ❌ |
| KernelReg | `KernelReg` | — | ❌ |
| Lowess | `lowess` | — | ❌ |

## 9. Duration/Survival (statsmodels.duration)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| Kaplan-Meier | `SurvfuncRight` | — | ❌ |
| Cox PH | `PHReg` | — | ❌ |

## 10. Imputation (statsmodels.imputation)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| MICE | `MICE` | — | ❌ |
| BayesGaussMI | `BayesGaussMI` | — | ❌ |

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
| Harvey-Collier | `linear_harvey_collier` | — | ❌ |
| DFBetas | `OLSInfluence` | — | ❌ |
| DFFITS | `OLSInfluence` | — | ❌ |
| CUSUM | `OLSInfluence` | — | ❌ |
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
| `summary_col` | tabela comparativa | — | ❌ | Comparacao lado-a-lado |
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
| `DescrStatsW` | descriptive stats com pesos | — | ❌ |
| `CompareMeans` | testes de medias | — | ❌ |
| `anova_lm` | ANOVA | — | ❌ |
| `multipletests` | correcao de multiplos testes | — | ❌ |
| `diagnostic` (vários) | Anderson-Darling, Lilliefors, etc. | — | ❌ |
| `proportion` | testes de proporcao | — | ❌ |
| `weightstats` | estatisticas ponderadas | — | ❌ |
| `moment_helpers` | skew, kurtosis | — | ❌ |
| `sandwich_covariance` | HAC, kernel covariance | parcial | 🔶 | NeweyWest existe em CovarianceType |
| `stattools` | varios testes | parcial | 🔶 | ADF existe |

## 15. Datasets (statsmodels.datasets)

| Feature | statsmodels | Greeners | Status |
|---|---|---|---|
| Datasets embutidos | ~30 datasets | — | ❌ | Longley, Macrodata, Sunspots, etc. |

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

## Prioridades de Implementacao

### Fase 1 — Completar o core existente
Itens que adicionam valor imediato aos modelos ja implementados.

1. **`conf_int()` e `get_prediction()` em OLS/GLM** — intervalos de confianca para coeficientes e predicoes
2. **ACF / PACF** — funcoes basicas essenciais para time series
3. **KPSS test** — complemento ao ADF
4. **Ljung-Box test** — diagnostico de autocorrelacao residual
5. **IRF e FEVD** no VAR — analise de impulso-resposta ja esperada pelo usuario do VAR
6. **Cointegration tests** (Engle-Granger, Johansen)
7. **Granger causality test**
8. **WLS dedicado** com interface de pesos
9. **Links adicionais** no GLM (CLogLog, Power)
10. **Wald/F/t tests** genericos nos resultados
11. **Omnibus test**
12. **ARCH test**

### Fase 2 — Modelos discretos e contagem
1. **Poisson dedicado** (interface separada do GLM, com overdispersion test)
2. **NegativeBinomial dedicado**
3. **MNLogit** (Multinomial Logit)
4. **OrderedModel** (Ordered Logit/Probit)
5. **Zero-Inflated** (ZIP, ZINB)
6. **Conditional models** (ConditionalLogit, ConditionalPoisson)

### Fase 3 — Time series avancado
1. **ExponentialSmoothing** (Holt-Winters)
2. **State Space framework** (Kalman filter) — base para DynamicFactor, UnobservedComponents
3. **seasonal_decompose / STL**
4. **Filtros** (HP, BK, CF)
5. **Markov Switching**
6. **ARDL / AutoReg**
7. **SVAR**
8. **VARMAX** (adicionar exog ao VARMA)

### Fase 4 — Extensoes
1. **RLM** (Robust Linear Model — M-estimation)
2. **MixedLM** (Mixed Effects)
3. **Nonparametric** (KDE, Lowess, Kernel Regression)
4. **PCA / Factor Analysis / MANOVA**
5. **Survival** (Kaplan-Meier, Cox PH)
6. **GEE**
7. **MICE** (Multiple Imputation)
8. **BetaModel**
9. **RecursiveLS, RollingOLS, RollingWLS**
10. **GLSAR, GLMGam**

### Fase 5 — Infraestrutura e polish
1. **summary_col** (comparacao de modelos lado-a-lado)
2. **Export** LaTeX / HTML / CSV de resultados
3. **Formulas avancadas** (`:`, `poly()`, `bs()`, `log()`)
4. **Influence diagnostics** completos (DFBetas, DFFITS, CUSUM)
5. **Stats module** (ANOVA, testes de proporcao, multipletests)
6. **Datasets embutidos**
7. **DescrStatsW** (estatisticas descritivas com pesos)
