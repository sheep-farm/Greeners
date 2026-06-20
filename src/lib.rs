// FFI exports for Odre plugin system (enabled with --features odre-ffi)
#[cfg(feature = "odre-ffi")]
pub mod odre_ffi;

pub mod distributions;
pub mod arima;
pub mod autoreg;
pub mod beta_model;
pub mod bootstrap;
pub mod column;
pub mod conditional;
pub mod dataframe;
pub mod datasets;
pub mod decomposition;
pub mod descrstatsw;
pub mod diagnostics;
pub mod did;
pub mod discrete;
pub mod dynamic_factor;
pub mod dynamic_panel;
pub mod error;
pub mod ets;
pub mod export;
pub mod formula;
pub mod garch;
pub mod gee;
pub mod glm;
pub mod glmgam;
pub mod gls;
pub mod glsar;
pub mod gmm;
pub mod hausman;
pub mod imputation;
pub mod influence;
pub mod iv;
pub mod linalg;
pub mod markov;
pub mod markov_autoreg;
pub mod mixed;
pub mod mnlogit;
pub mod model_selection;
pub mod mstl;
pub mod multipletests;
pub mod multivariate;
pub mod negbin;
pub mod nonparametric;
pub mod ols;
pub mod ordered;
pub mod panel;
pub mod poisson;
pub mod proportion;
pub mod quantile;
pub mod rlm;
pub mod rolling;
pub mod specification_tests;
pub mod statespace;
pub mod stats;
pub mod summary_col;
pub mod sur;
pub mod survival;
pub mod svar;
pub mod three_sls;
pub mod threshold;
pub mod timeseries;
pub mod unobserved_components;
pub mod var;
pub mod varma;
pub mod vecm;
pub mod wls;
pub mod zero_inflated;

pub use distributions::{chi2_pvalue, logistic, norm_pdf, t_pvalue_two, t_quantile};
pub use arima::{ArimaOrder, ArimaResult, SeasonalOrder, ARIMA};
pub use autoreg::{ARDLResult, AutoReg, AutoRegResult, ARDL};
pub use beta_model::{BetaLink, BetaModel, BetaResult};
pub use bootstrap::{Bootstrap, HypothesisTest};
pub use column::{CategoricalColumn, Column, DataType};
pub use conditional::{
    ConditionalLogit, ConditionalMNLogit, ConditionalPoisson, ConditionalResult,
};
pub use dataframe::DataFrame;
pub use datasets::Datasets;
pub use decomposition::{Decomposition, DecompositionResult};
pub use descrstatsw::DescrStatsW;
pub use diagnostics::{AndersonDarlingResult, Diagnostics};
pub use did::DiffInDiff;
pub use discrete::{Logit, Probit};
pub use dynamic_factor::{DynamicFactor, DynamicFactorResult};
pub use dynamic_panel::ArellanoBond;
pub use error::GreenersError;
pub use ets::{ETSResult, ExponentialSmoothing};
pub use export::{ExportData, ExportableResult};
pub use formula::Formula;
pub use garch::{GarchDist, GarchModelType, GarchResult, EGARCH, GARCH, GJRGARCH};
pub use gee::{CorrStructure, GeeResult, NominalGEE, OrdinalGEE, GEE};
pub use glm::{Family, GlmResult, Link, GLM};
pub use glmgam::{BSplineBasis, GLMGam, GamResult};
pub use gls::FGLS;
pub use glsar::{GlsarResult, GLSAR};
pub use gmm::GMM;
pub use hausman::HausmanTest;
pub use imputation::{BayesGaussMI, BayesGaussMIResult, MICEResult, MICE};
pub use influence::{CUSUMResult, CUSUMTest, Influence, InfluenceResult};
pub use iv::IV;
pub use markov::{MarkovSwitching, MarkovSwitchingResult};
pub use markov_autoreg::{MarkovAutoregResult, MarkovAutoregression};
pub use mixed::{BayesMixedGLM, BayesMixedGLMResult, MixedLM, MixedResult};
pub use mnlogit::{MNLogit, MNLogitResult};
pub use model_selection::{ModelSelection, PanelDiagnostics, SummaryStats};
pub use mstl::{MSTLResult, MSTL};
pub use multipletests::{MultiTestMethod, MultipleTests};
pub use multivariate::{
    CanCorr, CanCorrResult, FactorAnalysis, FactorResult, ManovaResult, PCAResult, Rotation,
    MANOVA, PCA,
};
pub use negbin::{GenPoisson, GenPoissonResult, NegBin, NegBinP, NegBinPResult, NegBinResult};
pub use nonparametric::{
    KDEMultivariate, KDEMultivariateResult, KDEResult, KDEUnivariate, Kernel, KernelReg,
    KernelRegResult, Lowess, LowessResult,
};
pub use ols::{OlsResult, PredictionResult, OLS};
pub use ordered::{OrderedLogit, OrderedProbit, OrderedResult};
pub use panel::BetweenEstimator;
pub use panel::FixedEffects;
pub use panel::RandomEffects;
pub use poisson::{Poisson, PoissonResult};
pub use proportion::ProportionTests;
pub use quantile::QuantileReg;
pub use rlm::{RlmResult, RobustNorm, RLM};
pub use rolling::{RecursiveLS, RecursiveLSResult, RollingOLS, RollingResult, RollingWLS};
pub use specification_tests::SpecificationTests;
pub use statespace::{
    state_space_estimate, KalmanFilter, KalmanResult, KalmanSmoother, SmoothedResult,
    StateSpaceModel, StateSpaceResult,
};
pub use stats::{AnovaRegressionResult, AnovaResult, CompareMeansResult, Stats};
pub use summary_col::{ModelSummary, SummaryCol, SummaryColResult};
pub use sur::{SurEquation, SUR};
pub use survival::{CoxPH, CoxResult, KMResult, KaplanMeier};
pub use svar::{SVarIdentification, SVarResult, SVAR};
pub use three_sls::{Equation, ThreeSLS};
pub use threshold::PanelThreshold;
pub use timeseries::{PhillipsPerronResult, TimeSeries, ZivotAndrewsResult};
pub use unobserved_components::{UCLevel, UCResult, UCSeasonal, UnobservedComponents};
pub use var::VAR;
pub use varma::VARMA;
pub use vecm::VECM;
pub use wls::WLS;
pub use zero_inflated::{ZeroInflatedResult, ZINB, ZIP};

#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceType {
    /// Standard OLS (Homoscedastic)
    NonRobust,
    /// White's Robust Errors (HC1) - Only Heteroscedasticity
    /// Uses small-sample correction: n/(n-k)
    HC1,
    /// HC2 - Leverage-adjusted heteroscedasticity-robust SE
    /// Adjusts for leverage: σ²_i / (1 - h_i)
    /// More efficient than HC1 with small samples
    HC2,
    /// HC3 - Jackknife heteroscedasticity-robust SE
    /// Uses squared leverage adjustment: σ²_i / (1 - h_i)²
    /// Most robust for small samples (MacKinnon & White, 1985)
    /// Recommended default robust SE estimator
    HC3,
    /// HC4 - Refined jackknife (Cribari-Neto, 2004)
    /// Uses power adjustment: σ²_i / (1 - h_i)^δᵢ where δᵢ = min(4, n·h_i/k)
    /// Best small-sample performance, especially with influential observations
    /// More refined than HC3 for datasets with high-leverage points
    HC4,
    /// Newey-West (HAC) - Heteroscedasticity + Autocorrelation
    /// The 'usize' parameter is the number of lags (L).
    /// Common rule of thumb: L = n^0.25
    NeweyWest(usize),
    /// Clustered Standard Errors (One-Way)
    /// Critical for panel data, experiments, and grouped observations
    /// The Vec<usize> contains cluster IDs for each observation
    Clustered(Vec<usize>),
    /// Two-Way Clustered Standard Errors (Cameron-Gelbach-Miller, 2011)
    /// For panel data with clustering along two dimensions (e.g., firm + time)
    /// First Vec: cluster IDs for dimension 1 (e.g., firm IDs)
    /// Second Vec: cluster IDs for dimension 2 (e.g., time periods)
    /// Formula: V = V₁ + V₂ - V₁₂ where V₁₂ is intersection clustering
    /// Essential for panel data with both cross-sectional and time correlation
    ClusteredTwoWay(Vec<usize>, Vec<usize>),
}

/// Inference distribution type for hypothesis testing and confidence intervals
///
/// Determines which statistical distribution to use when computing p-values
/// and confidence intervals for coefficient estimates.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum InferenceType {
    /// Student's t-distribution (default for finite samples)
    ///
    /// Uses t(df) distribution for hypothesis testing and confidence intervals.
    /// This is the exact finite-sample distribution under normality assumptions.
    ///
    /// **Recommended for:**
    /// - Small to medium samples (n < 100)
    /// - When exact finite-sample inference is desired
    /// - Conservative hypothesis testing
    ///
    /// **Used by:** OLS, IV/2SLS, Panel models (default)
    #[default]
    StudentT,

    /// Standard Normal distribution (z-distribution)
    ///
    /// Uses N(0,1) distribution for hypothesis testing and confidence intervals.
    /// This is the asymptotic distribution (as n → ∞).
    ///
    /// **Recommended for:**
    /// - Large samples (n > 1000)
    /// - Asymptotic theory contexts (MLE, GMM)
    /// - Compatibility with statsmodels/Python
    ///
    /// **Used by:** Logit, Probit, GMM, Quantile Regression (always)
    ///
    /// **Note:** For large samples (df > 30), t and z distributions are nearly identical.
    Normal,
}
