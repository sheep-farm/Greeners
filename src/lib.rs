// FFI exports for Odre plugin system (enabled with --features odre-ffi)
#[cfg(feature = "odre-ffi")]
pub mod odre_ffi;

pub mod arima;
pub mod autoreg;
pub mod bart;
pub mod bayesian_linear;
pub mod bayesian_sc;
pub mod bayesian_sfa;
pub mod beta_model;
pub mod binary_diagnostics;
pub mod biplot;
pub mod bootstrap;
pub mod bvar;
pub mod causal_forest;
pub mod causal_impact;
pub mod column;
pub mod conditional;
pub mod conformal;
pub mod copula;
pub mod cuped;
pub mod dataframe;
pub mod datasets;
pub mod dbscan;
pub mod dcc_garch;
pub mod decomposition;
pub mod descrstatsw;
pub mod dfm;
pub mod diagnostics;
pub mod did;
pub mod discrete;
pub mod distributions;
pub mod dml_crossfit;
pub mod double_ml;
pub mod dr_learner;
pub mod dynamic_factor;
pub mod dynamic_panel;
pub mod error;
pub mod ets;
pub mod event_study;
pub mod export;
pub mod fa_panel;
pub mod fama_macbeth;
pub mod favar;
pub mod fmols;
pub mod formula;
pub mod functional_coef;
pub mod garch;
pub mod gee;
pub mod glm;
pub mod glmgam;
pub mod gls;
pub mod glsar;
pub mod gmm;
pub mod gmm_clustering;
pub mod gp;
pub mod gradient_boosting;
pub mod grf;
pub mod hausman;
pub mod hawkes;
pub mod heckman;
pub mod hierarchical;
pub mod imputation;
pub mod influence;
pub mod isotonic;
pub mod iv;
pub mod johansen_break;
pub mod kmeans;
pub mod linalg;
pub mod lp_did;
pub mod lstm;
pub mod margins;
pub mod markov;
pub mod markov_autoreg;
pub mod mfvar;
pub mod mice;
pub mod midas;
pub mod mixed;
pub mod mlp;
pub mod mnlogit;
pub mod model_selection;
pub mod moment_helpers;
pub mod ms_var;
pub mod mstl;
pub mod multipletests;
pub mod multivariate;
pub mod nardl;
pub mod negbin;
pub mod nls;
pub mod nonparametric;
pub mod ols;
pub mod ordered;
pub mod orthogonal_forest;
pub mod panel;
pub mod panel_heckman;
pub mod panel_quantile;
pub mod panel_robust;
pub mod panel_tobit;
pub mod panel_var;
pub mod poisson;
pub mod proportion;
pub mod psm;
pub mod pstr;
pub mod qrf;
pub mod qrf_inference;
pub mod quantile;
pub mod quantile_var;
pub mod random_forest;
pub mod rd;
pub mod reg_path;
pub mod rlm;
pub mod rolling;
pub mod setar;
pub mod spatial;
pub mod spatial_durbin;
pub mod spatial_durbin_error;
pub mod spatial_panel;
pub mod specification_tests;
pub mod spectral;
pub mod statespace;
pub mod stats;
pub mod stochastic_frontier;
pub mod summary_col;
pub mod sur;
pub mod survival;
pub mod sv;
pub mod svar;
pub mod synth;
pub mod synth_did;
pub mod three_sls;
pub mod threshold;
pub mod timeseries;
pub mod tmle;
pub mod tobit;
pub mod transformer;
pub mod transforms;
pub mod tsne;
pub mod tv_copula;
pub mod tvar;
pub mod tvp;
pub mod tvp_var;
pub mod umap;
pub mod unobserved_components;
pub mod var;
pub mod varma;
pub mod vecm;
pub mod wavelet;
pub mod wls;
pub mod xgboost;
pub mod zero_inflated;

pub use arima::{ArimaOrder, ArimaResult, SeasonalOrder, ARIMA};
pub use autoreg::{ARDLResult, AutoReg, AutoRegResult, ARDL};
pub use bart::{BartResult, BART};
pub use bayesian_linear::{BayesianLinear, BayesianLinearResult};
pub use bayesian_sc::{BayesianSC, BayesianScResult};
pub use bayesian_sfa::{BayesianSFA, BayesianSfaResult};
pub use beta_model::{BetaLink, BetaModel, BetaResult};
pub use binary_diagnostics::{
    BinaryDiagnostics, ClassificationResult, HosmerLemeshowResult, LinktestResult, RocResult,
};
pub use biplot::{Biplot, BiplotResult, BiplotType};
pub use bootstrap::{Bootstrap, HypothesisTest};
pub use bvar::{BvarResult, BVAR};
pub use causal_forest::{CausalForest, CausalForestResult};
pub use causal_impact::{CausalImpact, CausalImpactResult};
pub use column::{CategoricalColumn, Column, DataType};
pub use conditional::{
    ConditionalLogit, ConditionalMNLogit, ConditionalPoisson, ConditionalResult,
};
pub use conformal::{ConformalPrediction, ConformalResult};
pub use copula::{Copula, CopulaResult, CopulaType};
pub use cuped::{CupedResult, CUPED};
pub use dataframe::DataFrame;
pub use datasets::Datasets;
pub use dbscan::{DbscanResult, DBSCAN};
pub use dcc_garch::{DccGarchResult, DCCGARCH};
pub use decomposition::{Decomposition, DecompositionResult};
pub use descrstatsw::DescrStatsW;
pub use dfm::{DfmResult, DFM};
pub use diagnostics::{
    AndersonDarlingResult, ArchTestResult, Diagnostics, LjungBoxResult, ShapiroFranciaResult,
    ShapiroWilkResult,
};
pub use did::{DidResult, DiffInDiff};
pub use discrete::{Logit, Probit};
pub use distributions::{chi2_pvalue, f_pvalue, logistic, norm_pdf, t_pvalue_two, t_quantile};
pub use dml_crossfit::{DmlResult, DML as DMLCrossfit};
pub use double_ml::{DoubleML, DoubleMLResult};
pub use dr_learner::{DRLearner, DrLearnerResult};
pub use dynamic_factor::{DynamicFactor, DynamicFactorResult};
pub use dynamic_panel::{ArellanoBond, ArellanoBondResult, SystemGmm, SystemGmmResult};
pub use error::GreenersError;
pub use ets::{ETSResult, ExponentialSmoothing};
pub use event_study::{EventStudy, EventStudyResult};
pub use export::{ExportData, ExportableResult};
pub use fa_panel::{FAPanel, FaPanelResult};
pub use fama_macbeth::{FamaMacBeth, FamaMacBethResult};
pub use favar::{FavarResult, FAVAR};
pub use fmols::{FmolsResult, FMOLS};
pub use formula::Formula;
pub use functional_coef::{FunctionalCoef, FunctionalCoefResult, KernelType};
pub use garch::{GarchDist, GarchModelType, GarchResult, EGARCH, GARCH, GJRGARCH};
pub use gee::{CorrStructure, GeeResult, NominalGEE, OrdinalGEE, GEE};
pub use glm::{Family, GlmResult, Link, GLM};
pub use glmgam::{BSplineBasis, GLMGam, GamResult};
pub use gls::FGLS;
pub use glsar::{GlsarResult, GLSAR};
pub use gmm::{GmmResult, GMM};
pub use gmm_clustering::{GmmClustering, GmmResult as GmmClusteringResult};
pub use gp::{GaussianProcess, GpResult};
pub use gradient_boosting::{GradientBoosting, GradientBoostingResult};
pub use grf::{GrfResult, GRF};
pub use hausman::HausmanTest;
pub use hawkes::{Hawkes, HawkesResult};
pub use heckman::{Heckman, HeckmanResult};
pub use hierarchical::{HierarchicalClustering, HierarchicalResult, Linkage};
pub use imputation::{BayesGaussMI, BayesGaussMIResult, MICEResult, MICE};
pub use influence::{CUSUMResult, CUSUMTest, Influence, InfluenceResult};
pub use isotonic::{IsotonicRegression, IsotonicResult};
pub use iv::{EndogeneityTestResult, IvResult, SarganTestResult, IV};
pub use johansen_break::{JohansenBreak, JohansenBreakResult};
pub use kmeans::{KMeans, KmeansResult};
pub use lp_did::{LpDid, LpDidResult};
pub use lstm::{LstmResult, LSTM};
pub use margins::{MarginalEffectsResult, Margins};
pub use markov::{MarkovSwitching, MarkovSwitchingResult};
pub use markov_autoreg::{MarkovAutoregResult, MarkovAutoregression};
pub use mfvar::{MfVarResult, MFVAR};
pub use mice::{MiceChained, MiceResult};
pub use midas::{Midas, MidasResult};
pub use mixed::{BayesMixedGLM, BayesMixedGLMResult, MixedLM, MixedResult};
pub use mlp::{MlpResult, MLP};
pub use mnlogit::{MNLogit, MNLogitResult};
pub use model_selection::{LrTestResult, ModelSelection, PanelDiagnostics, SummaryStats};
pub use moment_helpers::MomentHelpers;
pub use ms_var::{MsVarResult, MSVAR};
pub use mstl::{MSTLResult, MSTL};
pub use multipletests::{MultiTestMethod, MultipleTests};
pub use multivariate::{
    CanCorr, CanCorrResult, FactorAnalysis, FactorResult, ManovaResult, PCAResult, Rotation,
    MANOVA, PCA,
};
pub use nardl::{NardlResult, NARDL};
pub use negbin::{GenPoisson, GenPoissonResult, NegBin, NegBinP, NegBinPResult, NegBinResult};
pub use nls::{
    predict_ces, predict_cobb_douglas, predict_exp, predict_logistic, predict_power, NlsResult, NLS,
};
pub use nonparametric::{
    KDEMultivariate, KDEMultivariateResult, KDEResult, KDEUnivariate, Kernel, KernelReg,
    KernelRegResult, Lowess, LowessResult,
};
pub use ols::{OlsResult, PredictionResult, OLS};
pub use ordered::{OrderedLogit, OrderedProbit, OrderedResult};
pub use orthogonal_forest::{OrfResult, OrthogonalForest};
pub use panel::BetweenEstimator;
pub use panel::FixedEffects;
pub use panel::GlsPanels;
pub use panel::PanelGLS;
pub use panel::PanelGlsResult;
pub use panel::PanelIvResult;
pub use panel::PcseResult;
pub use panel::RandomEffects;
pub use panel::FE2SLS;
pub use panel::PCSE;
pub use panel_heckman::{PanelHeckman, PanelHeckmanResult};
pub use panel_quantile::{PanelQuantile, PanelQuantileResult};
pub use panel_robust::{RobustFTest, RobustFTestResult, RobustHausman, RobustHausmanResult};
pub use panel_tobit::{PanelTobit, PanelTobitResult};
pub use panel_var::{PanelVAR, PanelVarResult};
pub use poisson::{Poisson, PoissonResult};
pub use proportion::ProportionTests;
pub use psm::{BalanceRow, PsmResult, PSM};
pub use pstr::{PstrResult, PSTR};
pub use qrf::{QrfResult, QRF};
pub use qrf_inference::{QrfInference, QrfInferenceResult};
pub use quantile::{QuantileReg, QuantileResult};
pub use quantile_var::{QuantileVAR, QuantileVarResult};
pub use random_forest::{RandomForest, RandomForestResult};
pub use rd::{RdKernel, RdResult, RD};
pub use reg_path::{RegPath, RegPathResult};
pub use rlm::{RlmResult, RobustNorm, RLM};
pub use rolling::{RecursiveLS, RecursiveLSResult, RollingOLS, RollingResult, RollingWLS};
pub use setar::{SetarResult, SETAR};
pub use spatial::{Spatial, SpatialResult};
pub use spatial_durbin::{SpatialDurbin, SpatialDurbinResult};
pub use spatial_durbin_error::{SpatialDurbinError, SpatialDurbinErrorResult};
pub use spatial_panel::{SpatialPanel, SpatialPanelResult};
pub use specification_tests::SpecificationTests;
pub use spectral::{SpectralClustering, SpectralResult};
pub use statespace::{
    state_space_estimate, KalmanFilter, KalmanResult, KalmanSmoother, LocalLevel, LocalLevelResult,
    SmoothedResult, StateSpaceModel, StateSpaceResult,
};
pub use stats::{AnovaRegressionResult, AnovaResult, CompareMeansResult, Stats, TTestResult};
pub use stochastic_frontier::{SfaResult, StochasticFrontier};
pub use summary_col::{ModelSummary, SummaryCol, SummaryColResult};
pub use sur::{SurEquation, SUR};
pub use survival::{CoxPH, CoxResult, KMResult, KaplanMeier};
pub use sv::{SvResult, SV};
pub use svar::{SVarIdentification, SVarResult, SVAR};
pub use synth::{SynthResult, SyntheticControl};
pub use synth_did::{SyntheticDiD, SyntheticDidResult};
pub use three_sls::{Equation, ThreeSLS};
pub use threshold::PanelThreshold;
pub use timeseries::{PhillipsPerronResult, TimeSeries, ZivotAndrewsResult};
pub use tmle::{TmleResult, TMLE};
pub use tobit::{Tobit, TobitResult};
pub use transformer::{Transformer, TransformerResult};
pub use transforms::Transforms;
pub use tsne::{TsneResult, TSNE};
pub use tv_copula::{TvCopula, TvCopulaResult, TvCopulaType};
pub use tvar::{TvarResult, TVAR};
pub use tvp::{TvpResult, TVP};
pub use tvp_var::{TvpVar, TvpVarResult};
pub use umap::{UmapResult, UMAP};
pub use unobserved_components::{UCLevel, UCResult, UCSeasonal, UnobservedComponents};
pub use var::VAR;
pub use varma::VARMA;
pub use vecm::VECM;
pub use wavelet::{ModwtResult, MODWT};
pub use wls::WLS;
pub use xgboost::{XGBoost, XgboostResult};
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
