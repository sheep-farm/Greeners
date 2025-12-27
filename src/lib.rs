pub mod bootstrap;
pub mod dataframe;
pub mod diagnostics;
pub mod did;
pub mod discrete;
pub mod dynamic_panel;
pub mod error;
pub mod formula;
pub mod gls;
pub mod gmm;
pub mod hausman;
pub mod iv;
pub mod model_selection;
pub mod ols;
pub mod panel;
pub mod quantile;
pub mod specification_tests;
pub mod sur;
pub mod three_sls;
pub mod threshold;
pub mod timeseries;
pub mod var;
pub mod varma;
pub mod vecm;

pub use bootstrap::{Bootstrap, HypothesisTest};
pub use dataframe::DataFrame;
pub use diagnostics::Diagnostics;
pub use did::DiffInDiff;
pub use discrete::{Logit, Probit};
pub use dynamic_panel::ArellanoBond;
pub use error::GreenersError;
pub use formula::Formula;
pub use gls::FGLS;
pub use gmm::GMM;
pub use hausman::HausmanTest;
pub use iv::IV;
pub use model_selection::{ModelSelection, PanelDiagnostics, SummaryStats};
pub use ols::OLS;
pub use panel::BetweenEstimator;
pub use panel::FixedEffects;
pub use panel::RandomEffects;
pub use quantile::QuantileReg;
pub use specification_tests::SpecificationTests;
pub use sur::{SurEquation, SUR};
pub use three_sls::{Equation, ThreeSLS};
pub use threshold::PanelThreshold;
pub use timeseries::TimeSeries;
pub use var::VAR;
pub use varma::VARMA;
pub use vecm::VECM;

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
