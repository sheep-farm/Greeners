pub mod diagnostics;
pub mod did;
pub mod discrete;
pub mod error;
pub mod gmm;
pub mod iv;
pub mod ols;
pub mod panel;
pub mod timeseries;
pub mod gls;
pub mod three_sls;
pub mod sur;
pub mod quantile;
pub mod hausman;
pub mod dynamic_panel;
pub mod threshold;
pub mod var;
pub mod varma;
pub mod vecm;
pub mod formula;
pub mod dataframe;
pub mod bootstrap;
pub mod model_selection;

pub use diagnostics::Diagnostics;
pub use did::DiffInDiff;
pub use discrete::{Logit, Probit};
pub use error::GreenersError;
pub use gmm::GMM;
pub use iv::IV;
pub use ols::OLS;
pub use panel::FixedEffects;
pub use panel::RandomEffects;
pub use panel::BetweenEstimator;
pub use timeseries::TimeSeries;
pub use gls::FGLS;
pub use three_sls::{ThreeSLS, Equation};
pub use sur::{SUR, SurEquation};
pub use quantile::QuantileReg;
pub use hausman::HausmanTest;
pub use dynamic_panel::ArellanoBond;
pub use threshold::PanelThreshold;
pub use var::VAR;
pub use varma::VARMA;
pub use vecm::VECM;
pub use formula::Formula;
pub use dataframe::DataFrame;
pub use bootstrap::{Bootstrap, HypothesisTest};
pub use model_selection::{ModelSelection, PanelDiagnostics, SummaryStats};

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
