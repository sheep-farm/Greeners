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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovarianceType {
    /// Standard OLS (Homoscedastic)
    NonRobust,
    /// White's Robust Errors (HC1) - Only Heteroscedasticity
    HC1,
    /// Newey-West (HAC) - Heteroscedasticity + Autocorrelation
    /// The 'usize' parameter is the number of lags (L).
    /// Common rule of thumb: L = n^0.25
    NeweyWest(usize),
}
