pub mod diagnostics;
pub mod did;
pub mod discrete;
pub mod error;
pub mod gmm;
pub mod iv;
pub mod ols;
pub mod panel;
pub mod timeseries;

pub use diagnostics::Diagnostics;
pub use did::DiffInDiff;
pub use discrete::{Logit, Probit};
pub use error::GreenersError;
pub use gmm::GMM;
pub use iv::IV;
pub use ols::OLS;
pub use panel::FixedEffects;
pub use timeseries::TimeSeries;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovarianceType {
    /// OLS Padrão (Homocedástico)
    NonRobust,
    /// White's Robust Errors (HC1) - Apenas Heterocedasticidade
    HC1,
    /// Newey-West (HAC) - Heterocedasticidade + Autocorrelação
    /// O parâmetro 'usize' é o número de lags (L).
    /// Regra de bolso comum: L = n^0.25
    NeweyWest(usize),
}
