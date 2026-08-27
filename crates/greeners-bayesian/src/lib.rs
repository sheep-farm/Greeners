//! greeners-bayesian crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use bayesian_linear::{BayesianLinear, BayesianLinearResult};
#[cfg(feature = "experimental")]
pub use bayesian_sc::{BayesianSC, BayesianScResult};
#[cfg(feature = "experimental")]
pub use bayesian_sfa::{BayesianSFA, BayesianSfaResult};
#[cfg(feature = "experimental")]
pub use bvar::{BvarResult, BVAR};
pub use favar::{FavarResult, FAVAR};
#[cfg(feature = "experimental")]
pub use mfvar::{MfVarResult, MFVAR};
pub use mixed::{BayesMixedGLM, BayesMixedGLMResult, MixedLM, MixedResult};

pub mod bayesian_linear;
#[cfg(feature = "experimental")]
pub mod bayesian_sc;
#[cfg(feature = "experimental")]
pub mod bayesian_sfa;
#[cfg(feature = "experimental")]
pub mod bvar;
pub mod favar;
#[cfg(feature = "experimental")]
pub mod mfvar;
pub mod mixed;
