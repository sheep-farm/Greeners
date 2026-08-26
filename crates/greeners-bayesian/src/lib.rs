//! greeners-bayesian crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use bayesian_linear::{BayesianLinear, BayesianLinearResult};
pub use bayesian_sc::{BayesianSC, BayesianScResult};
pub use bayesian_sfa::{BayesianSFA, BayesianSfaResult};
pub use bvar::{BvarResult, BVAR};
pub use favar::{FavarResult, FAVAR};
pub use mfvar::{MfVarResult, MFVAR};
pub use mixed::{BayesMixedGLM, BayesMixedGLMResult, MixedLM, MixedResult};

pub mod bayesian_linear;
pub mod bayesian_sc;
pub mod bayesian_sfa;
pub mod bvar;
pub mod favar;
pub mod mfvar;
pub mod mixed;
