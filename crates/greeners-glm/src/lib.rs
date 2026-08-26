//! greeners-glm crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use beta_model::{BetaLink, BetaModel, BetaResult};
pub use conditional::{
    ConditionalLogit, ConditionalMNLogit, ConditionalPoisson, ConditionalResult,
};
pub use discrete::{BinaryModelResult, Logit, Probit};
pub use gee::{CorrStructure, GeeResult, NominalGEE, OrdinalGEE, GEE};
pub use glm::{Family, GlmResult, Link, GLM};
pub use glmgam::{GLMGam, GamResult};
pub use mnlogit::{MNLogit, MNLogitResult};
pub use negbin::{GenPoisson, GenPoissonResult, NegBin, NegBinP, NegBinPResult, NegBinResult};
pub use ordered::{OrderedLogit, OrderedProbit, OrderedResult};
pub use poisson::{Poisson, PoissonResult};
pub use zero_inflated::{ZeroInflatedResult, ZINB, ZIP};

pub mod beta_model;
pub mod conditional;
pub mod discrete;
pub mod gee;
pub mod glm;
pub mod glmgam;
pub mod mnlogit;
pub mod negbin;
pub mod ordered;
pub mod poisson;
pub mod zero_inflated;
