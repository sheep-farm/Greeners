//! greeners-imputation crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use imputation::{BayesGaussMI, BayesGaussMIResult, MICEResult, MICE};
pub use mice::{MiceChained, MiceResult};

pub mod imputation;
pub mod mice;
