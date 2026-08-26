//! greeners-survival crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use survival::{CoxPH, CoxResult, KMResult, KaplanMeier};

pub mod survival;
