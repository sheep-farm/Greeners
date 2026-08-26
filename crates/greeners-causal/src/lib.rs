//! greeners-causal crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use causal_forest::{CausalForest, CausalForestResult};
pub use causal_impact::{CausalImpact, CausalImpactResult};
pub use conformal::{ConformalPrediction, ConformalResult};
pub use cuped::{CupedResult, CUPED};
pub use did::{DidResult, DiffInDiff};
pub use dml_crossfit::{DmlResult, DML};
pub use double_ml::{DoubleML, DoubleMLResult};
pub use dr_learner::{DRLearner, DrLearnerResult};
pub use lp_did::{LpDid, LpDidResult};
pub use psm::{BalanceRow, PsmResult, PSM};
pub use rd::{RdKernel, RdResult, RD};
pub use synth::{SynthResult, SyntheticControl};
pub use synth_did::{SyntheticDiD, SyntheticDidResult};
pub use tmle::{TmleResult, TMLE};

pub mod causal_forest;
pub mod causal_impact;
pub mod conformal;
pub mod cuped;
pub mod did;
pub mod dml_crossfit;
pub mod double_ml;
pub mod dr_learner;
pub mod lp_did;
pub mod psm;
pub mod rd;
pub mod synth;
pub mod synth_did;
pub mod tmle;
