//! greeners-diagnostics crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use binary_diagnostics::{
    BinaryDiagnostics, ClassificationResult, HosmerLemeshowResult, LinktestResult, RocResult,
};
pub use diagnostics::{
    AndersonDarlingResult, ArchTestResult, Diagnostics, LjungBoxResult, ShapiroFranciaResult,
    ShapiroWilkResult,
};
pub use fama_macbeth::{FamaMacBeth, FamaMacBethResult};
pub use influence::{CUSUMResult, CUSUMTest, Influence, InfluenceResult};
pub use model_selection::{LrTestResult, ModelSelection, PanelDiagnostics, SummaryStats};
pub use specification_tests::SpecificationTests;

pub mod binary_diagnostics;
pub mod diagnostics;
pub mod fama_macbeth;
pub mod influence;
pub mod model_selection;
pub mod specification_tests;
