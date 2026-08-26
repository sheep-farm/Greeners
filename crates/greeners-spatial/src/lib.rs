//! greeners-spatial crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use spatial::{Spatial, SpatialResult};
pub use spatial_durbin::{SpatialDurbin, SpatialDurbinResult};
pub use spatial_durbin_error::{SpatialDurbinError, SpatialDurbinErrorResult};
pub use spatial_panel::{SpatialPanel, SpatialPanelResult};

pub mod spatial;
pub mod spatial_durbin;
pub mod spatial_durbin_error;
pub mod spatial_panel;
