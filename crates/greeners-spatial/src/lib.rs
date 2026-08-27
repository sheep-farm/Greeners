//! greeners-spatial crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

#[cfg(feature = "experimental")]
pub use spatial::{Spatial, SpatialResult};
#[cfg(feature = "experimental")]
pub use spatial_durbin::{SpatialDurbin, SpatialDurbinResult};
#[cfg(feature = "experimental")]
pub use spatial_durbin_error::{SpatialDurbinError, SpatialDurbinErrorResult};
#[cfg(feature = "experimental")]
pub use spatial_panel::{SpatialPanel, SpatialPanelResult};

#[cfg(feature = "experimental")]
pub mod spatial;
#[cfg(feature = "experimental")]
pub mod spatial_durbin;
#[cfg(feature = "experimental")]
pub mod spatial_durbin_error;
#[cfg(feature = "experimental")]
pub mod spatial_panel;
