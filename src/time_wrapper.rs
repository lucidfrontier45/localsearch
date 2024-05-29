#[cfg(not(target_family = "wasm"))]
pub use std::time::{Duration, Instant};

#[cfg(target_family = "wasm")]
pub use web_time::{Duration, Instant};
