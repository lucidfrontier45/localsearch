#![forbid(missing_docs)]
#![allow(clippy::non_ascii_literal)]
#![allow(clippy::module_name_repetitions)]
#![doc = include_str!("../README.md")]

pub mod optim;
pub mod utils;

mod callback;
pub use callback::{OptCallbackFn, OptProgress};

mod model;
pub use model::OptModel;

mod time_wrapper;
pub use time_wrapper::{Duration, Instant};

/// Crate verison string
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests;
