#![forbid(missing_docs)]
#![allow(clippy::non_ascii_literal)]
#![allow(clippy::module_name_repetitions)]
#![doc = include_str!("../README.md")]

pub mod optim;
pub mod utils;

use std::error::Error;

mod callback;
pub use callback::{OptCallbackFn, OptProgress};

/// Crate verison string
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// OptModel is a trait that defines requirements to be used with optimization algorithm
pub trait OptModel {
    /// Type of the Score
    type ScoreType: Ord + Copy + Sync + Send;
    /// Type of the State
    type StateType: Clone + Sync + Send;
    /// Type of the Transition
    type TransitionType: Clone + Sync + Send;

    /// Randomly generate a state
    fn generate_random_state<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<Self::StateType, Box<dyn Error>>;

    /// Generate a new trial state from current state
    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &Self::StateType,
        rng: &mut R,
        current_score: Option<Self::ScoreType>,
    ) -> (Self::StateType, Self::TransitionType, Self::ScoreType);

    /// Evaluate the given state
    fn evaluate_state(&self, state: &Self::StateType) -> Self::ScoreType;
}

#[cfg(test)]
mod tests;
