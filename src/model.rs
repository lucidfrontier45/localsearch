use std::error::Error;

use auto_impl::auto_impl;

/// OptModel is a trait that defines requirements to be used with optimization algorithm
#[auto_impl(&, Box, Rc, Arc)]
pub trait OptModel: Sync + Send {
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
