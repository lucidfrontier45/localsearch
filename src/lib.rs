use std::error::Error;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub mod optim;
pub mod utils;

pub trait OptModel {
    type ScoreType: Ord + Copy + Sync + Send;
    type StateType: Clone + Sync + Send;
    type TransitionType: Clone + Sync + Send;

    fn generate_random_state<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<Self::StateType, Box<dyn Error>>;
    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &Self::StateType,
        rng: &mut R,
        current_score: Option<Self::ScoreType>,
    ) -> (Self::StateType, Self::TransitionType, Self::ScoreType);
    fn evaluate_state(&self, state: &Self::StateType) -> Self::ScoreType;
}

#[cfg(test)]
mod tests;
