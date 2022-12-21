use std::error::Error;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub mod optim;
pub mod utils;

pub trait OptModel<StateType, TransitionType, O: Ord + Copy + Send + Sync> {
    fn generate_random_state<R: rand::Rng>(&self, rng: &mut R)
        -> Result<StateType, Box<dyn Error>>;
    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &StateType,
        rng: &mut R,
        current_score: Option<O>,
    ) -> (StateType, TransitionType, O);
    fn evaluate_state(&self, state: &StateType) -> O;
}

#[cfg(test)]
mod tests;
