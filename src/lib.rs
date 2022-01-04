use std::error::Error;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub mod optim;
pub mod utils;

pub trait OptModel<StateType, TransitionType> {
    fn generate_random_state<R: rand::Rng>(&self, rng: &mut R)
        -> Result<StateType, Box<dyn Error>>;
    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &StateType,
        rng: &mut R,
    ) -> (StateType, TransitionType);
    fn evaluate_state(&self, state: &StateType) -> f64;
}

#[cfg(test)]
mod tests;
