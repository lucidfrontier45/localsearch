use std::error::Error;

pub mod optim;

pub trait OptModel {
    type StateType;
    type TransitionType;

    fn generate_random_state<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<Self::StateType, Box<dyn Error>>;
    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &Self::StateType,
        rng: &mut R,
    ) -> (Self::StateType, Self::TransitionType);
    fn evaluate_state(&self, state: &Self::StateType) -> f64;
}

#[cfg(test)]
mod tests;
