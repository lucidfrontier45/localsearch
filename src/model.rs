use std::error::Error;

use auto_impl::auto_impl;

/// OptModel is a trait that defines requirements to be used with optimization algorithm
#[auto_impl(&, Box, Rc, Arc)]
pub trait OptModel: Sync + Send {
    /// Type of the Score
    type ScoreType: Ord + Copy + Sync + Send;
    /// Type of the Solution
    type SolutionType: Clone + Sync + Send;
    /// Type of the Transition
    type TransitionType: Clone + Sync + Send;

    /// Randomly generate a solution
    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<Self::SolutionType, Box<dyn Error>>;

    /// Generate a new trial solution from current solution
    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: &Self::SolutionType,
        rng: &mut R,
        current_score: Option<Self::ScoreType>,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType);

    /// Evaluate the given solution
    fn evaluate_solution(&self, solution: &Self::SolutionType) -> Self::ScoreType;
}
