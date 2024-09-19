use anyhow::Result as AnyResult;

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
    fn generate_random_solution<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> AnyResult<(Self::SolutionType, Self::ScoreType)>;

    /// Generate a new trial solution from current solution
    fn generate_trial_solution<R: rand::Rng + ?Sized>(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType);

    /// Preprocess the solution
    fn preprocess_solution(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
    ) -> AnyResult<(Self::SolutionType, Self::ScoreType)> {
        Ok((current_solution, current_score))
    }

    /// Postprocess the solution
    fn postprocess_solution(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
    ) -> (Self::SolutionType, Self::ScoreType) {
        (current_solution, current_score)
    }
}
