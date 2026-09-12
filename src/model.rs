use auto_impl::auto_impl;

use crate::LocalsearchError;

/// OptModel is a trait that defines requirements to be used with optimization algorithm
#[auto_impl(&, Box, Rc, Arc)]
pub trait OptModel: Sync + Send {
    /// Type of the Score
    type ScoreType: Ord + Copy + Sync + Send;
    /// Type of the Solution
    type SolutionType: Clone + Sync + Send;
    /// Type of the Transition
    type TransitionType: Clone + Sync + Send;
    /// Externally-owned mutable state shared with the optimizer.
    ///
    /// The model itself stays immutable; any bookkeeping that would otherwise
    /// live behind `Arc<Mutex<...>>` on the model (for example ALNS
    /// operator-selection weights, or an external best-score cache) is owned
    /// by the caller and threaded through `optimize` / `run` / `run_with_callback`
    /// as a `&Self::StateType` parameter. The state must be `Sync` because the
    /// optimizer typically drives `generate_trial_solution` from rayon worker
    /// threads in parallel.
    ///
    /// Stateless models set this to `()` and ignore the parameter.
    type StateType: Sync + Send;

    /// Randomly generate a solution
    fn generate_random_solution<R: rand::Rng>(
        &self,
        state: &Self::StateType,
        rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError>;

    /// Generate a new trial solution from current solution
    fn generate_trial_solution<R: rand::Rng>(
        &self,
        state: &Self::StateType,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType);

    /// Preprocess the solution
    fn preprocess_solution(
        &self,
        _state: &Self::StateType,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        Ok((current_solution, current_score))
    }

    /// Postprocess the solution
    fn postprocess_solution(
        &self,
        _state: &Self::StateType,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
    ) -> (Self::SolutionType, Self::ScoreType) {
        (current_solution, current_score)
    }
}
