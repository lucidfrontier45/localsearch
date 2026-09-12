use super::{EpsilonGreedy, GenericLocalSearchOptimizer, LocalSearchOptimizer};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements epsilon-greedy algorithm.
/// Unlike a total greedy algorithm such as hill climbing,
/// it allows transitions that worsens the score with a fixed probability
#[derive(Clone, Copy)]
#[allow(dead_code)] // fields document config for callers building the handler
pub struct EpsilonGreedyOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    epsilon: f64,
}

impl EpsilonGreedyOptimizer {
    /// Constructor of EpsilonGreedyOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `epsilon` : probability to accept a transition that worsens the score. Must be in [0, 1].
    pub const fn new(patience: usize, n_trials: usize, return_iter: usize, epsilon: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            epsilon,
        }
    }
}

impl<M, H> LocalSearchOptimizer<M, H> for EpsilonGreedyOptimizer
where
    M: OptModel,
    H: Into<EpsilonGreedy> + From<EpsilonGreedy>,
{
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        handler: H,
    ) -> (M::SolutionType, M::ScoreType, H) {
        let h: EpsilonGreedy = handler.into();
        let opt = GenericLocalSearchOptimizer::new(self.patience, self.n_trials, self.return_iter);
        let (solution, score, h) = opt.optimize_with_handler(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            h,
        );
        (solution, score, H::from(h))
    }
}
