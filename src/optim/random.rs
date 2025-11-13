use crate::{Duration, OptModel, callback::OptCallbackFn};

use super::{EpsilonGreedyOptimizer, LocalSearchOptimizer};

/// Optimizer that implements simple hill climbing algorithm
#[derive(Clone, Copy)]
pub struct RandomSearchOptimizer {
    patience: usize,
}

impl RandomSearchOptimizer {
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    pub fn new(patience: usize) -> Self {
        Self { patience }
    }
}

impl<M: OptModel> LocalSearchOptimizer<M> for RandomSearchOptimizer {
    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType) {
        let optimizer = EpsilonGreedyOptimizer::new(self.patience, 1, usize::MAX, 1.0);
        optimizer.optimize(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
        )
    }
}
