use super::{EpsilonGreedyOptimizer, LocalSearchOptimizer};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements random search algorithm
#[derive(Clone, Copy)]
pub struct RandomSearchOptimizer {
    patience: usize,
}

impl RandomSearchOptimizer {
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    pub const fn new(patience: usize) -> Self {
        Self { patience }
    }
}

impl<M, H> LocalSearchOptimizer<M, H> for RandomSearchOptimizer
where
    M: OptModel,
    H: Into<crate::optim::EpsilonGreedy> + From<crate::optim::EpsilonGreedy>,
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
        let inner = EpsilonGreedyOptimizer::new(self.patience, 1, usize::MAX, 1.0);
        inner.optimize(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            handler,
        )
    }
}
