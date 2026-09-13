use super::{EpsilonGreedyOptimizer, LocalSearchOptimizer};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements random search algorithm
#[derive(Clone, Copy)]
pub struct RandomSearchOptimizer {
    patience: usize,
    /// RNG seed for bit-reproducible runs. `None` (default)
    /// preserves the entropy-driven behavior; set via [`Self::with_seed`].
    seed: Option<u64>,

}

impl RandomSearchOptimizer {
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    pub const fn new(patience: usize) -> Self {
        Self { patience,
            seed: None, }
    }

    /// Pin the RNG seed so [`Self::optimize`] (and the tune helpers)
    /// yield bit-identical `(solution, score)` across calls with the
    /// same inputs.
    ///
    /// `None` (the default) keeps the historical entropy-driven behavior.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

}


impl<M: OptModel> LocalSearchOptimizer<M> for RandomSearchOptimizer {
    fn rng_seed(&self) -> Option<u64> {
        self.seed
    }

    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization
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
        // Random search = epsilon-greedy with epsilon = 1 (accept every move)
        let optimizer = match self.seed {
            Some(s) => EpsilonGreedyOptimizer::new(self.patience, 1, usize::MAX, 1.0).with_seed(s),
            None => EpsilonGreedyOptimizer::new(self.patience, 1, usize::MAX, 1.0),
        };
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
