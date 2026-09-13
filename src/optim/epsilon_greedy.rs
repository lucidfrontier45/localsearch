use super::{EpsilonGreedy, LocalSearchLoop, LocalSearchOptimizer};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements epsilon-greedy algorithm.
/// Unlike a total greedy algorithm such as hill climbing,
/// it allows transitions that worsens the score with a fixed probability
#[derive(Clone, Copy)]
pub struct EpsilonGreedyOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    handler: EpsilonGreedy,
    /// RNG seed for bit-reproducible runs. `None` (default)
    /// preserves the entropy-driven behavior; set via [`Self::with_seed`].
    seed: Option<u64>,
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
            handler: EpsilonGreedy::new(epsilon),
            seed: None,
        }
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

impl<M: OptModel> LocalSearchOptimizer<M> for EpsilonGreedyOptimizer {
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
        let opt = LocalSearchLoop::new(self.patience, self.n_trials, self.return_iter);
        let opt = match self.seed {
            Some(s) => opt.with_seed(s),
            None => opt,
        };

        let (result, _) = opt.step(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            self.handler,
        );
        (result.best_solution, result.best_score)
    }
}
