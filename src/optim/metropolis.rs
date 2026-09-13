use ordered_float::NotNan;

use super::{LocalSearchLoop, LocalSearchOptimizer, Metropolis};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements the Metropolis algorithm with constant beta.
#[derive(Clone, Copy)]
pub struct MetropolisOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    return_iter: usize,
    /// Transition handler that holds the inverse temperature
    handler: Metropolis,
    /// RNG seed for bit-reproducible runs. `None` (default)
    /// preserves the entropy-driven behavior; set via [`Self::with_seed`].
    seed: Option<u64>,
}

impl MetropolisOptimizer {
    /// Constructor of MetropolisOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `beta` : inverse temperature
    pub const fn new(patience: usize, n_trials: usize, return_iter: usize, beta: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            handler: Metropolis::new(beta),
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

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for MetropolisOptimizer {
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
