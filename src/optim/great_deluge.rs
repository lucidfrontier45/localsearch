use ordered_float::NotNan;

use super::{LocalSearchLoop, GreatDeluge, LocalSearchOptimizer};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements the Great Deluge Algorithm (GDA).
/// Unlike probabilistic methods like simulated annealing, GDA uses a deterministic
/// threshold ("water level") that decreases adaptively over iterations.
/// A trial solution is accepted if its score is below or equal to the current water level.
#[derive(Clone, Copy)]
pub struct GreatDelugeOptimizer {
    /// Patience: the optimizer will give up if there is no improvement after this many iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Return to the current best solution if there is no improvement after this many iterations
    return_iter: usize,
    /// Factor to initialize the water level as `initial_score * level_factor`
    level_factor: f64,
}

impl GreatDelugeOptimizer {
    /// Constructor for GreatDelugeOptimizer
    ///
    /// - `patience`: the optimizer will give up if there is no improvement after this many iterations
    /// - `n_trials`: number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter`: returns to the best solution if there is no improvement after this many iterations
    /// - `level_factor`: multiplier for initial water level (e.g., 1.1 for 10% above initial score)
    pub const fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        level_factor: f64,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            level_factor,
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for GreatDelugeOptimizer {
    /// Start optimization
    ///
    /// - `model`: the model to optimize
    /// - `initial_solution`: the initial solution to start optimization
    /// - `initial_score`: the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback`: callback function that will be invoked at the end of each iteration
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType) {
        // Initialize water level from this run's initial score
        let initial_level = initial_score.into_inner() * self.level_factor;
        let handler = GreatDeluge::new(initial_level);
        let opt = LocalSearchLoop::new(self.patience, self.n_trials, self.return_iter);
        let (result, _) = opt.step(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            handler,
        );
        (result.best_solution, result.best_score)
    }
}
