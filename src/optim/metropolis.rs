use ordered_float::NotNan;

use crate::{
    Duration, OptModel,
    callback::OptCallbackFn,
};

use super::{GenericLocalSearchOptimizer, LocalSearchOptimizer};

/// Optimizer that implements the Metropolis algorithm with constant temperature
#[derive(Clone, Copy)]
pub struct MetropolisOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    pub patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    pub n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    pub return_iter: usize,
    /// Constant temperature
    pub temperature: f64,
}

impl MetropolisOptimizer {
    /// Constructor of MetropolisOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the best solution if there is no improvement after this number of iterations.
    /// - `temperature` : constant temperature
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, temperature: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            temperature,
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for MetropolisOptimizer {
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
        let transition = |current: NotNan<f64>, trial: NotNan<f64>| {
            let ds = trial - current;
            if ds <= NotNan::new(0.0).unwrap() {
                1.0
            } else {
                (-ds.into_inner() / self.temperature).exp()
            }
        };
        let generic_optimizer = GenericLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            transition,
        );
        generic_optimizer.optimize(model, initial_solution, initial_score, n_iter, time_limit, callback)
    }
}
