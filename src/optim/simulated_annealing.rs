use std::num::NonZero;

use ordered_float::NotNan;

use super::{
    LocalSearchLoop, LocalSearchOptimizer, SimulatedAnnealing, tune_cooling_rate, tune_temperature,
};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements the simulated annealing algorithm
#[derive(Clone, Copy)]
pub struct SimulatedAnnealingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    return_iter: usize,
    /// Transition handler that holds the inverse temperature and cooling schedule
    handler: SimulatedAnnealing,
}

impl SimulatedAnnealingOptimizer {
    /// Constructor of SimulatedAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `initial_beta` : initial inverse temperature
    /// - `cooling_rate` : cooling rate
    /// - `update_frequency` : non-zero number of steps after which inverse temperature (beta) is updated
    pub const fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_beta: f64,
        cooling_rate: f64,
        update_frequency: NonZero<usize>,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            handler: SimulatedAnnealing::new(initial_beta, cooling_rate, update_frequency),
        }
    }

    /// Tune inverse temperature parameter beta based on initial random trials
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `n_warmup` : number of warmup iterations to run
    /// - `target_initial_prob` : target acceptance probability for uphill moves at the beginning
    pub fn tune_initial_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
        target_initial_prob: f64,
    ) -> Self {
        let tuned_beta = tune_temperature(model, initial_solution, n_warmup, target_initial_prob);

        Self {
            handler: SimulatedAnnealing {
                beta: tuned_beta,
                ..self.handler
            },
            ..self
        }
    }

    /// Tune cooling rate based on the handler's current initial beta, final beta of 1e2
    pub fn tune_cooling_rate(self, n_iter: usize) -> Self {
        let cooling_rate = tune_cooling_rate(
            self.handler.beta,
            1e2,
            n_iter / self.handler.update_frequency.get(),
        );

        Self {
            handler: SimulatedAnnealing {
                cooling_rate,
                ..self.handler
            },
            ..self
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for SimulatedAnnealingOptimizer {
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
