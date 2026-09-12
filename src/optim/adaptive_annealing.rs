use std::num::NonZero;

use ordered_float::NotNan;

use super::{
    AdaptiveAnnealing, AdaptiveScheduler, LocalSearchLoop, LocalSearchOptimizer, tune_temperature,
};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements the adaptive annealing algorithm which tries to adapt temperature
/// to realize target acceptance rate scheduling.
#[derive(Clone, Copy)]
pub struct AdaptiveAnnealingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    return_iter: usize,
    /// Transition handler that holds the temperature and target-acceptance scheduler
    handler: AdaptiveAnnealing,
}

impl AdaptiveAnnealingOptimizer {
    /// Creates a new `AdaptiveAnnealingOptimizer` instance with the specified parameters.
    pub const fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_beta: f64,
        scheduler: AdaptiveScheduler,
        update_frequency: NonZero<usize>,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            handler: AdaptiveAnnealing::new(initial_beta, scheduler, update_frequency),
        }
    }

    /// Tune inverse temperature parameter beta based on initial random trials
    pub fn tune_initial_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
    ) -> Self {
        let tuned_beta = tune_temperature(
            model,
            initial_solution,
            n_warmup,
            self.handler.scheduler.initial_target_acc,
        );

        Self {
            handler: AdaptiveAnnealing {
                beta: tuned_beta,
                ..self.handler
            },
            ..self
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for AdaptiveAnnealingOptimizer {
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
