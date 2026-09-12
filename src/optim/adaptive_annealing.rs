use std::num::NonZero;

use ordered_float::NotNan;

use super::{
    AdaptiveAnnealing, AdaptiveScheduler, GenericLocalSearchOptimizer, LocalSearchOptimizer,
};
use crate::{Duration, OptModel, callback::OptCallbackFn, optim::metropolis::tune_temperature};

/// Optimizer that implements the adaptive annealing algorithm which tries to adapt temperature
/// to realize target acceptance rate scheduling.
#[derive(Clone, Copy)]
#[allow(dead_code)] // fields document config for callers building the handler
pub struct AdaptiveAnnealingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    return_iter: usize,
    /// Initial inverse temperature
    initial_beta: f64,
    /// Scheduler for target acceptance rate
    scheduler: AdaptiveScheduler,
    /// Non-zero frequency (in iterations) at which adaptive parameters are updated
    update_frequency: NonZero<usize>,
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
            initial_beta,
            scheduler,
            update_frequency,
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
            self.scheduler.initial_target_acc,
        );

        Self {
            initial_beta: tuned_beta,
            ..self
        }
    }
}

impl<M, H> LocalSearchOptimizer<M, H> for AdaptiveAnnealingOptimizer
where
    M: OptModel<ScoreType = NotNan<f64>>,
    H: Into<AdaptiveAnnealing> + From<AdaptiveAnnealing>,
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
        let h: AdaptiveAnnealing = handler.into();
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
