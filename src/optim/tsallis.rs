use std::num::NonZero;

use ordered_float::NotNan;

use super::{
    AdaptiveScheduler, GenericLocalSearchOptimizer, LocalSearchOptimizer, TargetAccScheduleMode,
    TsallisAnnealing,
};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements Tsallis relative annealing algorithm
/// This is a generalization of relative annealing using Tsallis statistics.
/// The acceptance probability for worse solutions is
/// `[1 - (1-q) * beta * ΔE / (E - E_best + ξ)]^{1/(1-q)}`,
/// where `ΔE = trial - current`, `E = current`, `E_best = offset`.
/// Assumes `q > 1.0`.
#[derive(Clone, Copy)]
#[allow(dead_code)] // fields document config for callers building the handler
pub struct TsallisRelativeAnnealingOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    initial_beta: f64,
    scheduler: AdaptiveScheduler,
    update_frequency: NonZero<usize>,
    q: f64,
    xi: f64,
}

impl TsallisRelativeAnnealingOptimizer {
    /// Constructor of TsallisRelativeAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `initial_beta` : initial weight to be multiplied with the relative score difference.
    ///   Recommended value is reciprocal of expected relative score difference.
    /// - `update_frequency` : non-zero frequency at which certain parameters (like beta) are updated during optimization.
    /// - `q` : Tsallis parameter, assumed to be > 1.0. Recommended value is 2.5.
    /// - `xi` : parameter ξ in the acceptance probability formula.
    ///   Recommended value is 1.0 for integer objective and 0.1% of the objective value for continuous objective.
    pub const fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        beta: f64,
        update_frequency: NonZero<usize>,
        q: f64,
        xi: f64,
    ) -> Self {
        let scheduler = AdaptiveScheduler::new(0.3, 0.3, TargetAccScheduleMode::Constant, 0.05);
        Self {
            patience,
            n_trials,
            return_iter,
            initial_beta: beta,
            update_frequency,
            q,
            xi,
            scheduler,
        }
    }

    /// Sets the scheduler for the optimizer.
    pub const fn with_scheduler(mut self, scheduler: AdaptiveScheduler) -> Self {
        self.scheduler = scheduler;
        self
    }
}

impl<M, H> LocalSearchOptimizer<M, H> for TsallisRelativeAnnealingOptimizer
where
    M: OptModel<ScoreType = NotNan<f64>>,
    H: Into<TsallisAnnealing> + From<TsallisAnnealing>,
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
        let h: TsallisAnnealing = handler.into();
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
