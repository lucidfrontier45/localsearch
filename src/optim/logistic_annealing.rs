use ordered_float::NotNan;

use super::{GenericLocalSearchOptimizer, LocalSearchOptimizer, LogisticAnnealing};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Optimizer that implements logistic annealing algorithm
/// In this model, unlike simulated annealing, whether accept the trial solution or not is calculated based on relative score difference
///
/// 1. `d <- (trial_score - current_score) / current_score.abs()`
/// 2. `p <- 2.0 / (1.0 + exp(w * d))`
/// 3. accept if `p > rand(0, 1)`
///
/// Using `|current_score|` keeps the sign of `d` aligned with `(trial - current)`,
/// so the acceptance direction is correct for any sign of `current_score`.
/// `current_score == 0` is clamped to `f64::EPSILON`, keeping the result finite
/// and the acceptance direction intact (improvement accepted, worsening rejected).
#[derive(Clone, Copy)]
#[allow(dead_code)] // fields document config for callers building the handler
pub struct LogisticAnnealingOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    w: f64,
}

impl LogisticAnnealingOptimizer {
    /// Constructor of LogisticAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `w` : weight to be multiplied with the relative score difference.
    pub const fn new(patience: usize, n_trials: usize, return_iter: usize, w: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            w,
        }
    }
}

impl<M, H> LocalSearchOptimizer<M, H> for LogisticAnnealingOptimizer
where
    M: OptModel<ScoreType = NotNan<f64>>,
    H: Into<LogisticAnnealing> + From<LogisticAnnealing>,
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
        let h: LogisticAnnealing = handler.into();
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
