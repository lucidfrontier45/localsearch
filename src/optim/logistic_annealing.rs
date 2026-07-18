use ordered_float::NotNan;

use super::{GenericLocalSearchOptimizer, LocalSearchOptimizer};
use crate::{Duration, OptModel, callback::OptCallbackFn};

fn transition_prob<T: Into<f64>>(current: T, trial: T, w: f64) -> f64 {
    let current = current.into();
    let trial = trial.into();
    // Clamp denominator to `f64::EPSILON` to avoid division-by-zero when
    // `current_score == 0`. Relative-comparison semantics preserved for any
    // `|current_score| >= f64::EPSILON`.
    let d = (trial - current) / current.abs().max(f64::EPSILON);
    2.0 / (1.0 + (w * d).exp())
}

/// Optimizer that implements logistic annealing algorithm
/// In this model, unlike simulated annealing, whether accept the trial solution or not is calculated based on relative score difference
///
/// 1. d <- (trial_score - current_score) / current_score.abs()
/// 2. p <- 2.0 / (1.0 + exp(w * d))
/// 3. accept if p > rand(0, 1)
///
/// Using `|current_score|` keeps the sign of `d` aligned with `(trial - current)`,
/// so the acceptance direction is correct for any sign of `current_score`.
/// `current_score == 0` is clamped to `f64::EPSILON`, keeping the result finite
/// and the acceptance direction intact (improvement accepted, worsening rejected).
#[derive(Clone, Copy)]
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
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, w: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            w,
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for LogisticAnnealingOptimizer {
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
        let optimizer = GenericLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            |current, trial| transition_prob(current, trial, self.w),
        );

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

#[cfg(test)]
mod test {
    use super::transition_prob;

    #[test]
    fn test_transition_prob() {
        let w = 1e1;

        let p = transition_prob(1.0, 0.9, w);
        assert!(p >= 1.0);

        let p1 = transition_prob(1.0, 1.1, w);
        let p2 = transition_prob(1.0, 1.2, w);
        assert!(p1 > p2);

        // negative / sign-flipped current must NOT invert acceptance direction:
        // trial < current  ->  p >= 1 (accept improvement), regardless of sign of current.
        assert!(transition_prob(-1.0, -1.1, w) >= 1.0);
        assert!(transition_prob(-1.0, -0.9, w) < 1.0);
        // crossing zero: worsening move (trial > current) rejected more strongly with larger gap
        assert!(transition_prob(-1.0, -0.9, w) > transition_prob(-1.0, -0.8, w));
        assert!(transition_prob(-1.0, 0.5, w) < 1.0);

        // `current_score == 0`: relative comparison degenerates; clamp must keep
        // the result finite and the direction correct.
        assert!(transition_prob(0.0, 0.0, w).is_finite());
        // improvement (trial < current=0): p should approach 2 (accepted).
        assert!(transition_prob(0.0, -1.0, w) > 1.0);
        // worsening (trial > current=0): p should approach 0 (rejected, finite).
        let p_worse = transition_prob(0.0, 1.0, w);
        assert!(p_worse.is_finite());
        assert!(p_worse < 1.0);
    }
}
