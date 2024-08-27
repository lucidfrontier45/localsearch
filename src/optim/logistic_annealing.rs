use ordered_float::NotNan;

use crate::{callback::OptCallbackFn, Duration, OptModel};

use super::{GenericLocalSearchOptimizer, LocalSearchOptimizer};

fn transition_prob<T: Into<f64>>(current: T, trial: T, w: f64) -> f64 {
    let current = current.into();
    let trial = trial.into();
    let d = (trial - current) / current;
    2.0 / (1.0 + (w * d).exp())
}

/// Optimizer that implements logistic annealing algorithm
/// In this model, unlike simulated annealing, wether accept the trial solution or not is calculated based on relative score difference
///
/// 1. d <- (trial_score - current_score) / current_score
/// 2. p <- 2.0 / (1.0 + exp(w * d))
/// 3. accept if p > rand(0, 1)
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
    }
}
