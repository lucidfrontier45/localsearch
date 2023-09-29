use ordered_float::NotNan;

use crate::{callback::OptCallbackFn, OptModel};

use super::base::BaseLocalSearchOptimizer;

fn transition_prob<T: Into<f64>>(current: T, trial: T, w: f64) -> f64 {
    let current = current.into();
    let trial = trial.into();
    let d = (trial - current) / current;
    2.0 / (1.0 + (w * d).exp())
}

/// Optimizer that implements logistic annealing algorithm
/// In this model, unlike simulated annealing, wether accept the trial state or not is calculated based on relative score difference
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
    /// - `n_trials` : number of trial states to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best state if there is no improvement after this number of iterations.
    /// - `w` : weight to be multiplied with the relative score difference.
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, w: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            w,
        }
    }

    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_state` : the initial state to start optimization. If None, a random state will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    pub fn optimize<M, F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        callback: Option<&F>,
    ) -> (M::StateType, M::ScoreType)
    where
        M: OptModel<ScoreType = NotNan<f64>> + Sync + Send,
        F: OptCallbackFn<M::StateType, M::ScoreType>,
    {
        let optimizer = BaseLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            |current, trial| transition_prob(current, trial, self.w),
        );

        optimizer.optimize(model, initial_state, n_iter, callback)
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
