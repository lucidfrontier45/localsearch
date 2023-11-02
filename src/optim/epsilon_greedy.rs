use crate::{callback::OptCallbackFn, OptModel};

use super::{base::LocalSearchOptimizer, GenericLocalSearchOptimizer};

fn transition_prob<T: PartialOrd>(current: T, trial: T, epsilon: f64) -> f64 {
    if trial < current {
        return 1.0;
    }
    epsilon
}

/// Optimizer that implements epsilon-greedy algorithm.
/// Unlike a total greedy algorithm such as hill climbing,
/// it allows transitions that worsens the score with a fixed probability
#[derive(Clone, Copy)]
pub struct EpsilonGreedyOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    epsilon: f64,
}

impl EpsilonGreedyOptimizer {
    /// Constructor of EpsilonGreedyOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial states to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best state if there is no improvement after this number of iterations.
    /// - `epsilon` : probability to accept a transition that worsens the score. Must be in [0, 1].
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, epsilon: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            epsilon,
        }
    }
}

impl<M: OptModel> LocalSearchOptimizer<M> for EpsilonGreedyOptimizer {
    type ExtraIn = ();
    type ExtraOut = ();
    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_state` : the initial state to start optimization. If None, a random state will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    ///
    fn optimize<F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        callback: Option<&F>,
        _extra_in: Self::ExtraIn,
    ) -> (M::StateType, M::ScoreType, Self::ExtraOut)
    where
        M: OptModel + Sync + Send,
        F: OptCallbackFn<M::StateType, M::ScoreType>,
    {
        let optimizer = GenericLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            |current, trial| transition_prob(current, trial, self.epsilon),
        );
        optimizer.optimize(model, initial_state, n_iter, callback, _extra_in)
    }
}
