use crate::{callback::OptCallbackFn, OptModel};

use super::{EpsilonGreedyOptimizer, LocalSearchOptimizer};

/// Optimizer that implements simple hill climbing algorithm
#[derive(Clone, Copy)]
pub struct HillClimbingOptimizer {
    patience: usize,
    n_trials: usize,
}

impl HillClimbingOptimizer {
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial states to generate and evaluate at each iteration
    pub fn new(patience: usize, n_trials: usize) -> Self {
        Self { patience, n_trials }
    }
}

impl<M: OptModel> LocalSearchOptimizer<M> for HillClimbingOptimizer {
    type ExtraIn = ();
    type ExtraOut = ();

    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_state` : the initial state to start optimization. If None, a random state will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `_extra_in` : not used
    fn optimize<F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        callback: Option<&F>,
        _extra_in: Self::ExtraIn,
    ) -> (M::StateType, M::ScoreType, Self::ExtraOut)
    where
        F: OptCallbackFn<M::StateType, M::ScoreType>,
    {
        let optimizer = EpsilonGreedyOptimizer::new(self.patience, self.n_trials, usize::MAX, 0.0);
        optimizer.optimize(model, initial_state, n_iter, callback, _extra_in)
    }
}
