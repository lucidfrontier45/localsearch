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
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
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
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `_extra_in` : not used
    fn optimize<F>(
        &self,
        model: &M,
        initial_solution: Option<M::SolutionType>,
        n_iter: usize,
        callback: Option<&F>,
        _extra_in: Self::ExtraIn,
    ) -> (M::SolutionType, M::ScoreType, Self::ExtraOut)
    where
        F: OptCallbackFn<M::SolutionType, M::ScoreType>,
    {
        let optimizer = EpsilonGreedyOptimizer::new(self.patience, self.n_trials, usize::MAX, 0.0);
        optimizer.optimize(model, initial_solution, n_iter, callback, _extra_in)
    }
}
