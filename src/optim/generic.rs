use std::marker::PhantomData;

use super::{LocalSearchOptimizer, TransitionHandler, search_loop::LocalSearchLoop};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Generic local-search optimizer that owns a [`TransitionHandler`].
///
/// Wraps [`LocalSearchLoop`] (the trial/accept loop shared by every annealing
/// variant) and exposes it through the [`LocalSearchOptimizer`] trait so it
/// can be used wherever a concrete optimizer is expected.
///
/// The stored handler is treated as a *blueprint*: each call to
/// [`optimize`](LocalSearchOptimizer::optimize) clones it (via
/// [`Clone`]) and lets the per-iteration
/// [`TransitionHandler::update`] mutations run on the working copy. The stored
/// handler is therefore left untouched across runs and can be reused
/// indefinitely.
///
/// Use this when you want to drive an arbitrary algorithm through the
/// standard `LocalSearchOptimizer` interface. When you need full control over
/// the handler's lifetime (e.g. to inspect its post-run state), reach for
/// [`LocalSearchLoop`] directly.
pub struct GenericLocalSearchOptimizer<ST, H> {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    handler: H,
    phantom: PhantomData<ST>,
}

impl<ST, H> GenericLocalSearchOptimizer<ST, H>
where
    ST: Ord + Send + Sync + Copy,
    H: TransitionHandler<ST> + Send + Sync,
{
    /// Constructor of `GenericLocalSearchOptimizer`.
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations
    /// - `handler` : the [`TransitionHandler`] used to evaluate trial solutions
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, handler: H) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            handler,
            phantom: PhantomData,
        }
    }

    /// Borrow the stored handler blueprint.
    pub fn handler(&self) -> &H {
        &self.handler
    }
}

impl<M, H> LocalSearchOptimizer<M> for GenericLocalSearchOptimizer<M::ScoreType, H>
where
    M: OptModel,
    M::ScoreType: Ord + Send + Sync + Copy,
    H: TransitionHandler<M::ScoreType> + Clone,
{
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType) {
        let handler = self.handler.clone();
        let loop_ = LocalSearchLoop::<M::ScoreType>::new(
            self.patience,
            self.n_trials,
            self.return_iter,
        );
        let (solution, score, _) = loop_.optimize_with_handler(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            handler,
        );
        (solution, score)
    }
}
