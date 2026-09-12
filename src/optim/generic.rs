use std::marker::PhantomData;

use super::{
    DefaultTrialGenerator, LocalSearchOptimizer, TransitionHandler,
    search_loop::{LocalSearchLoop, TrialGenerator},
};
use crate::{Duration, OptModel, callback::OptCallbackFn};

/// Generic local-search optimizer that owns a [`TransitionHandler`] and,
/// optionally, a [`TrialGenerator`].
///
/// Wraps [`LocalSearchLoop`] (the trial/accept loop shared by every
/// annealing variant) and exposes it through the [`LocalSearchOptimizer`]
/// trait so it can be used wherever a concrete optimizer is expected.
///
/// The stored handler is treated as a *blueprint*: each call to
/// [`optimize`](LocalSearchOptimizer::optimize) clones it (via
/// [`Clone`]) and lets the per-iteration
/// [`TransitionHandler::update`] mutations run on the working copy. The
/// stored handler is therefore left untouched across runs and can be
/// reused indefinitely.
///
/// Use this when you want to drive an arbitrary algorithm through the
/// standard `LocalSearchOptimizer` interface. When you need full control
/// over the handler's lifetime (e.g. to inspect its post-run state), reach
/// for [`LocalSearchLoop`] directly.
///
/// # Trial generation
///
/// By default the optimizer uses [`DefaultTrialGenerator`], which simply
/// delegates to [`OptModel::generate_trial_solution`]. To plug in an
/// adaptive scheme (such as ALNS), call
/// [`Self::with_trial_generator`] with a custom generator. The generator
/// must implement [`Clone`] because it is cloned for each `optimize`
/// call, just like the handler.
pub struct GenericLocalSearchOptimizer<ST, H, G = DefaultTrialGenerator> {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    handler: H,
    generator: G,
    phantom: PhantomData<ST>,
}

impl<ST, H> GenericLocalSearchOptimizer<ST, H, DefaultTrialGenerator>
where
    ST: Ord + Send + Sync + Copy,
    H: TransitionHandler<ST>,
{
    /// Constructor of `GenericLocalSearchOptimizer`.
    ///
    /// Uses [`DefaultTrialGenerator`] for trial generation so existing
    /// call sites keep their previous behavior. To switch to ALNS or any
    /// other adaptive scheme, chain [`Self::with_trial_generator`].
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
            generator: DefaultTrialGenerator,
            phantom: PhantomData,
        }
    }
}

impl<ST, H, G> GenericLocalSearchOptimizer<ST, H, G>
where
    ST: Ord + Send + Sync + Copy,
    H: TransitionHandler<ST>,
{
    /// Borrow the stored handler blueprint.
    pub fn handler(&self) -> &H {
        &self.handler
    }

    /// Borrow the stored trial generator.
    pub fn generator(&self) -> &G {
        &self.generator
    }

    /// Builder that swaps in a different [`TrialGenerator`]. The original
    /// optimizer is consumed and a new optimizer is returned with the
    /// supplied generator.
    pub fn with_trial_generator<NG>(self, generator: NG) -> GenericLocalSearchOptimizer<ST, H, NG> {
        GenericLocalSearchOptimizer {
            patience: self.patience,
            n_trials: self.n_trials,
            return_iter: self.return_iter,
            handler: self.handler,
            generator,
            phantom: PhantomData,
        }
    }
}

impl<M, H, G> LocalSearchOptimizer<M> for GenericLocalSearchOptimizer<M::ScoreType, H, G>
where
    M: OptModel,
    M::ScoreType: Ord + Send + Sync + Copy,
    H: TransitionHandler<M::ScoreType> + Clone,
    G: TrialGenerator<M> + Clone,
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
        let generator = self.generator.clone();
        let loop_ =
            LocalSearchLoop::<M::ScoreType>::new(self.patience, self.n_trials, self.return_iter);
        let (result, _, _) = loop_.step_with_generator(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            handler,
            generator,
        );
        (result.best_solution, result.best_score)
    }
}
