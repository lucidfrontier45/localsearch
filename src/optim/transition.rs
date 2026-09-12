//! Core traits for the transition-handler refactor.
//!
//! Each local-search optimizer owns a [`TransitionHandler`] that decides
//! whether a trial solution is accepted and how its internal state
//! (e.g. inverse temperature) evolves across iterations.

/// Context passed to [`TransitionHandler::update`] at the start of each
/// iteration, before any trial solution is generated.
///
/// `best` is the best score observed at the time `update` is invoked
/// (i.e. before the current iteration's trials are evaluated).
/// `acc` is the acceptance ratio over the sliding window as of the
/// previous iteration's end.
pub struct UpdateCtx<'a, ST> {
    /// Zero-based iteration counter.
    pub iter: usize,
    /// Total number of iterations the caller intends to run.
    pub total: usize,
    /// Sliding-window acceptance ratio from the previous iteration.
    pub acc: f64,
    /// Best score observed before the current iteration's trials.
    pub best: &'a ST,
}

/// Per-algorithm transition logic.
///
/// Implementors own any state required to evaluate acceptance
/// probabilities and to mutate themselves between iterations.
/// No `Rc<RefCell>` is required because the handler is stored by
/// value inside [`super::generic::GenericLocalSearchOptimizer`].
pub trait TransitionHandler<ST: Ord + Send + Sync + Copy>: Send + Sync {
    /// Called once per iteration, before any trial is generated.
    ///
    /// Use this to update cooling schedules, water levels, offsets, etc.
    fn update(&mut self, ctx: &UpdateCtx<'_, ST>);

    /// Returns the acceptance probability for a transition from
    /// `current` to `trial`. Values `>= 1.0` mean "always accept".
    fn evaluate(&self, current: ST, trial: ST) -> f64;
}
