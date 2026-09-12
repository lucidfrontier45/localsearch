use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use rand::RngExt as _;
use rayon::prelude::*;

use super::transition::TransitionHandler;
use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
    counter::AcceptanceCounter,
};

/// Result of an optimization step, containing information about the best and last solutions,
/// as well as the acceptance counter for the step.
pub struct StepResult<S, ST> {
    /// The best solution found during this step.
    pub best_solution: S,
    /// The score of the best solution found during this step.
    pub best_score: ST,
    /// The last solution at the end of this step (may differ from the best).
    pub last_solution: S,
    /// The score of the last solution at the end of this step.
    pub last_score: ST,
    /// Acceptance counter for the step.
    pub acceptance_counter: AcceptanceCounter,
}

/// Inner trial-and-accept loop shared by every local-search optimizer.
///
/// The handler converts `(current_score, trial_score)` into an acceptance
/// probability (improving trials return `1.0` from the handler itself):
///
/// 1. `p <- handler.evaluate(current_score, trial_score)`
/// 2. accept if `p > rand(0, 1)`
///
/// At the start of every iteration the handler's [`TransitionHandler::update`]
/// is invoked so it can adapt its internal state (cooling schedule, water
/// level, …) before trials are evaluated.
///
/// The handler is supplied per-call via [`Self::step`] and
/// [`Self::optimize_with_handler`]; this loop stores no handler field.
pub struct LocalSearchLoop<ST: Ord + Sync + Send + Copy> {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    phantom: PhantomData<ST>,
}

impl<ST: Ord + Sync + Send + Copy> LocalSearchLoop<ST> {
    /// Constructor of `LocalSearchLoop`.
    ///
    /// - `patience` : the loop will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    pub fn new(patience: usize, n_trials: usize, return_iter: usize) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            phantom: PhantomData,
        }
    }

    /// Perform one optimization step (up to `n_iter` iterations or `time_limit`)
    /// with the supplied handler.
    ///
    /// Returns the [`StepResult`] alongside the (possibly mutated) handler so
    /// per-iteration state survives the call.
    #[allow(clippy::too_many_arguments)]
    pub fn step<M, H>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        mut handler: H,
    ) -> (StepResult<M::SolutionType, M::ScoreType>, H)
    where
        M: OptModel<ScoreType = ST>,
        H: TransitionHandler<ST>,
    {
        let start_time = Instant::now();
        let mut rng = rand::rng();
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut acceptance_counter = AcceptanceCounter::new(100);
        // Separate stagnation counters: one for triggering a return to best, one for early stopping (patience)
        let mut return_stagnation_counter = 0;
        let mut patience_stagnation_counter = 0;

        for it in 0..n_iter {
            // 1. Update time and iteration counters
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }

            // 2. Update handler state before trials, using the current best.
            let ctx = super::transition::UpdateCtx {
                iter: it,
                total: n_iter,
                acc: acceptance_counter.acceptance_ratio(),
                best: &best_score,
            };
            handler.update(&ctx);

            let (trial_solution, trial_score) = (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::rng();
                    let (solution, _, score) = model.generate_trial_solution(
                        current_solution.clone(),
                        current_score,
                        &mut rng,
                    );
                    (solution, score)
                })
                .min_by_key(|(_, score)| *score)
                .unwrap();

            // 3. Update best solution and score
            if trial_score < best_score {
                best_solution.replace(trial_solution.clone());
                best_score = trial_score;
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            } else {
                return_stagnation_counter += 1;
                patience_stagnation_counter += 1;
            }

            // 4. Update accepted counter and transitions
            let p = handler.evaluate(current_score, trial_score);
            let r: f64 = rng.random();
            let accepted = p > r;

            acceptance_counter.enqueue(accepted);

            // 5. Update current solution and score
            if accepted {
                current_solution = trial_solution;
                current_score = trial_score;
            }

            // 6. Check and handle return to best
            if return_stagnation_counter == self.return_iter {
                current_solution = best_solution.borrow().clone();
                current_score = best_score;
                return_stagnation_counter = 0;
            }

            // 7. Check patience
            if patience_stagnation_counter == self.patience {
                break;
            }

            // 8. Invoke callback
            let progress = OptProgress::new(
                it,
                acceptance_counter.acceptance_ratio(),
                best_solution.clone(),
                best_score,
            );
            callback(progress);
        }

        let best_solution = (*best_solution.borrow()).clone();
        let result = StepResult {
            best_solution,
            best_score,
            last_solution: current_solution,
            last_score: current_score,
            acceptance_counter,
        };
        (result, handler)
    }

    /// Run optimization with the supplied handler.
    ///
    /// Returns `(best_solution, best_score, handler)` where `handler` is the
    /// same instance passed in, with any per-iteration state mutations applied.
    #[allow(clippy::too_many_arguments)]
    pub fn optimize_with_handler<M, H>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        handler: H,
    ) -> (M::SolutionType, M::ScoreType, H)
    where
        M: OptModel<ScoreType = ST>,
        H: TransitionHandler<ST>,
    {
        let (result, handler) = self.step(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            handler,
        );
        (result.best_solution, result.best_score, handler)
    }
}
