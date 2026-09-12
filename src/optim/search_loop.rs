use std::marker::PhantomData;

use rand::{RngExt as _, SeedableRng as _};
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

/// Outcome classification for an attempt at a new trial solution.
///
/// Adaptive generators (such as ALNS) use this classification to credit the
/// operators that produced the trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialOutcome {
    /// Trial produced a new global best solution and was accepted.
    NewBest,
    /// Trial improved over the current solution and was accepted, but did
    /// not improve the global best.
    Improved,
    /// Trial was accepted but did not improve the current solution.
    Accepted,
    /// Trial was rejected by the acceptance criterion.
    Rejected,
}

/// Pluggable trial-generation abstraction that decouples neighborhood
/// generation from the accept/reject loop.
///
/// A generator produces candidate solutions from the current one, then
/// receives a [`TrialOutcome`] for the winning candidate via
/// [`TrialGenerator::feedback`] after the loop has decided what to do with
/// it. Adaptive generators (e.g. ALNS) use that feedback to update their
/// internal state — operator weights, scores, or anything else.
///
/// [`TrialGenerator::generate_trial`] takes `&self`, so the search loop can
/// invoke it from multiple rayon worker threads in parallel. The returned
/// [`TrialGenerator::Token`] identifies which operators produced the trial;
/// the loop hands the winner's token back to [`TrialGenerator::feedback`]
/// (winner-takes-all) while losers are discarded without reward.
///
/// The default behavior of [`OptModel::generate_trial_solution`] is
/// recovered by [`DefaultTrialGenerator`], so existing optimizers keep
/// their previous semantics.
pub trait TrialGenerator<M: OptModel> {
    /// Identifies the operators that produced a trial. The loop keeps each
    /// candidate's token and hands only the winner's token to
    /// [`TrialGenerator::feedback`]. Must be `Send` so candidates can be
    /// generated on rayon worker threads.
    type Token: Send;

    /// Generate a single trial solution, its score, and its token from the
    /// current solution. Called concurrently from rayon worker threads —
    /// implementations must be `Sync` and use only the supplied `rng`
    /// (a per-trial fork) for randomness.
    fn generate_trial(
        &self,
        model: &M,
        current_solution: &M::SolutionType,
        current_score: M::ScoreType,
        rng: &mut rand::rngs::StdRng,
    ) -> (M::SolutionType, M::ScoreType, Self::Token);

    /// Notify the generator about the outcome of the winning trial.
    /// Generators without adaptive state can ignore this call.
    fn feedback(&mut self, token: Self::Token, outcome: TrialOutcome);
}

/// Default trial generator that simply delegates to
/// [`OptModel::generate_trial_solution`].
///
/// This keeps the historical behavior of [`LocalSearchLoop::step`]: every
/// iteration's `n_trials` candidates come straight from the model's own
/// neighborhood generator and `feedback` is a no-op.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultTrialGenerator;

impl<M: OptModel> TrialGenerator<M> for DefaultTrialGenerator {
    type Token = ();

    fn generate_trial(
        &self,
        model: &M,
        current_solution: &M::SolutionType,
        current_score: M::ScoreType,
        rng: &mut rand::rngs::StdRng,
    ) -> (M::SolutionType, M::ScoreType, Self::Token) {
        let (solution, _transition, score) =
            model.generate_trial_solution(current_solution.clone(), current_score, rng);
        (solution, score, ())
    }

    fn feedback(&mut self, _token: Self::Token, _outcome: TrialOutcome) {}
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
/// The handler is supplied per-call via [`Self::step`];
/// this loop stores no handler field.
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
    /// per-iteration state survives the call. Trial generation goes through the
    /// built-in [`DefaultTrialGenerator`]; pass a custom generator via
    /// [`Self::step_with_generator`] to use ALNS or any other adaptive scheme.
    #[allow(clippy::too_many_arguments)]
    pub fn step<M, H>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        handler: H,
    ) -> (StepResult<M::SolutionType, M::ScoreType>, H)
    where
        M: OptModel<ScoreType = ST>,
        H: TransitionHandler<ST>,
    {
        let (result, handler, _generator) = self.step_with_generator(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            handler,
            DefaultTrialGenerator,
        );
        (result, handler)
    }

    /// Perform one optimization step that drives trial generation through a
    /// caller-supplied [`TrialGenerator`].
    ///
    /// The generator is invoked `n_trials` times per iteration, each call
    /// produces one candidate, and the best candidate is fed through the
    /// acceptance machinery. After the trial's outcome is known, the generator
    /// receives a [`TrialOutcome`] via [`TrialGenerator::feedback`] so adaptive
    /// schemes (ALNS, hyper-heuristics, …) can update their internal state.
    ///
    /// Returns the [`StepResult`], the (possibly mutated) handler, and the
    /// (possibly mutated) generator — so callers can inspect or reuse their
    /// adapted state after the step.
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_generator<M, H, G>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        handler: H,
        generator: G,
    ) -> (StepResult<M::SolutionType, M::ScoreType>, H, G)
    where
        M: OptModel<ScoreType = ST>,
        H: TransitionHandler<ST>,
        G: TrialGenerator<M> + Sync,
    {
        let start_time = Instant::now();
        let mut rng = rand::rng();
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;
        let mut acceptance_counter = AcceptanceCounter::new(100);
        // Separate stagnation counters: one for triggering a return to best, one for early stopping (patience)
        let mut return_stagnation_counter = 0;
        let mut patience_stagnation_counter = 0;
        let mut handler = handler;
        let mut generator = generator;

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

            // 3. Generate `n_trials` candidates through the supplied generator
            //    and keep the best-scoring one. Generation runs in parallel
            //    via rayon; each trial draws from a per-trial RNG fork seeded
            //    sequentially, so runs stay reproducible. Only the winner is
            //    evaluated and fed back (winner-takes-all).
            assert!(self.n_trials > 0, "n_trials must be at least 1");
            let seeds: Vec<u64> = (0..self.n_trials).map(|_| rng.random()).collect();
            let (trial_solution, trial_score, winner_token) = seeds
                .into_par_iter()
                .map(|seed| {
                    let mut local_rng = rand::rngs::StdRng::seed_from_u64(seed);
                    generator.generate_trial(
                        model,
                        &current_solution,
                        current_score,
                        &mut local_rng,
                    )
                })
                .min_by_key(|(_, score, _)| *score)
                .expect("n_trials must be at least 1");

            // 4. Classify the trial outcome and apply best-score bookkeeping.
            //    `previous_best` is captured before any updates so that the
            //    "new best" credit is given to the operators that actually
            //    produced the new global best.
            let previous_best = best_score;
            if trial_score < best_score {
                best_solution = trial_solution.clone();
                best_score = trial_score;
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            } else {
                return_stagnation_counter += 1;
                patience_stagnation_counter += 1;
            }

            // 5. Acceptance.
            let p = handler.evaluate(current_score, trial_score);
            let r: f64 = rng.random();
            let accepted = p > r;
            acceptance_counter.enqueue(accepted);

            // 6. Tell the generator what happened so adaptive schemes can
            //    update their internal state.
            let outcome = if accepted && trial_score < previous_best {
                TrialOutcome::NewBest
            } else if accepted && trial_score < current_score {
                TrialOutcome::Improved
            } else if accepted {
                TrialOutcome::Accepted
            } else {
                TrialOutcome::Rejected
            };
            generator.feedback(winner_token, outcome);

            // 7. Update current solution and score.
            if accepted {
                current_solution = trial_solution;
                current_score = trial_score;
            }

            // 8. Check and handle return to best.
            if return_stagnation_counter == self.return_iter {
                current_solution = best_solution.clone();
                current_score = best_score;
                return_stagnation_counter = 0;
            }

            // 9. Check patience.
            if patience_stagnation_counter == self.patience {
                break;
            }

            // 10. Invoke callback.
            let progress = OptProgress::new(
                it,
                acceptance_counter.acceptance_ratio(),
                std::rc::Rc::new(std::cell::RefCell::new(best_solution.clone())),
                best_score,
            );
            callback(progress);
        }

        let result = StepResult {
            best_solution,
            best_score,
            last_solution: current_solution,
            last_score: current_score,
            acceptance_counter,
        };
        (result, handler, generator)
    }
}
