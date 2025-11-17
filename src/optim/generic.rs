use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use rand::Rng;
use rayon::prelude::*;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{LocalSearchOptimizer, TransitionProbabilityFn};

/// Result of an optimization step, containing information about the best and last solutions,
/// as well as the transitions that were accepted or rejected during the step.
///
/// The transition tuples represent `(from_score, to_score)` pairs, indicating the score before
/// and after a proposed move. Accepted transitions are those that were applied; rejected transitions
/// are those that were considered but not applied.
pub struct StepResult<S, ST> {
    /// The best solution found during this step.
    pub best_solution: S,
    /// The score of the best solution found during this step.
    pub best_score: ST,
    /// The last solution at the end of this step (may differ from the best).
    pub last_solution: S,
    /// The score of the last solution at the end of this step.
    pub last_score: ST,
    /// List of accepted transitions as `(from_score, to_score)` pairs.
    pub accepted_transitions: Vec<(ST, ST)>,
    /// List of rejected transitions as `(from_score, to_score)` pairs.
    pub rejected_transitions: Vec<(ST, ST)>,
}

/// Optimizer that implements local search algorithm
/// Given a function f that converts a float number to probability,
/// the trial solution is accepted by the following procedure
///
/// 1. p <- f(current_score, trial_score)
/// 2. accept if p > rand(0, 1)
#[derive(Clone, Copy)]
pub struct GenericLocalSearchOptimizer<
    ST: Ord + Sync + Send + Copy,
    FT: TransitionProbabilityFn<ST>,
> {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    score_func: FT,
    phantom: PhantomData<ST>,
}

impl<ST: Ord + Sync + Send + Copy, FT: TransitionProbabilityFn<ST>>
    GenericLocalSearchOptimizer<ST, FT>
{
    /// Constructor of BaseLocalSearchOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `score_func` : score function to calculate transition probability.
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, score_func: FT) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            score_func,
            phantom: PhantomData,
        }
    }

    /// Start optimization, returns the best solution and last solution
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    pub fn step<M: OptModel<ScoreType = ST>>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> StepResult<M::SolutionType, M::ScoreType> {
        let start_time = Instant::now();
        let mut rng = rand::rng();
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut accepted_counter = 0;
        // Separate stagnation counters: one for triggering a return to best, one for early stopping (patience)
        let mut return_stagnation_counter = 0;
        let mut patience_stagnation_counter = 0;

        let mut accepted_transitions = Vec::with_capacity(n_iter);
        let mut rejected_transitions = Vec::with_capacity(n_iter);

        for it in 0..n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }
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

            let p = (self.score_func)(current_score, trial_score);
            let r: f64 = rng.random();

            if p > r {
                // Update accepted counter and transitions
                accepted_transitions.push((current_score, trial_score));
                accepted_counter += 1;

                // Update current solution and score
                current_solution = trial_solution;
                current_score = trial_score;
            } else {
                rejected_transitions.push((current_score, trial_score));
            }

            // Update best solution and score
            // Reset stagnation counters if improved
            if current_score < best_score {
                best_solution.replace(current_solution.clone());
                best_score = current_score;
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            }

            // Update stagnation counters
            return_stagnation_counter += 1;
            patience_stagnation_counter += 1;

            // Check and handle return to best
            if return_stagnation_counter == self.return_iter {
                current_solution = best_solution.borrow().clone();
                current_score = best_score;
                return_stagnation_counter = 0;
            }

            // Check patience
            if patience_stagnation_counter == self.patience {
                break;
            }

            // Invoke callback
            let progress =
                OptProgress::new(it, accepted_counter, best_solution.clone(), best_score);
            callback(progress);
        }

        let best_solution = (*best_solution.borrow()).clone();
        StepResult {
            best_solution,
            best_score,
            last_solution: current_solution,
            last_score: current_score,
            accepted_transitions,
            rejected_transitions,
        }
    }
}

impl<ST, FT, M> LocalSearchOptimizer<M> for GenericLocalSearchOptimizer<ST, FT>
where
    ST: Ord + Sync + Send + Copy,
    FT: TransitionProbabilityFn<ST>,
    M: OptModel<ScoreType = ST>,
{
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
        let step_result = self.step(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
        );
        (step_result.best_solution, step_result.best_score)
    }
}
