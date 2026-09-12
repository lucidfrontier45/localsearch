use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use rand::RngExt as _;
use rayon::prelude::*;

use super::{LocalSearchOptimizer, TransitionProbabilityFn};
use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
    counter::AcceptanceCounter,
};

/// Result of an optimization step, containing information about the best and last solutions,
/// as well as the acceptance counter for the step.
///
/// The `O` type parameter carries algorithm-specific output (for example, ALNS operator
/// selection statistics). Defaults to `()` so existing callers using
/// `StepResult<S, ST>` keep working unchanged.
pub struct StepResult<S, ST, O = ()> {
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
    /// Algorithm-specific output for this step (e.g., ALNS operator statistics).
    pub output: O,
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
    /// - `initial_solution` : the initial solution to start optimization
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    #[allow(clippy::too_many_arguments)]
    pub fn step<M: OptModel<ScoreType = ST>>(
        &self,
        model: &M,
        state: &M::StateType,
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

            let (trial_solution, trial_score) = (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::rng();
                    let (solution, _, score) = model.generate_trial_solution(
                        state,
                        current_solution.clone(),
                        current_score,
                        &mut rng,
                    );
                    (solution, score)
                })
                .min_by_key(|(_, score)| *score)
                .unwrap();

            // 2. Update best solution and score
            if trial_score < best_score {
                best_solution.replace(trial_solution.clone());
                best_score = trial_score;
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            } else {
                return_stagnation_counter += 1;
                patience_stagnation_counter += 1;
            }

            // 3. Update accepted counter and transitions
            let accepted = if trial_score < current_score {
                true
            } else {
                let p = (self.score_func)(current_score, trial_score);
                let r: f64 = rng.random();
                p > r
            };

            acceptance_counter.enqueue(accepted);

            // 4. Update current solution and score
            if accepted {
                current_solution = trial_solution;
                current_score = trial_score;
            }

            // 5. Check and handle return to best
            if return_stagnation_counter == self.return_iter {
                current_solution = best_solution.borrow().clone();
                current_score = best_score;
                return_stagnation_counter = 0;
            }

            // 6. Check patience
            if patience_stagnation_counter == self.patience {
                break;
            }

            // 7. Update algorithm-specific state (none)

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
        StepResult {
            best_solution,
            best_score,
            last_solution: current_solution,
            last_score: current_score,
            acceptance_counter,
            output: (),
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
    /// - `initial_solution` : the initial solution to start optimization
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    fn optimize(
        &self,
        model: &M,
        state: &M::StateType,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType) {
        let step_result = self.step(
            model,
            state,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
        );
        (step_result.best_solution, step_result.best_score)
    }
}


#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use super::StepResult;
    use crate::counter::AcceptanceCounter;

    #[test]
    fn step_result_with_custom_output_constructs() {
        #[derive(Debug, PartialEq)]
        struct MyOut {
            used: usize,
            improved: bool,
        }

        let result: StepResult<Vec<f64>, NotNan<f64>, MyOut> = StepResult {
            best_solution: vec![1.0, 2.0],
            best_score: NotNan::new(0.0).unwrap(),
            last_solution: vec![1.0, 2.0],
            last_score: NotNan::new(0.0).unwrap(),
            acceptance_counter: AcceptanceCounter::new(10),
            output: MyOut {
                used: 3,
                improved: true,
            },
        };

        assert_eq!(result.output.used, 3);
        assert!(result.output.improved);
    }

    #[test]
    fn step_result_struct_update_into_custom_output() {
        let base: StepResult<Vec<f64>, NotNan<f64>> = StepResult {
            best_solution: vec![1.0, 2.0],
            best_score: NotNan::new(0.5).unwrap(),
            last_solution: vec![1.0, 2.0],
            last_score: NotNan::new(0.5).unwrap(),
            acceptance_counter: AcceptanceCounter::new(10),
            output: (),
        };

        // Struct-update ergonomics for a new `O`: copy existing fields, then assign
        // a new `output`. Stable Rust allows this when fields (not `O`) line up.
        let with_output: StepResult<Vec<f64>, NotNan<f64>, Vec<u32>> = StepResult {
            best_solution: base.best_solution,
            best_score: base.best_score,
            last_solution: base.last_solution,
            last_score: base.last_score,
            acceptance_counter: base.acceptance_counter,
            output: vec![1, 2, 3],
        };

        assert_eq!(with_output.output, vec![1, 2, 3]);
        assert_eq!(with_output.best_score, NotNan::new(0.5).unwrap());
    }
}
