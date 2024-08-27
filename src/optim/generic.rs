use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use rand::Rng;
use rayon::prelude::*;

use crate::{
    callback::{OptCallbackFn, OptProgress},
    Duration, Instant, OptModel,
};

use super::{LocalSearchOptimizer, TransitionProbabilityFn};

/// Optimizer that implements local search algorithm
/// Given a functin f that converts a float number to probability,
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
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut accepted_counter = 0;
        let mut counter = 0;

        for it in 0..n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }
            let (trial_solution, trial_score) = (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
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
            let r: f64 = rng.gen();

            if p > r {
                current_solution = trial_solution;
                current_score = trial_score;
                accepted_counter += 1;
            }

            if current_score < best_score {
                best_solution.replace(current_solution.clone());
                best_score = current_score;
                counter = 0;
            }

            counter += 1;

            if counter == self.return_iter {
                current_solution = best_solution.borrow().clone();
                current_score = best_score;
            }

            if counter == self.patience {
                break;
            }

            let progress =
                OptProgress::new(it, accepted_counter, best_solution.clone(), best_score);
            callback(progress);
        }

        let best_solution = (*best_solution.borrow()).clone();
        (best_solution, best_score)
    }
}
