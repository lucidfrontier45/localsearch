use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use rand::Rng;
use rayon::prelude::*;
use trait_set::trait_set;

use crate::{
    callback::{OptCallbackFn, OptProgress},
    OptModel,
};

trait_set! {
    pub trait TransitionProbabilityFn<ST: Ord + Sync + Send + Copy> = Fn(ST, ST) -> f64;
}

/// Optimizer that implements local search algorithm
/// Given a functin f that converts a float number to probability,
/// the trial state is accepted by the following procedure
///
/// 1. p <- f(current_score, trial_score)
/// 2. accept if p > rand(0, 1)
#[derive(Clone, Copy)]
pub struct BaseLocalSearchOptimizer<ST: Ord + Sync + Send + Copy, FT: TransitionProbabilityFn<ST>> {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    score_func: FT,
    phantom: PhantomData<ST>,
}

impl<ST: Ord + Sync + Send + Copy, FT: TransitionProbabilityFn<ST>>
    BaseLocalSearchOptimizer<ST, FT>
{
    /// Constructor of BaseLocalSearchOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial states to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best state if there is no improvement after this number of iterations.
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

    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_state` : the initial state to start optimization. If None, a random state will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    pub fn optimize<M, F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        callback: Option<&F>,
    ) -> (M::StateType, M::ScoreType)
    where
        M: OptModel<ScoreType = ST> + Sync + Send,
        F: OptCallbackFn<M::StateType, M::ScoreType>,
    {
        let mut rng = rand::thread_rng();
        let mut current_state = if let Some(s) = initial_state {
            s
        } else {
            model.generate_random_state(&mut rng).unwrap()
        };
        let mut current_score = model.evaluate_state(&current_state);
        let best_state = Rc::new(RefCell::new(current_state.clone()));
        let mut best_score = current_score;
        let mut accepted_counter = 0;
        let mut counter = 0;

        for it in 0..n_iter {
            let (trial_state, trial_score) = (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    let (state, _, score) =
                        model.generate_trial_state(&current_state, &mut rng, Some(current_score));
                    (state, score)
                })
                .min_by_key(|(_, score)| *score)
                .unwrap();

            let p = (self.score_func)(current_score, trial_score);
            let r: f64 = rng.gen();

            if p > r {
                current_state = trial_state;
                current_score = trial_score;
                accepted_counter += 1;
            }

            if current_score < best_score {
                best_state.replace(current_state.clone());
                best_score = current_score;
                counter = 0;
            }

            counter += 1;

            if counter == self.return_iter {
                current_state = best_state.borrow().clone();
                current_score = best_score;
            }

            if counter == self.patience {
                break;
            }

            if let Some(f) = callback {
                let progress =
                    OptProgress::new(it, accepted_counter, best_state.clone(), best_score);
                f(progress);
            }
        }

        let best_state = (*best_state.borrow()).clone();
        (best_state, best_score)
    }
}
