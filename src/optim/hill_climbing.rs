use std::{cell::RefCell, rc::Rc};

use rayon::prelude::*;

use crate::callback::{OptCallbackFn, OptProgress};
use crate::OptModel;

/// Optimizer that implements simple hill climbing algorithm
#[derive(Clone, Copy)]
pub struct HillClimbingOptimizer {
    patience: usize,
    n_trials: usize,
}

impl HillClimbingOptimizer {
    /// Constructor of HillClimbingOPtimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial states to generate and evaluate at each iteration
    pub fn new(patience: usize, n_trials: usize) -> Self {
        Self { patience, n_trials }
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
        M: OptModel + Sync + Send,
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
        let mut counter = 0;
        let mut accepted_counter = 0;
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

            if trial_score < current_score {
                current_state = trial_state;
                current_score = trial_score;
                best_state.replace(current_state.clone());
                best_score = current_score;
                counter = 0;
                accepted_counter += 1;
            } else {
                counter += 1;
                if counter >= self.patience {
                    break;
                }
            }

            if let Some(f) = callback {
                let progress =
                    OptProgress::new(it, accepted_counter, best_state.clone(), best_score);
                f(progress);
            }
        }

        (current_state, current_score)
    }
}
