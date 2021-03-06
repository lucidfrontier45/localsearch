use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rayon::prelude::*;

use crate::OptModel;

#[derive(Clone, Copy)]
pub struct HillClimbingOptimizer {
    patience: usize,
    n_trials: usize,
}

impl HillClimbingOptimizer {
    pub fn new(patience: usize, n_trials: usize) -> Self {
        Self { patience, n_trials }
    }

    pub fn optimize<S, T, M, F>(
        &self,
        model: &M,
        initial_state: Option<S>,
        n_iter: usize,
        callback: Option<&F>,
    ) -> (S, f64)
    where
        M: OptModel<S, T> + Sync + Send,
        S: Clone + Sync + Send,
        F: Fn(usize, Rc<RefCell<S>>, f64),
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
        for it in 0..n_iter {
            let (trial_state, trial_score) = (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    let (state, _, score) =
                        model.generate_trial_state(&current_state, &mut rng, Some(current_score));
                    (state, score)
                })
                .min_by_key(|(_, score)| NotNan::new(*score).unwrap())
                .unwrap();
            // .sort_unstable_by_key(|(_, _, score)| NotNan::new(*score).unwrap());
            if trial_score < current_score {
                current_state = trial_state;
                current_score = trial_score;
                best_state.replace(current_state.clone());
                best_score = current_score;
                counter = 0;
            } else {
                counter += 1;
                if counter >= self.patience {
                    break;
                }
            }

            if let Some(f) = callback {
                f(it, best_state.clone(), best_score);
            }
        }

        (current_state, current_score)
    }
}
