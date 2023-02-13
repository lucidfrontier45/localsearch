use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::Rng;
use rayon::prelude::*;

use crate::callback::{OptCallbackFn, OptProgress};
use crate::OptModel;

/// pre-defined functions to convert relative difference of scores to probability
pub mod relative_transition_score {
    /// exp(-w * d)
    pub fn exp(d: f64, w: f64) -> f64 {
        (-w * d).exp()
    }

    /// 2.0 / (1.0 + exp(w * d))
    pub fn logistic(d: f64, w: f64) -> f64 {
        2.0 / (1.0 + (w * d).exp())
    }
}

fn relative_difference(trial: f64, current: f64) -> f64 {
    (trial - current) / current
}

/// Optimizer that implements relative annealing algorithm
/// In this model, unlike simulated annealing, wether accept the trial state or not is calculated based on relative score difference
/// Given a functin f that converts a float number to probability, the actual procedure is as follows
///
/// 1. d <- (trial_score - current_score) / current_score
/// 2. p <- f(d)
/// 3. accept if p > rand(0, 1)
#[derive(Clone, Copy)]
pub struct RelativeAnnealingOptimizer<FS: Fn(f64) -> f64> {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    score_func: FS,
}

impl<FS: Fn(f64) -> f64> RelativeAnnealingOptimizer<FS> {
    /// Constructor of RelativeAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial states to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best state if there is no improvement after this number of iterations.
    /// - `score_func` : score function to calculate transition probability from relative difference.
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, score_func: FS) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            score_func,
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
        M: OptModel<ScoreType = NotNan<f64>> + Sync + Send,
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

            let ds = relative_difference(trial_score.into(), current_score.into());
            let p = (self.score_func)(ds);
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

#[cfg(test)]
mod test {
    use crate::optim::relative_annealing::relative_difference;

    use super::relative_transition_score;

    #[test]
    fn test_exp_transition_score() {
        let w = 1e1;
        let f = |ds| relative_transition_score::exp(ds, w);

        let p = f(relative_difference(0.9, 1.0));
        assert!(p >= 1.0);

        let p1 = f(relative_difference(1.1, 1.0));
        let p2 = f(relative_difference(1.2, 1.0));
        assert!(p1 > p2);
    }

    #[test]
    fn test_logistic_transition_score() {
        let w = 1e1;
        let f = |ds| relative_transition_score::logistic(ds, w);

        let p = f(relative_difference(0.9, 1.0));
        assert!(p >= 1.0);

        let p1 = f(relative_difference(1.1, 1.0));
        let p2 = f(relative_difference(1.2, 1.0));
        assert!(p1 > p2);
    }
}
