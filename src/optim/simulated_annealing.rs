use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::Rng;
use rayon::prelude::*;

use crate::{
    callback::{OptCallbackFn, OptProgress},
    OptModel,
};

use super::LocalSearchOptimizer;

/// Optimizer that implements the simulated annealing algorithm
#[derive(Clone, Copy)]
pub struct SimulatedAnnealingOptimizer {
    patience: usize,
    n_trials: usize,
}

impl SimulatedAnnealingOptimizer {
    /// Constructor of SimulatedAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial states to generate and evaluate at each iteration
    pub fn new(patience: usize, n_trials: usize) -> Self {
        Self { patience, n_trials }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for SimulatedAnnealingOptimizer {
    /// max temperature, min temperature
    type ExtraIn = (f64, f64);
    type ExtraOut = ();

    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_state` : the initial state to start optimization. If None, a random state will be generated.
    /// - `n_iter`: maximum iterations
    /// - `max_temperature` : the initial temperature at the begining of the optimization
    /// - `min_temperature` : the final temperature at the end of the optimization
    /// - `callback` : callback function that will be invoked at the end of each iteration
    fn optimize<F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        callback: Option<&F>,
        max_min_temperatures: Self::ExtraIn,
    ) -> (M::StateType, M::ScoreType, Self::ExtraOut)
    where
        F: OptCallbackFn<M::StateType, M::ScoreType>,
    {
        let (max_temperature, min_temperature) = max_min_temperatures;
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
        let mut temperature = max_temperature;
        let t_factor = (min_temperature / max_temperature).ln();
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

            let ds = trial_score - current_score;
            let p = (-ds / temperature).exp();
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

            temperature = max_temperature * (t_factor * (it as f64 / n_iter as f64)).exp();

            counter += 1;
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
        (best_state, best_score, ())
    }
}
