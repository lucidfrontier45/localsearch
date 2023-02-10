use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::Rng;
use rayon::prelude::*;

use crate::OptModel;

use super::callback::{OptCallbackFn, OptProgress};

#[derive(Clone, Copy)]
pub struct SimulatedAnnealingOptimizer {
    patience: usize,
    n_trials: usize,
}

impl SimulatedAnnealingOptimizer {
    pub fn new(patience: usize, n_trials: usize) -> Self {
        Self { patience, n_trials }
    }

    pub fn optimize<M, F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        max_temperature: f64,
        min_temperature: f64,
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
        let mut temperature = max_temperature;
        let t_factor = (min_temperature / max_temperature).ln();

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
            // .sort_unstable_by_key(|(_, _, score)| NotNan::new(*score).unwrap());
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
            }

            temperature = max_temperature * (t_factor * (it as f64 / n_iter as f64)).exp();

            if let Some(f) = callback {
                let progress =
                    OptProgress::new(it, accepted_counter, best_state.clone(), best_score);
                f(progress);
            }
        }

        (current_state, current_score)
    }
}
