use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::Rng;
use rayon::prelude::*;

use crate::{
    callback::{OptCallbackFn, OptProgress},
    Duration, Instant, OptModel,
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
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
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
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `max_min_temperatures` : (max_temperature, min_temperature)
    fn optimize<F>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: Option<&F>,
        max_min_temperatures: Self::ExtraIn,
    ) -> (M::SolutionType, M::ScoreType, Self::ExtraOut)
    where
        F: OptCallbackFn<M::SolutionType, M::ScoreType>,
    {
        let start_time = Instant::now();
        let (max_temperature, min_temperature) = max_min_temperatures;
        let mut rng = rand::thread_rng();
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut accepted_counter = 0;
        let mut temperature = max_temperature;
        let t_factor = (min_temperature / max_temperature).ln();
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

            let ds = trial_score - current_score;
            let p = (-ds / temperature).exp();
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

            temperature = max_temperature * (t_factor * (it as f64 / n_iter as f64)).exp();

            counter += 1;
            if counter == self.patience {
                break;
            }

            if let Some(f) = callback {
                let progress =
                    OptProgress::new(it, accepted_counter, best_solution.clone(), best_score);
                f(progress);
            }
        }

        let best_solution = (*best_solution.borrow()).clone();
        (best_solution, best_score, ())
    }
}
