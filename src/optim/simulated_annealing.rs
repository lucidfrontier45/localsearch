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
    max_temperature: f64,
    min_temperature: f64,
}

impl SimulatedAnnealingOptimizer {
    /// Constructor of SimulatedAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `max_temperature` : maximum temperature
    /// - `min_temperature` : minimum temperature
    pub fn new(
        patience: usize,
        n_trials: usize,
        max_temperature: f64,
        min_temperature: f64,
    ) -> Self {
        Self {
            patience,
            n_trials,
            max_temperature,
            min_temperature,
        }
    }
}

impl SimulatedAnnealingOptimizer {
    /// Start optimization with given temperature range
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `max_temperature` : maximum temperature
    /// - `min_temperature` : minimum temperature
    fn optimize_with_temperature<M, F>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: Option<&F>,
        max_temperature: f64,
        min_temperature: f64,
    ) -> (M::SolutionType, M::ScoreType)
    where
        M: OptModel<ScoreType = NotNan<f64>>,
        F: OptCallbackFn<M::SolutionType, M::ScoreType>,
    {
        let start_time = Instant::now();
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
        (best_solution, best_score)
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for SimulatedAnnealingOptimizer {
    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    fn optimize<F>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: Option<&F>,
    ) -> (M::SolutionType, M::ScoreType)
    where
        F: OptCallbackFn<M::SolutionType, M::ScoreType>,
    {
        self.optimize_with_temperature(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            self.max_temperature,
            self.min_temperature,
        )
    }
}
