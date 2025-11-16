use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::Rng;
use rayon::prelude::*;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::LocalSearchOptimizer;

/// Tune cooling rate based on initial and final temperatures and number of iterations
/// initial temperature will be cooled to final temperature after n_iter iterations
/// - `initial_temperature` : initial temperature
/// - `final_temperature` : final temperature
/// - `n_iter` : number of iterations
/// - returns : cooling rate
pub fn tune_cooling_rate(initial_temperature: f64, final_temperature: f64, n_iter: usize) -> f64 {
    (final_temperature / initial_temperature).powf(1.0 / n_iter as f64)
}

pub fn tune_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
    model: &M,
    initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
    n_warmup: usize,
    target_initial_prob: f64,
) -> f64 {
    // 1. First run n_warmup completely random iterations from initial_solution
    // 2. calculate initial_temperature so that the average acceptance probability of sampled trial solutions are target_initial_prob
    // 3. return new SimulatedAnnealingOptimizer with updated temperatures
    let mut rng = rand::rng();
    let (mut current_solution, mut current_score) =
        initial_solution_and_score.unwrap_or(model.generate_random_solution(&mut rng).unwrap());

    let mut energy_diffs = Vec::new();

    for _ in 0..n_warmup {
        let (trial_solution, _, trial_score) =
            model.generate_trial_solution(current_solution.clone(), current_score, &mut rng);
        let ds = trial_score - current_score;
        if ds <= NotNan::new(0.0).unwrap() {
            continue;
        }
        energy_diffs.push(ds.into_inner());
        current_solution = trial_solution;
        current_score = trial_score;
    }

    // Calculate initial_temperature based on target_initial_prob
    // p = exp(-ds / T) => T = -ds / ln(p)
    // Average across all energy differences

    if energy_diffs.is_empty() {
        1.0
    } else {
        let avg_energy_diff = energy_diffs.iter().sum::<f64>() / energy_diffs.len() as f64;
        let ln_prob = target_initial_prob.ln().clamp(-100.0, -0.01);
        (-avg_energy_diff / ln_prob).max(1.0)
    }
}

/// Optimizer that implements the simulated annealing algorithm
#[derive(Clone, Copy)]
pub struct SimulatedAnnealingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    pub patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    pub n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    pub return_iter: usize,
    /// Initial temperature
    pub initial_temperature: f64,
    /// Cooling rate
    pub cooling_rate: f64,
}

impl SimulatedAnnealingOptimizer {
    /// Constructor of SimulatedAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the best solution if there is no improvement after this number of iterations.
    /// - `initial_temperature` : initial temperature
    /// - `cooling_rate` : cooling rate
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_temperature: f64,
        cooling_rate: f64,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            initial_temperature,
            cooling_rate,
        }
    }
}

impl SimulatedAnnealingOptimizer {
    /// Tune temperature parameters based on initial random trials
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `n_warmup` : number of warmup iterations to run
    /// - `target_initial_prob` : target acceptance probability for uphill moves at the beginning
    pub fn tune_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
        target_initial_prob: f64,
    ) -> SimulatedAnnealingOptimizer {
        let tuned_temperature =
            tune_temperature(model, initial_solution, n_warmup, target_initial_prob);

        SimulatedAnnealingOptimizer {
            initial_temperature: tuned_temperature,
            ..self
        }
    }

    /// Tune cooling rate based on self.initial_temperature, final temperature of 1e-2
    pub fn tune_cooling_rate(self, n_iter: usize) -> SimulatedAnnealingOptimizer {
        let cooling_rate = tune_cooling_rate(self.initial_temperature, 1e-2, n_iter);

        SimulatedAnnealingOptimizer {
            cooling_rate,
            ..self
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Start optimization with given temperature range
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `initial_temperature` : initial temperature
    /// - `cooling_rate` : cooling rate
    pub fn optimize_with_temperature_and_cooling_rate<M: OptModel<ScoreType = NotNan<f64>>>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        initial_temperature: f64,
        cooling_rate: f64,
    ) -> (M::SolutionType, M::ScoreType, f64) {
        let start_time = Instant::now();
        let mut rng = rand::rng();
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut accepted_counter = 0;
        let mut temperature = initial_temperature;
        let mut counter = 0;

        for it in 0..n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }
            let (trial_solution, trial_score) = (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::rng();
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
            let r: f64 = rng.random();

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

            temperature *= cooling_rate;

            let progress =
                OptProgress::new(it, accepted_counter, best_solution.clone(), best_score);
            callback(progress);
        }

        let best_solution = (*best_solution.borrow()).clone();
        (best_solution, best_score, temperature)
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
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType) {
        let (best_solution, best_score, _final_temperature) = self
            .optimize_with_temperature_and_cooling_rate(
                model,
                initial_solution,
                initial_score,
                n_iter,
                time_limit,
                callback,
                self.initial_temperature,
                self.cooling_rate,
            );
        (best_solution, best_score)
    }
}
