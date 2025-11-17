use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::Rng;
use rayon::prelude::*;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::simulated_annealing::tune_initial_temperature;
use super::{LocalSearchOptimizer, simulated_annealing::tune_cooling_rate};

const MIN_TEMPERATURE: f64 = 0.01;

/// Adaptive simulated annealing optimizer that tunes temperature based on acceptance rate and re-anneals when stagnating.
/// Uses exponential cooling schedule that cools from initial_temperature to 0.01 over reanneal_interval steps.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct AdaptiveSimulatedAnnealingOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    initial_temperature: f64,
    cooling_rate: f64,
    reanneal_interval: usize,
}

impl AdaptiveSimulatedAnnealingOptimizer {
    /// Create a new adaptive simulated annealing optimizer.
    ///
    /// - `patience`: stop if no improvement is seen for this many iterations.
    /// - `n_trials`: number of trial solutions to evaluate per iteration.
    /// - `return_iter`: revert to the best solution when there is no improvement for this many iterations.
    /// - `initial_temperature`: starting temperature, clamped to the enforced minimum of 0.01.
    /// - `adapt_interval`: number of iterations between temperature adaptation steps.
    /// - `target_acceptance`: desired acceptance rate for uphill moves (between 0 and 1).
    /// - `adaptation_step`: relative adjustment applied when acceptance rate deviates from the target.
    /// - `reanneal_interval`: iterations without improvement before re-annealing; temperature cools exponentially to 0.01 over this many steps.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_temperature: f64,
        cooling_rate: f64,
        reanneal_interval: usize,
    ) -> Self {
        let initial_temperature = initial_temperature.max(MIN_TEMPERATURE);

        Self {
            patience,
            n_trials,
            return_iter,
            initial_temperature,
            cooling_rate,
            reanneal_interval: reanneal_interval.max(1),
        }
    }

    /// Tune the initial temperature based on acceptance rate from warm-up trials.
    pub fn tune_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
        target_initial_prob: f64,
    ) -> Self {
        let tuned_temperature = tune_initial_temperature(
            model,
            initial_solution_and_score,
            n_warmup,
            target_initial_prob,
        );

        Self {
            initial_temperature: tuned_temperature,
            ..self
        }
    }

    /// Tune cooling rate based on self.initial_temperature, final temperature of 1e-2
    pub fn tune_cooling_rate(self, n_iter: usize) -> Self {
        let cooling_rate = tune_cooling_rate(self.initial_temperature, 1e-2, n_iter);

        Self {
            cooling_rate,
            ..self
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M>
    for AdaptiveSimulatedAnnealingOptimizer
{
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
        let mut rng = rand::rng();
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut accepted_counter = 0;
        let mut stagnation_counter = 0;
        let mut step_counter = 0;
        let mut temperature = self.initial_temperature;
        let cooling_factor =
            (MIN_TEMPERATURE / self.initial_temperature).powf(1.0 / self.reanneal_interval as f64);

        for it in 0..n_iter {
            if Instant::now().duration_since(start_time) > time_limit {
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
            let mut accepted = false;

            if ds <= NotNan::new(0.0).unwrap() {
                accepted = true;
            } else {
                let p = (-ds / temperature.max(MIN_TEMPERATURE)).exp();
                if rng.random::<f64>() < p {
                    accepted = true;
                }
            }

            if accepted {
                current_solution = trial_solution;
                current_score = trial_score;
                accepted_counter += 1;
            }

            if current_score < best_score {
                best_solution.replace(current_solution.clone());
                best_score = current_score;
                stagnation_counter = 0;
            } else {
                stagnation_counter += 1;
            }

            temperature = (self.initial_temperature * cooling_factor.powf(step_counter as f64))
                .max(MIN_TEMPERATURE);

            step_counter += 1;

            if self.reanneal_interval > 0
                && stagnation_counter > 0
                && stagnation_counter % self.reanneal_interval == 0
            {
                current_solution = best_solution.borrow().clone();
                current_score = best_score;
                temperature = self.initial_temperature;
                step_counter = 0;
            }

            if stagnation_counter == self.return_iter {
                current_solution = best_solution.borrow().clone();
                current_score = best_score;
            }

            if stagnation_counter == self.patience {
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
