use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{LocalSearchOptimizer, simulated_annealing::tune_cooling_rate};
use super::{SimulatedAnnealingOptimizer, simulated_annealing::tune_initial_temperature};

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
    pub fn tune_cooling_rate(self) -> Self {
        let cooling_rate =
            tune_cooling_rate(self.initial_temperature, 1e-2, self.reanneal_interval);

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
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        // Separate counters for return-to-best and patience
        let mut return_stagnation_counter = 0;
        let mut patience_stagnation_counter = 0;
        let mut iter = 0;
        let mut dummy_callback = &mut |_: OptProgress<M::SolutionType, M::ScoreType>| {};
        let mut accepted_counter = 0;

        while iter < n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }

            let sa = SimulatedAnnealingOptimizer::new(
                usize::MAX,
                self.n_trials,
                usize::MAX,
                self.initial_temperature,
                self.cooling_rate,
                1,
            );
            let step_result = sa.step(
                model,
                current_solution,
                current_score,
                self.reanneal_interval,
                time_limit,
                &mut dummy_callback,
            );

            //  Update current solution and score
            current_solution = step_result.last_solution;
            current_score = step_result.last_score;

            // Update best solution and score
            if step_result.best_score < best_score {
                best_solution.replace(step_result.best_solution.clone());
                best_score = step_result.best_score;
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            }

            // Update accepted counter
            let n_accepted = step_result.accepted_transitions.len();
            accepted_counter += n_accepted;

            // Update stagnation counters
            return_stagnation_counter += self.reanneal_interval;
            patience_stagnation_counter += self.reanneal_interval;

            // Check and handle return to best
            if return_stagnation_counter >= self.return_iter {
                current_solution = (*best_solution.borrow()).clone();
                current_score = best_score;
                return_stagnation_counter = 0;
            }

            // Check patience
            if patience_stagnation_counter >= self.patience {
                break;
            }

            // Update iteration counter
            iter += self.reanneal_interval;

            // Invoke callback
            let progress =
                OptProgress::new(iter, accepted_counter, best_solution.clone(), best_score);
            callback(progress);
        }

        let best_solution = (*best_solution.borrow()).clone();
        (best_solution, best_score)
    }
}
