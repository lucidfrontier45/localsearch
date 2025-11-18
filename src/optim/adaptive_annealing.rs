use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{LocalSearchOptimizer, MetropolisOptimizer, simulated_annealing::tune_temperature};

const MIN_TEMPERATURE: f64 = 0.01;

/// Adaptive simulated annealing optimizer that adjusts temperature based on acceptance rate.
/// Temperature updates continuously using T_new = T_current * exp(eta * (target_acc - acc))
/// to keep the acceptance rate close to target_acc.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct AdaptiveAnnealingOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    initial_temperature: f64,
    update_frequency: usize,
    eta: f64,
    target_acc: f64,
}

impl AdaptiveAnnealingOptimizer {
    /// Create a new adaptive simulated annealing optimizer.
    ///
    /// - `patience`: stop if no improvement is seen for this many iterations.
    /// - `n_trials`: number of trial solutions to evaluate per iteration.
    /// - `return_iter`: revert to the best solution when there is no improvement for this many iterations.
    /// - `initial_temperature`: starting temperature, clamped to the enforced minimum of 0.01.
    /// - `update_frequency`: number of iterations between temperature adaptation steps.
    /// - `eta`: hyperparameter controlling adaptation sensitivity (must be > 0).
    /// - `target_acc`: target acceptance rate for uphill moves (between 0 and 1).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_temperature: f64,
        update_frequency: usize,
        eta: f64,
        target_acc: f64,
    ) -> Self {
        assert!(eta > 0.0, "eta must be positive");
        assert!(
            (0.0..=1.0).contains(&target_acc),
            "target_acc must be between 0 and 1"
        );
        let initial_temperature = initial_temperature.max(MIN_TEMPERATURE);

        Self {
            patience,
            n_trials,
            return_iter,
            initial_temperature,
            update_frequency: update_frequency.max(1),
            eta,
            target_acc,
        }
    }

    /// Tune the initial temperature based on acceptance rate from warm-up trials.
    pub fn tune_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
    ) -> Self {
        let tuned_temperature =
            tune_temperature(model, initial_solution_and_score, n_warmup, self.target_acc);

        Self {
            initial_temperature: tuned_temperature,
            ..self
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for AdaptiveAnnealingOptimizer {
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
        let mut current_temperature = self.initial_temperature;
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

            let metropolis = MetropolisOptimizer::new(
                usize::MAX,
                self.n_trials,
                usize::MAX,
                current_temperature,
            );
            let step_result = metropolis.step(
                model,
                current_solution,
                current_score,
                self.update_frequency,
                time_limit,
                &mut dummy_callback,
            );

            // 1. Update time and iteration counters
            iter += self.update_frequency;

            // 2. Update best solution and score
            if step_result.best_score < best_score {
                best_solution.replace(step_result.best_solution.clone());
                best_score = step_result.best_score;
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            } else {
                return_stagnation_counter += self.update_frequency;
                patience_stagnation_counter += self.update_frequency;
            }

            // 3. Update accepted counter
            let n_accepted = step_result.accepted_transitions.len();
            accepted_counter += n_accepted;

            // 4. Update current solution and score
            current_solution = step_result.last_solution;
            current_score = step_result.last_score;

            // 5. Check and handle return to best
            if return_stagnation_counter >= self.return_iter {
                current_solution = (*best_solution.borrow()).clone();
                current_score = best_score;
                return_stagnation_counter = 0;
            }

            // 6. Check patience
            if patience_stagnation_counter >= self.patience {
                break;
            }

            // 7. Update temperature adaptively
            let acc = n_accepted as f64 / self.update_frequency as f64;
            current_temperature *= (self.eta * (self.target_acc - acc)).exp();
            current_temperature = current_temperature.max(MIN_TEMPERATURE);

            // 8. Invoke callback
            let progress =
                OptProgress::new(iter, accepted_counter, best_solution.clone(), best_score);
            callback(progress);
        }

        let best_solution = (*best_solution.borrow()).clone();
        (best_solution, best_score)
    }
}
