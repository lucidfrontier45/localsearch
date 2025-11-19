use std::{cell::RefCell, f64::consts::PI, rc::Rc};

use ordered_float::NotNan;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{
    LocalSearchOptimizer, MetropolisOptimizer, generic::StepResult,
    simulated_annealing::tune_temperature,
};

#[derive(Clone, Copy, Debug, Default)]
/// Target acceptance rate scheduling mode
pub enum TargetAccScheduleMode {
    /// Linearly cool from initial_target_acc to final_target_acc
    Linear,
    /// Exponentially cool from initial_target_acc to final_target_acc
    Exponential,
    /// Cosine schedule from initial_target_acc to final_target_acc
    #[default]
    Cosine,
}

/// Scheduler for adaptive annealing optimizer
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveScheduler {
    initial_target_acc: f64,
    final_target_acc: f64,
    schedule_mode: TargetAccScheduleMode,
    // speed parameter for temperature update
    gamma: f64,
}

impl Default for AdaptiveScheduler {
    fn default() -> Self {
        Self {
            initial_target_acc: 0.5,
            final_target_acc: 0.05,
            schedule_mode: TargetAccScheduleMode::Cosine,
            gamma: 0.05,
        }
    }
}

impl AdaptiveScheduler {
    /// Creates a new `AdaptiveScheduler` instance with the specified parameters.
    /// # Arguments
    /// * `initial_target_acc` - The initial target acceptance rate.
    /// * `final_target_acc` - The final target acceptance rate.
    /// * `schedule_mode` - The scheduling mode for target acceptance rate.
    /// * `gamma` - The speed parameter for temperature update.
    /// # Returns
    /// A new `AdaptiveScheduler` configured with the provided parameters.
    pub fn new(
        initial_target_acc: f64,
        final_target_acc: f64,
        schedule_mode: TargetAccScheduleMode,
        gamma: f64,
    ) -> Self {
        Self {
            initial_target_acc,
            final_target_acc,
            schedule_mode,
            gamma,
        }
    }

    fn calculate_target_acc(&self, current_iter: usize, total_iter: usize) -> f64 {
        let initial_target_acc = self.initial_target_acc;
        let final_target_acc = self.final_target_acc;
        let schedule_mode = self.schedule_mode;
        let fraction = current_iter as f64 / total_iter as f64;
        match schedule_mode {
            TargetAccScheduleMode::Linear => {
                // linearly cool from initial_target_acc to final_target_acc
                initial_target_acc + fraction * (final_target_acc - initial_target_acc)
            }
            TargetAccScheduleMode::Exponential => {
                // cool from initial_target_acc to final_target_acc exponentially
                initial_target_acc * (final_target_acc / initial_target_acc).powf(fraction)
            }
            TargetAccScheduleMode::Cosine => {
                // cosine schedule
                final_target_acc
                    + 0.5 * (initial_target_acc - final_target_acc) * (1.0 + (PI * fraction).cos())
            }
        }
    }

    fn update_temperature<ST>(
        &self,
        current_temp: f64,
        current_iter: usize,
        total_iter: usize,
        step_result: &StepResult<ST, NotNan<f64>>,
    ) -> f64 {
        let n_accepted = step_result.accepted_transitions.len();
        let n_rejected = step_result.rejected_transitions.len();
        let n_total = n_accepted + n_rejected;
        let acc = n_accepted as f64 / n_total as f64;
        let target_acc = self.calculate_target_acc(current_iter, total_iter);
        current_temp * ((self.gamma * (target_acc - acc) / acc).exp())
    }
}

/// Optimizer that implements the adaptive annealing algorithm which tries to adapt temperature
/// to realize target acceptance rate scheduling
#[derive(Clone, Copy)]
pub struct AdaptiveAnnealingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    return_iter: usize,
    /// Frequency (in iterations) at which adaptive parameters are updated
    update_frequency: usize,
    /// Scheduler for target acceptance rate
    scheduler: AdaptiveScheduler,
}

impl AdaptiveAnnealingOptimizer {
    /// Creates a new `AdaptiveAnnealingOptimizer` instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `patience` - The number of iterations without improvement before terminating the optimization.
    /// * `n_trials` - The number of candidate solutions to evaluate per iteration.
    /// * `return_iter` - The number of iterations without improvement before reverting to the best solution.
    /// * `update_frequency` - The frequency (in iterations) at which adaptive parameters are updated.
    /// * `scheduler` - The adaptive scheduler for target acceptance rate.
    ///
    /// # Returns
    ///
    /// A new `AdaptiveAnnealingOptimizer` configured with the provided parameters.
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        update_frequency: usize,
        scheduler: AdaptiveScheduler,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            update_frequency,
            scheduler,
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for AdaptiveAnnealingOptimizer {
    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization.
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
        let mut current_temperature = tune_temperature(
            model,
            Some((initial_solution.clone(), initial_score)),
            self.update_frequency,
            self.scheduler.initial_target_acc,
        );
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut iter = 0;
        // Separate counters: one to trigger return-to-best, one for early stopping (patience)
        let mut return_stagnation_counter = 0;
        let mut patience_stagnation_counter = 0;
        let now = Instant::now();
        let mut remaining_time_limit = time_limit;
        let mut accepted_counter = 0;
        // let mut accepted_transitions = Vec::with_capacity(n_iter);
        // let mut rejected_transitions = Vec::with_capacity(n_iter);

        while iter < n_iter {
            let metropolis = MetropolisOptimizer::new(
                usize::MAX,
                self.n_trials,
                usize::MAX,
                current_temperature,
            );
            // make dummy callback
            let mut dummy_callback = |_: OptProgress<M::SolutionType, M::ScoreType>| {};
            let step_result = metropolis.step(
                model,
                current_solution.clone(),
                current_score,
                self.update_frequency,
                remaining_time_limit,
                &mut dummy_callback,
            );

            // 1. Update time and iteration counters
            let elapsed = now.elapsed();
            if elapsed >= time_limit {
                break;
            }
            remaining_time_limit = time_limit - elapsed;
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

            // 3. Update accepted counter and transitions
            let n_accepted = step_result.accepted_transitions.len();
            accepted_counter += n_accepted;
            // 4. Update current solution and score
            current_solution = step_result.last_solution.clone();
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

            // 7. Update algorithm-specific state
            current_temperature =
                self.scheduler
                    .update_temperature(current_temperature, iter, n_iter, &step_result);

            // 8. Invoke callback
            let progress = OptProgress {
                iter,
                accepted_count: accepted_counter,
                solution: best_solution.clone(),
                score: best_score,
            };
            callback(progress);
        }

        ((*best_solution.borrow()).clone(), best_score)
    }
}
