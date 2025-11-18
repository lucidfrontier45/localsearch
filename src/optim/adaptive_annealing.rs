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

fn update_target_acc(
    initial_target_acc: f64,
    final_target_acc: f64,
    current_iter: usize,
    total_iter: usize,
    schedule_mode: TargetAccScheduleMode,
) -> f64 {
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
    current_temp: f64,
    target_acc: f64,
    step_result: &StepResult<ST, NotNan<f64>>,
) -> f64 {
    let n_accepted = step_result.accepted_transitions.len();
    let n_rejected = step_result.rejected_transitions.len();
    let n_total = n_accepted + n_rejected;
    let acc = n_accepted as f64 / n_total as f64;
    current_temp * (target_acc / acc)
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
    /// Initial target acceptance rate
    initial_target_acc: f64,
    /// Final target acceptance rate
    final_target_acc: f64,
    /// Number of steps after which temperature is updated
    update_frequency: usize,
    /// Target acceptance rate scheduling mode
    target_acc_schedule_mode: TargetAccScheduleMode,
}

impl AdaptiveAnnealingOptimizer {
    /// Creates a new `AdaptiveAnnealingOptimizer` instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `patience` - The number of iterations without improvement before terminating the optimization.
    /// * `n_trials` - The number of candidate solutions to evaluate per iteration.
    /// * `return_iter` - The iteration interval at which to potentially revert to a previous state.
    /// * `initial_target_acc` - The initial target acceptance rate for the annealing process.
    /// * `final_target_acc` - The final target acceptance rate for the annealing process.
    /// * `update_frequency` - The frequency (in iterations) at which adaptive parameters are updated.
    /// * `target_acc_schedule_mode` - The scheduling mode for adjusting the target acceptance rate over time.
    ///
    /// # Returns
    ///
    /// A new `AdaptiveAnnealingOptimizer` configured with the provided parameters.
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_target_acc: f64,
        final_target_acc: f64,
        update_frequency: usize,
        target_acc_schedule_mode: TargetAccScheduleMode,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            initial_target_acc,
            final_target_acc,
            update_frequency,
            target_acc_schedule_mode,
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for AdaptiveAnnealingOptimizer {
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
        let mut current_temperature = tune_temperature(
            model,
            Some((initial_solution.clone(), initial_score)),
            self.update_frequency,
            self.initial_target_acc,
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
            let current_target_acc = update_target_acc(
                self.initial_target_acc,
                self.final_target_acc,
                iter,
                n_iter,
                self.target_acc_schedule_mode,
            );
            current_temperature =
                update_temperature(current_temperature, current_target_acc, &step_result);

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
