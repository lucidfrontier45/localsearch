use std::{cell::RefCell, f64::consts::PI, rc::Rc};

use ordered_float::NotNan;

use crate::{
    Duration, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{
    GenericLocalSearchOptimizer, LocalSearchOptimizer, metropolis::metropolis_transition,
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
    /// Constant target acceptance rate
    Constant,
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
            TargetAccScheduleMode::Constant => initial_target_acc,
        }
    }

    pub(crate) fn update_temperature(
        &self,
        current_beta: f64,
        current_iter: usize,
        total_iter: usize,
        acc: f64,
    ) -> f64 {
        // beta = beta * exp(-gamma * (target_acc - acc) / target_acc)
        let target_acc = self.calculate_target_acc(current_iter, total_iter);
        current_beta * ((-self.gamma * (target_acc - acc) / target_acc).exp())
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
    /// Initial inverse temperature
    initial_beta: f64,
    /// Scheduler for target acceptance rate
    scheduler: AdaptiveScheduler,
    /// Frequency (in iterations) at which adaptive parameters are updated
    update_frequency: usize,
}

impl AdaptiveAnnealingOptimizer {
    /// Creates a new `AdaptiveAnnealingOptimizer` instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `patience` - The number of iterations without improvement before terminating the optimization.
    /// * `n_trials` - The number of candidate solutions to evaluate per iteration.
    /// * `return_iter` - The number of iterations without improvement before reverting to the best solution.
    /// * `initial_beta` - The initial inverse temperature for the annealing process.
    /// * `scheduler` - The adaptive scheduler for target acceptance rate.
    /// * `update_frequency` - The frequency (in iterations) at which adaptive parameters are updated.
    ///
    /// # Returns
    ///
    /// A new `AdaptiveAnnealingOptimizer` configured with the provided parameters.
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_beta: f64,
        scheduler: AdaptiveScheduler,
        update_frequency: usize,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            initial_beta,
            scheduler,
            update_frequency,
        }
    }

    /// Tune inverse temperature parameter beta based on initial random trials
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `n_warmup` : number of warmup iterations to run
    /// - `target_initial_prob` : target acceptance probability for uphill moves at the beginning
    pub fn tune_initial_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
    ) -> Self {
        let tuned_beta = tune_temperature(
            model,
            initial_solution,
            n_warmup,
            self.scheduler.initial_target_acc,
        );

        Self {
            initial_beta: tuned_beta,
            ..self
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
        let current_beta = Rc::new(RefCell::new(self.initial_beta));
        let transition = {
            let current_beta = Rc::clone(&current_beta);
            move |current: NotNan<f64>, trial: NotNan<f64>| {
                let beta = *current_beta.borrow();
                metropolis_transition(beta)(current, trial)
            }
        };
        let mut callback_with_update = |progress: OptProgress<M::SolutionType, M::ScoreType>| {
            if progress.iter % self.update_frequency == 0 && progress.iter > 0 {
                let new_beta = self.scheduler.update_temperature(
                    *current_beta.borrow(),
                    progress.iter,
                    n_iter,
                    progress.acceptance_ratio,
                );
                current_beta.replace(new_beta);
            }
            callback(progress);
        };
        let generic_optimizer = GenericLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            transition,
        );
        generic_optimizer.optimize(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            &mut callback_with_update,
        )
    }
}
