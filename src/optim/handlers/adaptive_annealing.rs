use std::{f64::consts::PI, num::NonZero};

use ordered_float::NotNan;

use super::metropolis::{metropolis_probability, tune_temperature};
use crate::{
    OptModel,
    optim::transition::{TransitionHandler, UpdateCtx},
};

/// Target acceptance-rate schedule used by [`AdaptiveScheduler`].
#[derive(Clone, Copy, Debug, Default)]
pub enum TargetAccScheduleMode {
    /// Linearly cool from `initial_target_acc` to `final_target_acc`.
    Linear,
    /// Exponentially cool from `initial_target_acc` to `final_target_acc`.
    Exponential,
    /// Cosine schedule from `initial_target_acc` to `final_target_acc`.
    #[default]
    Cosine,
    /// Constant target acceptance rate.
    Constant,
}

/// Scheduler that adapts the inverse temperature `beta` so that the
/// running acceptance rate tracks a target schedule.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveScheduler {
    /// Target acceptance rate at iter 0.
    pub initial_target_acc: f64,
    /// Target acceptance rate at the end of optimization.
    pub final_target_acc: f64,
    /// Schedule shape between `initial_target_acc` and `final_target_acc`.
    pub schedule_mode: TargetAccScheduleMode,
    /// Learning rate for the exponential `beta` update.
    pub gamma: f64,
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
    /// Constructor.
    pub const fn new(
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
        let fraction = current_iter as f64 / total_iter as f64;
        match self.schedule_mode {
            TargetAccScheduleMode::Linear => {
                self.initial_target_acc
                    + fraction * (self.final_target_acc - self.initial_target_acc)
            }
            TargetAccScheduleMode::Exponential => {
                self.initial_target_acc
                    * (self.final_target_acc / self.initial_target_acc).powf(fraction)
            }
            TargetAccScheduleMode::Cosine => {
                self.final_target_acc
                    + 0.5
                        * (self.initial_target_acc - self.final_target_acc)
                        * (1.0 + (PI * fraction).cos())
            }
            TargetAccScheduleMode::Constant => self.initial_target_acc,
        }
    }

    /// Update `beta` using the current acceptance ratio.
    pub fn update_temperature(
        &self,
        current_beta: f64,
        current_iter: usize,
        total_iter: usize,
        acc: f64,
    ) -> f64 {
        let target_acc = self.calculate_target_acc(current_iter, total_iter);
        current_beta * ((-self.gamma * (target_acc - acc) / target_acc).exp())
    }
}

/// Adaptive-annealing handler: Metropolis acceptance with a `beta` that
/// is adapted every `update_frequency` iterations by an [`AdaptiveScheduler`].
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveAnnealing {
    /// Current inverse temperature.
    pub beta: f64,
    /// Scheduler controlling how `beta` tracks the target acceptance rate.
    pub scheduler: AdaptiveScheduler,
    /// Non-zero number of iterations between updates.
    pub update_frequency: NonZero<usize>,
}

impl AdaptiveAnnealing {
    /// Constructor.
    pub const fn new(
        beta: f64,
        scheduler: AdaptiveScheduler,
        update_frequency: NonZero<usize>,
    ) -> Self {
        Self {
            beta,
            scheduler,
            update_frequency,
        }
    }

    /// Tune the initial inverse temperature `beta` based on random warmup trials.
    ///
    /// The target acceptance probability is taken from `scheduler.initial_target_acc`.
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution_and_score` : the initial solution to start warmup from. If `None`, a random solution will be generated.
    /// - `n_warmup` : number of warmup iterations to run
    /// - `returns` : handler with `beta` tuned for the scheduler's initial target acceptance rate
    pub fn tune_initial_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
    ) -> Self {
        Self {
            beta: tune_temperature(
                model,
                initial_solution_and_score,
                n_warmup,
                self.scheduler.initial_target_acc,
            ),
            ..self
        }
    }
}

impl TransitionHandler<NotNan<f64>> for AdaptiveAnnealing {
    fn update(&mut self, ctx: &UpdateCtx<'_, NotNan<f64>>) {
        if ctx.iter > 0 && ctx.iter.is_multiple_of(self.update_frequency.get()) {
            self.beta = self
                .scheduler
                .update_temperature(self.beta, ctx.iter, ctx.total, ctx.acc);
        }
    }

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        metropolis_probability(self.beta, current, trial)
    }
}
