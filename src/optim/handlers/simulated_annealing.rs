use super::metropolis::{metropolis_probability, tune_temperature};
use std::num::NonZero;

use ordered_float::NotNan;

use crate::OptModel;
use crate::optim::transition::{TransitionHandler, UpdateCtx};

/// Simulated-annealing handler with geometric cooling of `beta`.
///
/// `update` multiplies `beta` by `cooling_rate` whenever `iter` is a
/// non-zero multiple of `update_frequency`.
#[derive(Clone, Copy, Debug)]
pub struct SimulatedAnnealing {
    /// Current inverse temperature.
    pub beta: f64,
    /// Geometric cooling factor applied to `beta` on each update.
    pub cooling_rate: f64,
    /// Non-zero number of iterations between updates.
    pub update_frequency: NonZero<usize>,
}

impl SimulatedAnnealing {
    /// Constructor.
    pub const fn new(beta: f64, cooling_rate: f64, update_frequency: NonZero<usize>) -> Self {
        Self {
            beta,
            cooling_rate,
            update_frequency,
        }
    }

    /// Tune the initial inverse temperature `beta` based on random warmup trials.
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution_and_score` : the initial solution to start warmup from. If `None`, a random solution will be generated.
    /// - `n_warmup` : number of warmup iterations to run
    /// - `target_initial_prob` : target acceptance probability for uphill moves at the beginning
    /// - `returns` : handler with `beta` tuned for the target initial acceptance probability
    pub fn tune_initial_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
        target_initial_prob: f64,
    ) -> Self {
        Self {
            beta: tune_temperature(
                model,
                initial_solution_and_score,
                n_warmup,
                target_initial_prob,
            ),
            ..self
        }
    }

    /// Tune the cooling rate so that `beta` is cooled to `1e2` after `n_iter` iterations.
    ///
    /// - `n_iter` : total number of iterations planned for the optimization
    /// - `returns` : handler with `cooling_rate` computed from the current `beta`
    pub fn tune_cooling_rate(self, n_iter: usize) -> Self {
        Self {
            cooling_rate: tune_cooling_rate(
                self.beta,
                1e2,
                n_iter / self.update_frequency.get(),
            ),
            ..self
        }
    }
}

/// Tune cooling rate based on initial and final inverse temperatures.
///
/// Initial beta will be cooled to final beta after `n_iter` iterations.
/// - `initial_beta` : initial inverse temperature
/// - `final_beta` : final inverse temperature
/// - `n_iter` : number of iterations
/// - `returns` : cooling rate
pub fn tune_cooling_rate(initial_beta: f64, final_beta: f64, n_iter: usize) -> f64 {
    (final_beta / initial_beta).powf(1.0 / n_iter as f64)
}

impl TransitionHandler<NotNan<f64>> for SimulatedAnnealing {
    fn update(&mut self, ctx: &UpdateCtx<'_, NotNan<f64>>) {
        if ctx.iter > 0 && ctx.iter.is_multiple_of(self.update_frequency.get()) {
            self.beta *= self.cooling_rate;
        }
    }

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        metropolis_probability(self.beta, current, trial)
    }
}
