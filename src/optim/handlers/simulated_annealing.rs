use super::metropolis::metropolis_probability;
use std::num::NonZero;

use ordered_float::NotNan;

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
