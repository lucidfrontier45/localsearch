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

impl TransitionHandler<NotNan<f64>> for SimulatedAnnealing {
    fn update(&mut self, ctx: &UpdateCtx<'_, NotNan<f64>>) {
        if ctx.iter > 0 && ctx.iter.is_multiple_of(self.update_frequency.get()) {
            self.beta *= self.cooling_rate;
        }
    }

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        let ds = trial - current;
        if ds <= NotNan::new(0.0).unwrap() {
            1.0
        } else {
            (-self.beta * ds.into_inner()).exp()
        }
    }
}
