use std::num::NonZero;

use ordered_float::NotNan;

use crate::optim::transition::{TransitionHandler, UpdateCtx};

use super::adaptive_annealing::AdaptiveScheduler;

/// Tsallis relative-annealing handler.
///
/// Acceptance probability for a worsening trial is
/// `max([1 - (1-q) * beta * d]^{1/(1-q)}, 0.01)`
/// where `d = (trial - current) / (current - offset + xi)`.
#[derive(Clone, Copy, Debug)]
pub struct TsallisAnnealing {
    /// `offset` (best-so-far) tracked across iterations.
    pub offset: f64,
    /// Current inverse temperature.
    pub beta: f64,
    /// Tsallis `q` parameter (must be `> 1`).
    pub q: f64,
    /// Regularization `xi` added to the denominator.
    pub xi: f64,
    /// Scheduler controlling `beta` updates.
    pub scheduler: AdaptiveScheduler,
    /// Non-zero number of iterations between `beta` updates.
    pub update_frequency: NonZero<usize>,
}

impl TsallisAnnealing {
    /// Constructor.
    ///
    /// `initial_offset` is the starting value of the offset (typically the
    /// initial solution's score).
    pub const fn new(
        initial_offset: f64,
        beta: f64,
        q: f64,
        xi: f64,
        scheduler: AdaptiveScheduler,
        update_frequency: NonZero<usize>,
    ) -> Self {
        Self {
            offset: initial_offset,
            beta,
            q,
            xi,
            scheduler,
            update_frequency,
        }
    }
}

impl TransitionHandler<NotNan<f64>> for TsallisAnnealing {
    fn update(&mut self, ctx: &UpdateCtx<'_, NotNan<f64>>) {
        // offset tracks the best score as observed at the start of this iter.
        self.offset = ctx.best.into_inner();
        if ctx.iter > 0 && ctx.iter.is_multiple_of(self.update_frequency.get()) {
            self.beta = self
                .scheduler
                .update_temperature(self.beta, ctx.iter, ctx.total, ctx.acc);
        }
    }

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        let current = current.into_inner();
        let trial = trial.into_inner();
        let delta_e = trial - current;
        let denominator = current - self.offset + self.xi;
        let d = delta_e / denominator;
        if delta_e <= 0.0 {
            1.0
        } else {
            let arg = 1.0 - (1.0 - self.q) * self.beta * d;
            arg.powf(1.0 / (1.0 - self.q)).max(0.01)
        }
    }
}
