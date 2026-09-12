use ordered_float::NotNan;

use crate::optim::transition::{TransitionHandler, UpdateCtx};

/// Great-Deluge handler.
///
/// `update` linearly interpolates the water level from
/// `initial_level` (at iter 0) toward the best observed score (at iter `total`).
/// `evaluate` returns 1 for improving trials (`trial < current`)
/// and otherwise 1 iff `trial < level`.
#[derive(Clone, Copy, Debug)]
pub struct GreatDeluge {
    /// Water level at iter 0.
    pub level: f64,
    /// `initial_level` (frozen, used to recompute the schedule each iter).
    pub initial_level: f64,
}

impl GreatDeluge {
    /// Constructor.
    ///
    /// `initial_level` is the water level at iter 0; the schedule interpolates
    /// linearly toward `ctx.best` by `ctx.total`.
    pub const fn new(initial_level: f64) -> Self {
        Self {
            level: initial_level,
            initial_level,
        }
    }
}

impl TransitionHandler<NotNan<f64>> for GreatDeluge {
    fn update(&mut self, ctx: &UpdateCtx<'_, NotNan<f64>>) {
        let best = ctx.best.into_inner();
        let progress_ratio = (ctx.iter as f64) / (ctx.total as f64);
        self.level = self.initial_level - (self.initial_level - best) * progress_ratio;
    }

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        if trial < current {
            return 1.0;
        }
        if trial.into_inner() < self.level {
            1.0
        } else {
            0.0
        }
    }
}
