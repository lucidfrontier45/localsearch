use ordered_float::NotNan;

use crate::optim::transition::{TransitionHandler, UpdateCtx};

/// Logistic-annealing handler: `2 / (1 + exp(w * d))` with
/// `d = (trial - current) / |current|` (clamped to `f64::EPSILON`).
#[derive(Clone, Copy, Debug)]
pub struct LogisticAnnealing {
    /// Weight applied to the relative score difference.
    pub w: f64,
}

impl LogisticAnnealing {
    /// Constructor.
    pub const fn new(w: f64) -> Self {
        Self { w }
    }
}

impl TransitionHandler<NotNan<f64>> for LogisticAnnealing {
    fn update(&mut self, _ctx: &UpdateCtx<'_, NotNan<f64>>) {}

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        if trial < current {
            return 1.0;
        }
        let current = current.into_inner();
        let trial = trial.into_inner();
        let d = (trial - current) / current.abs().max(f64::EPSILON);
        2.0 / (1.0 + (self.w * d).exp())
    }
}
