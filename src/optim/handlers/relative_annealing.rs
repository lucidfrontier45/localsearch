use ordered_float::NotNan;

use crate::optim::transition::{TransitionHandler, UpdateCtx};

/// Relative-annealing handler: `exp(-beta * (trial - current) / |current|)`.
///
/// `current_score == 0` is clamped to `f64::EPSILON` to keep the result finite.
#[derive(Clone, Copy, Debug)]
pub struct RelativeAnnealing {
    /// Weight applied to the relative score difference.
    pub beta: f64,
}

impl RelativeAnnealing {
    /// Constructor.
    pub const fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl TransitionHandler<NotNan<f64>> for RelativeAnnealing {
    fn update(&mut self, _ctx: &UpdateCtx<'_, NotNan<f64>>) {}

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        if trial < current {
            return 1.0;
        }
        let current = current.into_inner();
        let trial = trial.into_inner();
        let d = (trial - current) / current.abs().max(f64::EPSILON);
        (-self.beta * d).exp()
    }
}
