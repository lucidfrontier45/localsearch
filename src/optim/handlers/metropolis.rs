use ordered_float::NotNan;

use crate::optim::transition::{TransitionHandler, UpdateCtx};

/// Classic Metropolis acceptance with a constant inverse temperature `beta`.
#[derive(Clone, Copy, Debug)]
pub struct Metropolis {
    /// Inverse temperature.
    pub beta: f64,
}

impl Metropolis {
    /// Constructor.
    pub const fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl TransitionHandler<NotNan<f64>> for Metropolis {
    fn update(&mut self, _ctx: &UpdateCtx<'_, NotNan<f64>>) {}

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        let ds = trial - current;
        if ds <= NotNan::new(0.0).unwrap() {
            1.0
        } else {
            (-self.beta * ds.into_inner()).exp()
        }
    }
}
