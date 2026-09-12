use crate::optim::transition::{TransitionHandler, UpdateCtx};

/// ε-greedy acceptance: improvements always accepted; otherwise accept
/// with fixed probability `epsilon`.
#[derive(Clone, Copy, Debug)]
pub struct EpsilonGreedy {
    /// Probability of accepting a worsening move. Must be in `[0, 1]`.
    pub epsilon: f64,
}

impl EpsilonGreedy {
    /// Constructor.
    pub const fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl<ST: Ord + Send + Sync + Copy> TransitionHandler<ST> for EpsilonGreedy {
    fn update(&mut self, _ctx: &UpdateCtx<'_, ST>) {}

    fn evaluate(&self, current: ST, trial: ST) -> f64 {
        if trial < current { 1.0 } else { self.epsilon }
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use super::EpsilonGreedy;
    use crate::optim::transition::TransitionHandler;

    #[test]
    fn improvement_always_accepted() {
        let h = EpsilonGreedy::new(0.1);
        let p = h.evaluate(NotNan::new(1.0).unwrap(), NotNan::new(0.5).unwrap());
        assert_eq!(p, 1.0);
    }

    #[test]
    fn worsening_uses_epsilon() {
        let h = EpsilonGreedy::new(0.25);
        let p = h.evaluate(NotNan::new(1.0).unwrap(), NotNan::new(1.5).unwrap());
        assert_eq!(p, 0.25);
    }
}
