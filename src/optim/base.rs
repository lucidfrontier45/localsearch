use auto_impl::auto_impl;
use trait_set::trait_set;

use crate::{callback::OptCallbackFn, OptModel};

/// Optimizer that implements local search algorithm
#[auto_impl(&, Box, Rc, Arc)]
pub trait LocalSearchOptimizer<M: OptModel> {
    /// Extra input type
    type ExtraIn;
    /// Extra output type
    type ExtraOut;

    /// Start optimization
    fn optimize<F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        callback: Option<&F>,
        ext_in: Self::ExtraIn,
    ) -> (M::StateType, M::ScoreType, Self::ExtraOut)
    where
        M: OptModel,
        F: OptCallbackFn<M::StateType, M::ScoreType>;
}

trait_set! {
    /// Transition probability function
    pub trait TransitionProbabilityFn<ST: Ord + Sync + Send + Copy> = Fn(ST, ST) -> f64;
}
