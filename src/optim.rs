mod hill_climbing;

use crate::OptModel;

pub trait Optimizer {
    type AdditionalArgType;
    type AdditionalRetType;

    fn optimize<ModelType, StateType>(
        &self,
        model: &ModelType,
        initial_state: Option<&StateType>,
        n_iter: usize,
        arg: &Self::AdditionalArgType,
    ) -> (StateType, f64, Self::AdditionalRetType)
    where
        ModelType: OptModel<StateType = StateType> + Sync + Send,
        StateType: Clone + Sync + Send;
}

pub use hill_climbing::HillClimbingOptimizer;
