use crate::OptModel;

pub trait Optimizer<StateType, TransitionType, ModelType, AdditionalArgType, AdditionalRetType>
where
    ModelType: OptModel<StateType, TransitionType> + Sync + Send,
    StateType: Clone + Sync + Send,
{
    fn optimize(
        &self,
        model: &ModelType,
        initial_state: Option<&StateType>,
        n_iter: usize,
        arg: AdditionalArgType,
    ) -> (StateType, f64, AdditionalRetType);
}

mod hill_climbing;
pub use hill_climbing::HillClimbingOptimizer;

mod tabu_search;
pub use tabu_search::{TabuList, TabuSearchOptimizer};
