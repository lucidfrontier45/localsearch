use auto_impl::auto_impl;
use trait_set::trait_set;

use crate::{callback::OptCallbackFn, Duration, OptModel};

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
        initial_solution: M::SolutionType,
        n_iter: usize,
        time_limit: Duration,
        callback: Option<&F>,
        extra_in: Self::ExtraIn,
    ) -> (M::SolutionType, M::ScoreType, Self::ExtraOut)
    where
        M: OptModel,
        F: OptCallbackFn<M::SolutionType, M::ScoreType>;

    /// Start optimization
    fn run<F>(
        &self,
        model: &M,
        initial_solution: Option<M::SolutionType>,
        n_iter: usize,
        time_limit: Duration,
        callback: Option<&F>,
        extra_in: Self::ExtraIn,
    ) -> (M::SolutionType, M::ScoreType, Self::ExtraOut)
    where
        M: OptModel,
        F: OptCallbackFn<M::SolutionType, M::ScoreType>,
    {
        let initial_solution = model.preprocess_solution(initial_solution.unwrap_or_else(|| {
            let mut rng = rand::thread_rng();
            model.generate_random_solution(&mut rng).unwrap()
        }));

        let (solution, score, extra) = self.optimize(
            model,
            initial_solution,
            n_iter,
            time_limit,
            callback,
            extra_in,
        );

        (model.postprocess_solution(solution), score, extra)
    }
}

trait_set! {
    /// Transition probability function
    pub trait TransitionProbabilityFn<ST: Ord + Sync + Send + Copy> = Fn(ST, ST) -> f64;
}
