use auto_impl::auto_impl;

use crate::{Duration, LocalsearchError, OptModel, callback::OptCallbackFn};

/// Optimizer that implements local search algorithm.
///
/// `optimize`/`run_with_callback`/`run` take a per-call `handler: H` parameter
/// that owns algorithm-specific state (transition handler, tabu list, …).
/// The handler is mutated during the run and returned back to the caller so
/// any per-iteration state survives the call.
///
/// `H` is a type parameter of the trait (not a method generic) so that each
/// impl can declare its own bound on `H` via the impl block's `where` clause
/// without triggering "impl has stricter requirements than trait".
#[auto_impl(&, Box, Rc, Arc)]
pub trait LocalSearchOptimizer<M: OptModel, H> {
    /// Start optimization with the supplied handler.
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `handler` : per-call algorithm-specific state; returned to the caller after the run.
    #[allow(clippy::too_many_arguments)]
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        handler: H,
    ) -> (M::SolutionType, M::ScoreType, H);

    /// generate initial solution if not given and run optimization with callback and handler.
    #[allow(clippy::too_many_arguments)]
    fn run_with_callback(
        &self,
        model: &M,
        initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        handler: H,
    ) -> Result<(M::SolutionType, M::ScoreType), LocalsearchError> {
        let (initial_solution, initial_score) = match initial_solution_and_score {
            Some((solution, score)) => (solution, score),
            None => {
                let mut rng = rand::rng();
                model.generate_random_solution(&mut rng)?
            }
        };

        let (initial_solution, initial_score) =
            model.preprocess_solution(initial_solution, initial_score)?;

        let (solution, score, _handler) = self.optimize(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            handler,
        );

        let (solution, score) = model.postprocess_solution(solution, score);
        Ok((solution, score))
    }

    /// generate initial solution if not given and run optimization with handler.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        model: &M,
        initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
        n_iter: usize,
        time_limit: Duration,
        handler: H,
    ) -> Result<(M::SolutionType, M::ScoreType), LocalsearchError> {
        let mut noop =
            |_progress: crate::callback::OptProgress<M::SolutionType, M::ScoreType>| {};
        self.run_with_callback(
            model,
            initial_solution_and_score,
            n_iter,
            time_limit,
            &mut noop,
            handler,
        )
    }
}
