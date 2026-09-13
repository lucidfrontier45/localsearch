use auto_impl::auto_impl;
use rand::SeedableRng as _;

use crate::{
    Duration, LocalsearchError, OptModel,
    callback::OptCallbackFn,
    optim::search_loop::{derive_seed, make_master_rng},
};

/// Optimizer that implements local search algorithm.
#[auto_impl(&, Box, Rc, Arc)]
pub trait LocalSearchOptimizer<M: OptModel> {
    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType);

    /// Seed for reproducible, bit-identical runs.
    ///
    /// Returning `Some(seed)` makes the optimizer deterministic across calls:
    /// the same seed with the same inputs yields the same `(solution, score)`.
    /// Returning `None` (the default) preserves the previous entropy-driven
    /// behavior. Implementations that split work across phases (initial
    /// solution, loop, warmup, replica swap, etc.) must feed each phase a
    /// distinct seeded stream so the same user `seed` never flows through two
    /// phases via an identical `seed_from_u64` chain.
    fn rng_seed(&self) -> Option<u64> {
        None
    }

    /// generate initial solution if not given and run optimization with callback
    fn run_with_callback(
        &self,
        model: &M,
        initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> Result<(M::SolutionType, M::ScoreType), LocalsearchError> {
        let (initial_solution, initial_score) = match initial_solution_and_score {
            Some((solution, score)) => (solution, score),
            None => {
                // salt = 1: independent from the loop's master RNG stream (salt 2)
                // so the two phases do not share an identical `seed_from_u64`
                // stream even when both derive from the same user seed.
                let mut rng = match self.rng_seed() {
                    Some(seed) => rand::rngs::StdRng::seed_from_u64(derive_seed(seed, 1)),
                    None => make_master_rng(None),
                };
                model.generate_random_solution(&mut rng)?
            }
        };

        let (initial_solution, initial_score) =
            model.preprocess_solution(initial_solution, initial_score)?;

        let (solution, score) = self.optimize(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
        );

        let (solution, score) = model.postprocess_solution(solution, score);
        Ok((solution, score))
    }

    /// generate initial solution if not given and run optimization
    fn run(
        &self,
        model: &M,
        initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
        n_iter: usize,
        time_limit: Duration,
    ) -> Result<(M::SolutionType, M::ScoreType), LocalsearchError> {
        self.run_with_callback(
            model,
            initial_solution_and_score,
            n_iter,
            time_limit,
            &mut |_| {},
        )
    }
}
