use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use rand::{RngExt as _, SeedableRng as _};
use rayon::prelude::*;

use super::LocalSearchOptimizer;
use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
    counter::AcceptanceCounter,
};

/// Trait that a tabu list must satisfies
pub trait TabuList: Default {
    /// The type of item stored in the tabu list.
    type Item: Clone + Sync + Send;

    /// Set the length of the tabu list
    fn set_size(&mut self, n: usize);

    /// Check if the item is a Tabu
    fn contains(&self, transition: &Self::Item) -> bool;

    /// Append the item to the list
    fn append(&mut self, transition: Self::Item);
}

/// Optimizer that implements the tabu search algorithm
pub struct TabuSearchOptimizer<T: TabuList> {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    default_tabu_size: usize,
    /// RNG seed for bit-reproducible runs. `None` (default) preserves the
    /// entropy-driven behavior; set via [`Self::with_seed`].
    seed: Option<u64>,
    phantom: PhantomData<T>,
}

fn find_accepted_solution<M, L>(
    samples: Vec<(M::SolutionType, M::TransitionType, M::ScoreType)>,
    tabu_list: &L,
    best_score: M::ScoreType,
) -> Option<(M::SolutionType, M::TransitionType, M::ScoreType)>
where
    M: OptModel,
    L: TabuList<Item = M::TransitionType>,
{
    for (solution, transition, score) in samples.into_iter() {
        #[allow(unused_parens)]
        if (
            // Aspiration Criterion
            score < best_score ||
            // Not Tabu
            !tabu_list.contains(&transition)
        ) {
            return Some((solution, transition, score));
        }
    }

    None
}

impl<T: TabuList> TabuSearchOptimizer<T> {
    /// Constructor of TabuSearchOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `default_tabu_size` : default size of the tabu list
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        default_tabu_size: usize,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            default_tabu_size,
            seed: None,
            phantom: PhantomData,
        }
    }

    /// Pin the RNG seed so [`Self::optimize`] and [`Self::optimize_with_handler`]
    /// yield bit-identical `(solution, score)` across calls with the same
    /// inputs — independent of the rayon worker-thread count.
    ///
    /// `None` (the default) keeps the historical entropy-driven behavior.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Convenience: build a fresh `T::default()` tabu list with the optimizer's
    /// `default_tabu_size` and run optimization on it.
    pub fn optimize_default<M: OptModel<TransitionType = T::Item>>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType, T) {
        let mut tabu_list = T::default();
        tabu_list.set_size(self.default_tabu_size);
        self.optimize_with_handler(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            tabu_list,
        )
    }
}

impl<T: TabuList> TabuSearchOptimizer<T> {
    /// Start optimization with the supplied tabu list as the per-call handler.
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `handler` : the tabu list to use; returned alongside the best solution/score.
    #[allow(clippy::too_many_arguments)]
    pub fn optimize_with_handler<M, H>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        mut tabu_list: H,
    ) -> (M::SolutionType, M::ScoreType, H)
    where
        M: OptModel<TransitionType = H::Item>,
        H: TabuList,
    {
        let start_time = Instant::now();
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut return_stagnation_counter = 0;
        let mut patience_stagnation_counter = 0;
        let mut acceptance_counter = AcceptanceCounter::new(100);
        // Master RNG for sequential forks. Salted (3) so the trial stream
        // does not correlate with any other phase that hashes the same seed.
        let mut master = super::search_loop::make_master_rng(
            self.seed.map(|s| super::search_loop::derive_seed(s, 3)),
        );
        for it in 0..n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }
            let mut samples = vec![];
            let trial_seeds: Vec<u64> = (0..self.n_trials).map(|_| master.random()).collect();
            trial_seeds
                .into_par_iter()
                .map(|trial_seed| {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(trial_seed);
                    let (solution, transitions, score) = model.generate_trial_solution(
                        current_solution.clone(),
                        current_score,
                        &mut rng,
                    );
                    (solution, transitions, score)
                })
                .collect_into_vec(&mut samples);

            samples.sort_unstable_by_key(|(_, _, score)| *score);

            let res = find_accepted_solution::<M, H>(samples, &tabu_list, best_score);

            let accepted = res.is_some();
            acceptance_counter.enqueue(accepted);

            if accepted {
                let (solution, trans, score) = res.unwrap();
                // Accepted
                // 2. Update best solution and score
                if score < best_score {
                    best_score = score;
                    best_solution.replace(solution.clone());
                    return_stagnation_counter = 0;
                    patience_stagnation_counter = 0;
                } else {
                    return_stagnation_counter += 1;
                    patience_stagnation_counter += 1;
                }

                // 3. Update accepted counter and transitions (no transitions here)

                // 4. Update current solution and score
                current_score = score;
                current_solution = solution;

                // 7. Update algorithm-specific state
                tabu_list.append(trans);
            } else {
                // rejected
                // If no accepted, increment stagnation
                return_stagnation_counter += 1;
                patience_stagnation_counter += 1;
            }

            // 5. Check and handle return to best
            if return_stagnation_counter == self.return_iter {
                current_solution = best_solution.borrow().clone();
                current_score = best_score;
                return_stagnation_counter = 0;
            }

            // 6. Check patience
            if patience_stagnation_counter == self.patience {
                break;
            }

            // 8. Invoke callback
            let progress = OptProgress::new(
                it,
                acceptance_counter.acceptance_ratio(),
                best_solution.clone(),
                best_score,
            );
            callback(progress);
        }

        let best_solution = (*best_solution.borrow()).clone();

        (best_solution, best_score, tabu_list)
    }
}

impl<T: TabuList, M: OptModel<TransitionType = T::Item>> LocalSearchOptimizer<M>
    for TabuSearchOptimizer<T>
{
    fn rng_seed(&self) -> Option<u64> {
        self.seed
    }

    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType) {
        let (solution, score, _) = self.optimize_default(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
        );
        (solution, score)
    }
}
