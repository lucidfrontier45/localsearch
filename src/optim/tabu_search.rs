use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use auto_impl::auto_impl;
use rayon::prelude::*;

use crate::{
    callback::{OptCallbackFn, OptProgress},
    Duration, Instant, OptModel,
};

use super::LocalSearchOptimizer;

/// Trait that a tabu list must satisfies
#[auto_impl(&mut, Box)]
pub trait TabuList {
    /// Item type of the likst
    type Item;

    /// Check if the item is a Tabu
    fn contains(&self, item: &Self::Item) -> bool;

    /// Append the item to the list
    fn append(&mut self, item: Self::Item);
}

/// Optimizer that implements the tabu search algorithm
pub struct TabuSearchOptimizer<T: TabuList> {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    phantom: PhantomData<T>,
}

fn find_accepted_solution<S, T, L, O>(
    samples: Vec<(S, T, O)>,
    tabu_list: &L,
    best_score: O,
) -> Option<(S, T, O)>
where
    L: TabuList<Item = (S, T)>,
    O: Ord,
{
    for (solution, transition, score) in samples.into_iter() {
        // Aspiration Criterion
        if score < best_score {
            return Some((solution, transition, score));
        }

        // Not Tabu
        let item = (solution, transition);
        if !tabu_list.contains(&item) {
            return Some((item.0, item.1, score));
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
    pub fn new(patience: usize, n_trials: usize, return_iter: usize) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            phantom: PhantomData,
        }
    }
}

impl<M: OptModel, T: TabuList<Item = (M::SolutionType, M::TransitionType)>> LocalSearchOptimizer<M>
    for TabuSearchOptimizer<T>
{
    type ExtraIn = T;
    type ExtraOut = T;

    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `tabu_list` : initial tabu list
    fn optimize<F>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        n_iter: usize,
        time_limit: Duration,
        callback: Option<&F>,
        mut tabu_list: Self::ExtraIn,
    ) -> (M::SolutionType, M::ScoreType, Self::ExtraOut)
    where
        F: OptCallbackFn<M::SolutionType, M::ScoreType>,
    {
        let start_time = Instant::now();
        let mut current_solution = initial_solution;
        let mut current_score = model.evaluate_solution(&current_solution);
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut counter = 0;
        let mut accepted_counter = 0;

        for it in 0..n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }
            let mut samples = vec![];
            (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    let (solution, transitions, score) = model.generate_trial_solution(
                        &current_solution,
                        &mut rng,
                        Some(current_score),
                    );
                    (solution, transitions, score)
                })
                .collect_into_vec(&mut samples);

            samples.sort_unstable_by_key(|(_, _, score)| *score);

            let res = find_accepted_solution(samples, &tabu_list, best_score);

            if let Some((solution, trans, score)) = res {
                if score < best_score {
                    best_score = score;
                    best_solution.replace(solution.clone());
                    counter = 0;
                }
                current_score = score;
                current_solution = solution.clone();
                tabu_list.append((solution, trans));
                accepted_counter += 1;
            }

            counter += 1;

            if counter == self.return_iter {
                current_solution = best_solution.borrow().clone();
                current_score = best_score;
            }

            if counter == self.patience {
                break;
            }

            if let Some(f) = callback {
                let progress =
                    OptProgress::new(it, accepted_counter, best_solution.clone(), best_score);
                f(progress);
            }
        }

        let best_solution = (*best_solution.borrow()).clone();

        (best_solution, best_score, tabu_list)
    }
}
