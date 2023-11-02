use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use auto_impl::auto_impl;
use rayon::prelude::*;

use crate::{
    callback::{OptCallbackFn, OptProgress},
    OptModel,
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
    for (state, transition, score) in samples.into_iter() {
        // Aspiration Criterion
        if score < best_score {
            return Some((state, transition, score));
        }

        // Not Tabu
        let item = (state, transition);
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
    /// - `n_trials` : number of trial states to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best state if there is no improvement after this number of iterations.
    pub fn new(patience: usize, n_trials: usize, return_iter: usize) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            phantom: PhantomData,
        }
    }
}

impl<M: OptModel, T: TabuList<Item = (M::StateType, M::TransitionType)>> LocalSearchOptimizer<M>
    for TabuSearchOptimizer<T>
{
    type ExtraIn = T;
    type ExtraOut = T;

    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_state` : the initial state to start optimization. If None, a random state will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `tabu_list` : initial tabu list
    fn optimize<F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        callback: Option<&F>,
        mut tabu_list: Self::ExtraIn,
    ) -> (M::StateType, M::ScoreType, Self::ExtraOut)
    where
        F: OptCallbackFn<M::StateType, M::ScoreType>,
    {
        let mut rng = rand::thread_rng();
        let mut current_state = if let Some(s) = initial_state {
            s
        } else {
            model.generate_random_state(&mut rng).unwrap()
        };
        let mut current_score = model.evaluate_state(&current_state);
        let best_state = Rc::new(RefCell::new(current_state.clone()));
        let mut best_score = current_score;
        let mut counter = 0;
        let mut accepted_counter = 0;

        for it in 0..n_iter {
            let mut samples = vec![];
            (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    let (state, transitions, score) =
                        model.generate_trial_state(&current_state, &mut rng, Some(current_score));
                    (state, transitions, score)
                })
                .collect_into_vec(&mut samples);

            samples.sort_unstable_by_key(|(_, _, score)| *score);

            let res = find_accepted_solution(samples, &tabu_list, best_score);

            if let Some((state, trans, score)) = res {
                if score < best_score {
                    best_score = score;
                    best_state.replace(state.clone());
                    counter = 0;
                }
                current_score = score;
                current_state = state.clone();
                tabu_list.append((state, trans));
                accepted_counter += 1;
            }

            counter += 1;

            if counter == self.return_iter {
                current_state = best_state.borrow().clone();
                current_score = best_score;
            }

            if counter == self.patience {
                break;
            }

            if let Some(f) = callback {
                let progress =
                    OptProgress::new(it, accepted_counter, best_state.clone(), best_score);
                f(progress);
            }
        }

        let best_state = (*best_state.borrow()).clone();

        (best_state, best_score, tabu_list)
    }
}
