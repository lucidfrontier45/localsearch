use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};

use crate::OptModel;

pub trait TabuList {
    type Item;

    fn contains(&self, item: &Self::Item) -> bool;
    fn append(&mut self, item: Self::Item);
}

pub struct TabuSearchOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
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

impl TabuSearchOptimizer {
    pub fn new(patience: usize, n_trials: usize, return_iter: usize) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
        }
    }

    pub fn optimize<M, L, F, S, T>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        mut tabu_list: L,
        callback: Option<&F>,
    ) -> (M::StateType, M::ScoreType, L)
    where
        M: OptModel<StateType = S, TransitionType = T> + Sync + Send,
        L: TabuList<Item = (M::StateType, M::TransitionType)>,
        F: Fn(usize, Rc<RefCell<M::StateType>>, M::ScoreType),
        S: Clone + Sync + Send,
        T: Clone + Sync + Send,
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
                f(it, best_state.clone(), best_score);
            }
        }

        let best_state = (*best_state.borrow()).clone();

        (best_state, best_score, tabu_list)
    }
}
