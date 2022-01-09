use ordered_float::NotNan;
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

impl TabuSearchOptimizer {
    pub fn new(patience: usize, n_trials: usize, return_iter: usize) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
        }
    }

    pub fn optimize<S, T, M, L, F>(
        &self,
        model: &M,
        initial_state: Option<&S>,
        n_iter: usize,
        arg: (L, Option<&F>),
    ) -> (S, f64, L)
    where
        S: Clone + Sync + Send,
        T: Clone + Sync + Send,
        M: OptModel<S, T> + Sync + Send,
        L: TabuList<Item = (S, T)>,
        F: Fn(usize, Rc<RefCell<S>>, f64),
    {
        let mut rng = rand::thread_rng();
        let mut current_state = if let Some(s) = initial_state {
            s.clone()
        } else {
            model.generate_random_state(&mut rng).unwrap()
        };
        let mut current_score = model.evaluate_state(&current_state);
        let best_state = Rc::new(RefCell::new(current_state.clone()));
        let mut best_score = current_score;
        let (mut tabu_list, callback) = arg;
        let mut counter = 0;

        for it in 0..n_iter {
            let mut res = vec![];
            (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    let (state, transitions, score) =
                        model.generate_trial_state(&current_state, &mut rng, Some(current_score));
                    (state, transitions, score)
                })
                .collect_into_vec(&mut res);

            res.sort_unstable_by_key(|(_, _, score)| NotNan::new(*score).unwrap());

            for (state, transition, score) in res {
                if score < best_score {
                    current_state = state.clone();
                    current_score = score;
                    best_state.replace(state.clone());
                    best_score = score;
                    counter = 0;
                    let item = (state, transition);
                    tabu_list.append(item);
                    break;
                } else {
                    let item = (state, transition);
                    if !tabu_list.contains(&item) {
                        current_state = item.0.clone();
                        current_score = score;
                        tabu_list.append(item);
                        break;
                    }
                }
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
