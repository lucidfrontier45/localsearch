use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rayon::prelude::*;

use crate::OptModel;

fn optimize<ModelType, StateType, TransitionType>(
    model: &ModelType,
    initial_state: StateType,
    initial_score: f64,
    n_iter: usize,
    patience: usize,
    counter: usize,
) -> (StateType, f64, usize)
where
    ModelType: OptModel<StateType, TransitionType>,
    StateType: Clone,
{
    if counter >= patience {
        return (initial_state, initial_score, counter);
    }

    let mut rng = rand::thread_rng();
    let mut current_state = initial_state;
    let mut current_score = initial_score;
    let mut counter = 0;
    for _ in 0..n_iter {
        let (trial_state, _) = model.generate_trial_state(&current_state, &mut rng);
        let trial_score = model.evaluate_state(&trial_state);
        if trial_score < current_score {
            current_state = trial_state;
            current_score = trial_score;
            counter = 0;
        } else {
            counter += 1;
            if counter >= patience {
                break;
            }
        }
    }
    (current_state, current_score, counter)
}

#[derive(Clone, Copy)]
pub struct HillClimbingOptimizer {
    patience: usize,
    n_restarts: usize,
}

impl HillClimbingOptimizer {
    pub fn new(patience: usize, n_restarts: usize) -> Self {
        Self {
            patience,
            n_restarts,
        }
    }

    pub fn optimize<S, T, M, F>(
        &self,
        model: &M,
        initial_state: Option<&S>,
        n_iter: usize,
        batch_size: usize,
        callback: Option<&F>,
    ) -> (S, f64)
    where
        M: OptModel<S, T> + Sync + Send,
        S: Clone + Sync + Send,
        F: Fn(usize, Rc<RefCell<S>>, f64),
    {
        let mut rng = rand::thread_rng();
        let n = (n_iter as f32 / batch_size as f32).ceil() as usize;
        let final_state = Rc::new(RefCell::new(model.generate_random_state(&mut rng).unwrap()));
        let mut final_score = 0.0;

        let mut states = if let Some(s) = initial_state {
            (0..self.n_restarts)
                .into_iter()
                .map(|_| s.clone())
                .collect::<Vec<_>>()
        } else {
            (0..self.n_restarts)
                .into_iter()
                .map(|_| model.generate_random_state(&mut rng).unwrap())
                .collect::<Vec<_>>()
        };

        let mut scores = states
            .iter()
            .map(|s| model.evaluate_state(s))
            .collect::<Vec<_>>();

        let mut counters = vec![0; self.n_restarts];

        for it in 0..n {
            let remained_iter = n_iter - it * batch_size;
            let target_iter = std::cmp::min(batch_size, remained_iter);
            let mut res = vec![];
            (0..self.n_restarts)
                .into_par_iter()
                .map(|i| {
                    optimize(
                        model,
                        states[i].clone(),
                        scores[i],
                        target_iter,
                        self.patience,
                        counters[i],
                    )
                })
                .collect_into_vec(&mut res);
            for (i, (state, score, counter)) in res.into_iter().enumerate() {
                states[i] = state;
                scores[i] = score;
                counters[i] = counter;
            }

            let (best_score, best_state) = scores
                .iter()
                .zip(states.iter())
                .min_by_key(|&(score, _)| NotNan::new(*score).unwrap())
                .unwrap();

            final_score = *best_score;
            final_state.replace(best_state.clone());

            if let Some(f) = callback {
                f(n_iter - remained_iter, final_state.clone(), final_score);
            }
        }

        let final_state = final_state.borrow().clone();

        (final_state, final_score)
    }
}
