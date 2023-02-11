use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::Rng;
use rayon::prelude::*;

use super::callback::{OptCallbackFn, OptProgress};
use crate::OptModel;

fn logistic_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn calc_transition_score(trial_score: f64, current_score: f64, w: f64) -> f64 {
    let ds = (trial_score - current_score) / current_score;

    if ds < 0.0 {
        1.0
    } else {
        logistic_sigmoid(-ds * w) + 0.5
    }
}

#[derive(Clone, Copy)]
pub struct LogisticAnnealingOptimizer {
    patience: usize,
    n_trials: usize,
    w: f64,
}

impl LogisticAnnealingOptimizer {
    pub fn new(patience: usize, n_trials: usize, w: f64) -> Self {
        Self {
            patience,
            n_trials,
            w,
        }
    }

    pub fn optimize<M, F>(
        &self,
        model: &M,
        initial_state: Option<M::StateType>,
        n_iter: usize,
        callback: Option<&F>,
    ) -> (M::StateType, M::ScoreType)
    where
        M: OptModel<ScoreType = NotNan<f64>> + Sync + Send,
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
        let mut accepted_counter = 0;
        let mut counter = 0;

        for it in 0..n_iter {
            let (trial_state, trial_score) = (0..self.n_trials)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    let (state, _, score) =
                        model.generate_trial_state(&current_state, &mut rng, Some(current_score));
                    (state, score)
                })
                .min_by_key(|(_, score)| *score)
                .unwrap();

            let p =
                calc_transition_score(trial_score.into_inner(), current_score.into_inner(), self.w);
            let r: f64 = rng.gen();

            if p > r {
                current_state = trial_state;
                current_score = trial_score;
                accepted_counter += 1;
            }

            if current_score < best_score {
                best_state.replace(current_state.clone());
                best_score = current_score;
                counter = 0;
            }

            counter += 1;
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
        (best_state, best_score)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;

    use super::calc_transition_score;

    #[test]
    fn test_calc_transition_score() {
        let w = 1000.0;
        assert_abs_diff_eq!(calc_transition_score(0.9, 1.0, w), 1.0, epsilon = 0.01);

        let p1 = calc_transition_score(1.1, 1.0, w);
        let p2 = calc_transition_score(1.2, 1.0, w);
        assert!(p1 > p2);
    }
}
