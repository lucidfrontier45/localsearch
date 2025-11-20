use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;

use crate::{
    Duration, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{GenericLocalSearchOptimizer, LocalSearchOptimizer};

fn tsallis_transition_prob(current: f64, trial: f64, offset: f64, w: f64, q: f64, xi: f64) -> f64 {
    let delta_e = trial - current;
    let denominator = current * (1.0 + xi) - offset;
    let d = delta_e / denominator;
    if delta_e <= 0.0 {
        1.0
    } else {
        let arg = 1.0 - (1.0 - q) * w * d;
        arg.powf(1.0 / (1.0 - q)).max(0.01)
    }
}

fn tsallis_transition_prob_wrapper(
    current: NotNan<f64>,
    trial: NotNan<f64>,
    offset: Rc<RefCell<f64>>,
    w: f64,
    q: f64,
    xi: f64,
) -> f64 {
    tsallis_transition_prob(
        current.into_inner(),
        trial.into_inner(),
        *offset.borrow(),
        w,
        q,
        xi,
    )
}

/// Optimizer that implements Tsallis relative annealing algorithm
/// This is a generalization of relative annealing using Tsallis statistics.
/// The acceptance probability for worse solutions is [1 - (1-q) * w * ΔE / (E - E_best + ξ)]^{1/(1-q)},
/// where ΔE = trial - current, E = current, E_best = offset.
/// Assumes q > 1.0.
#[derive(Clone, Copy)]
pub struct TsallisRelativeAnnealingOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    w: f64,
    q: f64,
    xi: f64,
}

impl TsallisRelativeAnnealingOptimizer {
    /// Constructor of TsallisRelativeAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `w` : weight to be multiplied with the relative score difference.
    ///   Recommended value is reciprocal of expected relative score difference.
    /// - `q` : Tsallis parameter, assumed to be > 1.0. Recommended value is 2.5.
    /// - `xi` : parameter ξ in the acceptance probability formula.
    ///   Recommended value is 1.0 for integer objective and 0.1% of the objective value for continuous objective.
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        w: f64,
        q: f64,
        xi: f64,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            w,
            q,
            xi,
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M>
    for TsallisRelativeAnnealingOptimizer
{
    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
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
    ) -> (M::SolutionType, M::ScoreType) {
        // Rc<ReffCell> of the current best score
        let offset = Rc::new(RefCell::new(initial_score.into_inner()));

        // create transition probability function
        let transition_prob = |current: NotNan<f64>, trial: NotNan<f64>| {
            tsallis_transition_prob_wrapper(current, trial, offset.clone(), self.w, self.q, self.xi)
        };

        // wrap callback to update offset
        let mut callback_with_offset = |progress: OptProgress<M::SolutionType, M::ScoreType>| {
            // update offset
            offset.replace(progress.score.into_inner());
            // invoke original callback
            callback(progress);
        };

        // create generic optimizer and run optimization
        let optimizer = GenericLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            transition_prob,
        );

        optimizer.optimize(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            &mut callback_with_offset,
        )
    }
}

#[cfg(test)]
mod test {
    use super::tsallis_transition_prob;

    #[test]
    fn test_tsallis_transition_prob() {
        let w = 1e1;
        let q = 1.5;
        let offset = 0.0;

        // Improvement: should accept
        let p = tsallis_transition_prob(1.0, 0.9, offset, w, q, 1.0);
        assert!(p >= 1.0);

        // Worse: probability should decrease as d increases
        let p1 = tsallis_transition_prob(1.0, 1.1, offset, w, q, 1.0);
        let p2 = tsallis_transition_prob(1.0, 1.2, offset, w, q, 1.0);
        assert!(p1 > p2);
    }
}
