use crate::{callback::OptCallbackFn, Duration, OptModel};

use super::{base::LocalSearchOptimizer, GenericLocalSearchOptimizer};

fn transition_prob<T: PartialOrd>(current: T, trial: T, epsilon: f64) -> f64 {
    if trial < current {
        return 1.0;
    }
    epsilon
}

/// Optimizer that implements epsilon-greedy algorithm.
/// Unlike a total greedy algorithm such as hill climbing,
/// it allows transitions that worsens the score with a fixed probability
#[derive(Clone, Copy)]
pub struct EpsilonGreedyOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    epsilon: f64,
}

impl EpsilonGreedyOptimizer {
    /// Constructor of EpsilonGreedyOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `epsilon` : probability to accept a transition that worsens the score. Must be in [0, 1].
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, epsilon: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            epsilon,
        }
    }
}

impl<M: OptModel> LocalSearchOptimizer<M> for EpsilonGreedyOptimizer {
    type ExtraIn = ();
    type ExtraOut = ();
    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `n_iter`: maximum iterations
    /// - `callback` : callback function that will be invoked at the end of each iteration
    /// - `_extra_in` : not used
    fn optimize<F>(
        &self,
        model: &M,
        initial_solution: Option<M::SolutionType>,
        n_iter: usize,
        time_limit: Duration,
        callback: Option<&F>,
        _extra_in: Self::ExtraIn,
    ) -> (M::SolutionType, M::ScoreType, Self::ExtraOut)
    where
        M: OptModel + Sync + Send,
        F: OptCallbackFn<M::SolutionType, M::ScoreType>,
    {
        let optimizer = GenericLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            |current, trial| transition_prob(current, trial, self.epsilon),
        );
        optimizer.optimize(
            model,
            initial_solution,
            n_iter,
            time_limit,
            callback,
            _extra_in,
        )
    }
}
