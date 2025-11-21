use ordered_float::NotNan;

use crate::{Duration, OptModel, callback::OptCallbackFn};

use super::{GenericLocalSearchOptimizer, LocalSearchOptimizer, generic::StepResult};

pub fn metropolis_transition(beta: f64) -> impl Fn(NotNan<f64>, NotNan<f64>) -> f64 {
    move |current: NotNan<f64>, trial: NotNan<f64>| {
        let ds = trial - current;
        if ds <= NotNan::new(0.0).unwrap() {
            1.0
        } else {
            (-beta * ds.into_inner()).exp()
        }
    }
}

/// Optimizer that implements the Metropolis algorithm with constant beta
#[derive(Clone, Copy)]
pub struct MetropolisOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    return_iter: usize,
    /// Inverse temperature (beta)
    beta: f64,
}

impl MetropolisOptimizer {
    /// Constructor of MetropolisOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the best solution if there is no improvement after this number of iterations.
    /// - `beta` : inverse temperature
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, beta: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            beta,
        }
    }

    /// Perform one optimization step
    pub fn step<M: OptModel<ScoreType = NotNan<f64>>>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> StepResult<M::SolutionType, M::ScoreType> {
        let transition = |current: NotNan<f64>, trial: NotNan<f64>| {
            metropolis_transition(self.beta)(current, trial)
        };
        let generic_optimizer = GenericLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            transition,
        );
        generic_optimizer.step(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
        )
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for MetropolisOptimizer {
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
        initial_solution: <M as OptModel>::SolutionType,
        initial_score: <M as OptModel>::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<<M as OptModel>::SolutionType, <M as OptModel>::ScoreType>,
    ) -> (<M as OptModel>::SolutionType, <M as OptModel>::ScoreType) {
        let step_result = self.step(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
        );
        (step_result.best_solution, step_result.best_score)
    }
}
