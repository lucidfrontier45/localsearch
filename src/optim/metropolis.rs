use ordered_float::NotNan;
use rayon::prelude::*;

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

// Calculate target based on target_prob
// p = exp(-beta * ds ) => beta = -ln(p) / ds
// Average across all energy differences
pub(crate) fn calculate_temperature_from_acceptance_prob(
    energy_diffs: &[f64],
    target_acceptance_prob: f64,
) -> f64 {
    let average_energy_diff = energy_diffs.iter().sum::<f64>() / energy_diffs.len() as f64;
    let ln_prob = target_acceptance_prob.ln().clamp(-100.0, -0.01);
    -ln_prob / average_energy_diff.clamp(0.01, 100.0)
}

pub(crate) fn gather_energy_diffs<M: OptModel<ScoreType = NotNan<f64>>>(
    model: &M,
    initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
    n_warmup: usize,
) -> Vec<f64> {
    let mut rng = rand::rng();
    let (current_solution, current_score) =
        initial_solution_and_score.unwrap_or(model.generate_random_solution(&mut rng).unwrap());

    let energy_diffs: Vec<f64> = (0..n_warmup)
        .into_par_iter()
        .filter_map(|_| {
            let mut rng = rand::rng();
            let (_, _, trial_score) =
                model.generate_trial_solution(current_solution.clone(), current_score, &mut rng);
            let ds = trial_score - current_score;
            if ds > NotNan::new(0.0).unwrap() {
                Some(ds.into_inner())
            } else {
                None
            }
        })
        .collect();

    energy_diffs
}

/// Tune inverse temperature beta based on initial random trials
pub fn tune_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
    model: &M,
    initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
    n_warmup: usize,
    target_prob: f64,
) -> f64 {
    let energy_diffs = gather_energy_diffs(model, initial_solution_and_score, n_warmup);
    if energy_diffs.is_empty() {
        1.0
    } else {
        calculate_temperature_from_acceptance_prob(&energy_diffs, target_prob)
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
    /// - `initial_solution` : the initial solution to start optimization
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
