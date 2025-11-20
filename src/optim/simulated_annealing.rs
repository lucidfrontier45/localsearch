use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rayon::prelude::*;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
    counter::AcceptanceCounter,
};

use super::{LocalSearchOptimizer, MetropolisOptimizer, generic::StepResult};

/// Tune cooling rate based on initial and final inverse temperatures and number of iterations
/// initial beta will be cooled to final beta after n_iter iterations
/// - `initial_beta` : initial inverse temperature
/// - `final_beta` : final inverse temperature
/// - `n_iter` : number of iterations
/// - returns : cooling rate
pub fn tune_cooling_rate(initial_beta: f64, final_beta: f64, n_iter: usize) -> f64 {
    (final_beta / initial_beta).powf(1.0 / n_iter as f64)
}

// Calculate target based on target_prob
// p = exp(-beta * ds ) => beta = -ln(p) / ds
// Average across all energy differences
fn calculate_temperature_from_acceptance_prob(
    energy_diffs: &[f64],
    target_acceptance_prob: f64,
) -> f64 {
    let average_energy_diff = energy_diffs.iter().sum::<f64>() / energy_diffs.len() as f64;
    let ln_prob = target_acceptance_prob.ln().clamp(-100.0, -0.01);
    -ln_prob / average_energy_diff
}

fn gather_energy_diffs<M: OptModel<ScoreType = NotNan<f64>>>(
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

/// Optimizer that implements the simulated annealing algorithm
#[derive(Clone, Copy)]
pub struct SimulatedAnnealingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    return_iter: usize,
    /// Initial inverse temperature
    initial_beta: f64,
    /// Cooling rate
    cooling_rate: f64,
    /// Number of steps after which temperature is updated
    update_frequency: usize,
}

impl SimulatedAnnealingOptimizer {
    /// Constructor of SimulatedAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the best solution if there is no improvement after this number of iterations.
    /// - `initial_beta` : initial inverse temperature
    /// - `cooling_rate` : cooling rate
    /// - `update_frequency` : number of steps after which temperature is updated
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_beta: f64,
        cooling_rate: f64,
        update_frequency: usize,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            initial_beta,
            cooling_rate,
            update_frequency,
        }
    }
}

impl SimulatedAnnealingOptimizer {
    /// Tune inverse temperature parameter beta based on initial random trials
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `n_warmup` : number of warmup iterations to run
    /// - `target_initial_prob` : target acceptance probability for uphill moves at the beginning
    pub fn tune_initial_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
        target_initial_prob: f64,
    ) -> Self {
        let tuned_beta = tune_temperature(model, initial_solution, n_warmup, target_initial_prob);

        Self {
            initial_beta: tuned_beta,
            ..self
        }
    }

    /// Tune cooling rate based on self.initial_beta, final beta of 1e2
    pub fn tune_cooling_rate(self, n_iter: usize) -> Self {
        let cooling_rate =
            tune_cooling_rate(self.initial_beta, 1e2, n_iter / self.update_frequency);

        Self {
            cooling_rate,
            ..self
        }
    }

    /// Start optimization
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization. If None, a random solution will be generated.
    /// - `initial_score` : the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback` : callback function that will be invoked at the end of each iteration
    pub fn step<M: OptModel<ScoreType = NotNan<f64>>>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> StepResult<M::SolutionType, M::ScoreType> {
        let mut current_beta = self.initial_beta;
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut iter = 0;
        // Separate counters: one to trigger return-to-best, one for early stopping (patience)
        let mut return_stagnation_counter = 0;
        let mut patience_stagnation_counter = 0;
        let now = Instant::now();
        let mut remaining_time_limit = time_limit;
        let mut acceptance_counter = AcceptanceCounter::default();

        while iter < n_iter {
            let metropolis = MetropolisOptimizer::new(
                self.patience,
                self.n_trials,
                self.return_iter,
                current_beta,
            );
            // make dummy callback
            let mut dummy_callback = |_: OptProgress<M::SolutionType, M::ScoreType>| {};
            let step_result = metropolis.step(
                model,
                current_solution.clone(),
                current_score,
                self.update_frequency,
                remaining_time_limit,
                &mut dummy_callback,
            );

            // 1. Update time and iteration counters
            let elapsed = now.elapsed();
            if elapsed >= time_limit {
                break;
            }
            remaining_time_limit = time_limit - elapsed;
            iter += self.update_frequency;

            // 2. Update best solution and score
            if step_result.best_score < best_score {
                best_solution.replace(step_result.best_solution);
                best_score = step_result.best_score;
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            } else {
                return_stagnation_counter += self.update_frequency;
                patience_stagnation_counter += self.update_frequency;
            }

            // 3. Update accepted counter and transitions
            acceptance_counter = step_result.acceptance_counter;

            // 4. Update current solution and score
            current_solution = step_result.last_solution;
            current_score = step_result.last_score;

            // 5. Check and handle return to best
            if return_stagnation_counter >= self.return_iter {
                current_solution = (*best_solution.borrow()).clone();
                current_score = best_score;
                return_stagnation_counter = 0;
            }

            // 6. Check patience
            if patience_stagnation_counter >= self.patience {
                break;
            }

            // 7. Update algorithm-specific state
            current_beta *= self.cooling_rate;

            // 8. Invoke callback
            let progress = OptProgress {
                iter,
                acceptance_ratio: acceptance_counter.acceptance_ratio(),
                solution: best_solution.clone(),
                score: best_score,
            };
            callback(progress);
        }

        let best_solution = (*best_solution.borrow()).clone();
        StepResult {
            best_solution,
            best_score,
            last_solution: current_solution,
            last_score: current_score,
            acceptance_counter,
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for SimulatedAnnealingOptimizer {
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
