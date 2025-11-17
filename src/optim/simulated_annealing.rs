use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{LocalSearchOptimizer, MetropolisOptimizer, generic::StepResult};

/// Tune cooling rate based on initial and final temperatures and number of iterations
/// initial temperature will be cooled to final temperature after n_iter iterations
/// - `initial_temperature` : initial temperature
/// - `final_temperature` : final temperature
/// - `n_iter` : number of iterations
/// - returns : cooling rate
pub fn tune_cooling_rate(initial_temperature: f64, final_temperature: f64, n_iter: usize) -> f64 {
    (final_temperature / initial_temperature).powf(1.0 / n_iter as f64)
}

pub fn tune_initial_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
    model: &M,
    initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
    n_warmup: usize,
    target_initial_prob: f64,
) -> f64 {
    // 1. First run n_warmup completely random iterations from initial_solution
    // 2. calculate initial_temperature so that the average acceptance probability of sampled trial solutions are target_initial_prob
    // 3. return new SimulatedAnnealingOptimizer with updated temperatures
    let mut rng = rand::rng();
    let (mut current_solution, mut current_score) =
        initial_solution_and_score.unwrap_or(model.generate_random_solution(&mut rng).unwrap());

    let mut energy_diffs = Vec::new();

    for _ in 0..n_warmup {
        let (trial_solution, _, trial_score) =
            model.generate_trial_solution(current_solution.clone(), current_score, &mut rng);
        let ds = trial_score - current_score;
        if ds <= NotNan::new(0.0).unwrap() {
            continue;
        }
        energy_diffs.push(ds.into_inner());
        current_solution = trial_solution;
        current_score = trial_score;
    }

    // Calculate initial_temperature based on target_initial_prob
    // p = exp(-ds / T) => T = -ds / ln(p)
    // Average across all energy differences

    if energy_diffs.is_empty() {
        1.0
    } else {
        let avg_energy_diff = energy_diffs.iter().sum::<f64>() / energy_diffs.len() as f64;
        let ln_prob = target_initial_prob.ln().clamp(-100.0, -0.01);
        (-avg_energy_diff / ln_prob).max(1.0)
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
    /// Initial temperature
    initial_temperature: f64,
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
    /// - `initial_temperature` : initial temperature
    /// - `cooling_rate` : cooling rate
    /// - `update_frequency` : number of steps after which temperature is updated
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_temperature: f64,
        cooling_rate: f64,
        update_frequency: usize,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            initial_temperature,
            cooling_rate,
            update_frequency,
        }
    }
}

impl SimulatedAnnealingOptimizer {
    /// Tune temperature parameters based on initial random trials
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
        let tuned_temperature =
            tune_initial_temperature(model, initial_solution, n_warmup, target_initial_prob);

        Self {
            initial_temperature: tuned_temperature,
            ..self
        }
    }

    /// Tune cooling rate based on self.initial_temperature, final temperature of 1e-2
    pub fn tune_cooling_rate(self, n_iter: usize) -> Self {
        let cooling_rate = tune_cooling_rate(
            self.initial_temperature,
            1e-2,
            n_iter / self.update_frequency,
        );

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
        let mut current_temperature = self.initial_temperature;
        let mut current_solution = initial_solution;
        let mut current_score = initial_score;
        let best_solution = Rc::new(RefCell::new(current_solution.clone()));
        let mut best_score = current_score;
        let mut iter = 0;
        let mut stagnation_counter = 0;
        let now = Instant::now();
        let mut remaining_time_limit = time_limit;
        let mut accepted_counter = 0;
        let mut accepted_transitions = Vec::with_capacity(n_iter);
        let mut rejected_transitions = Vec::with_capacity(n_iter);

        while iter < n_iter {
            let metropolis = MetropolisOptimizer::new(
                usize::MAX,
                self.n_trials,
                usize::MAX,
                current_temperature,
            );
            // make dummy callback
            let mut dummy_callback = |_: OptProgress<M::SolutionType, M::ScoreType>| {};
            let step_result = metropolis.step(
                model,
                current_solution,
                current_score,
                self.update_frequency,
                remaining_time_limit,
                &mut dummy_callback,
            );
            stagnation_counter += self.update_frequency;

            // update current solution
            current_solution = step_result.last_solution;
            current_score = step_result.last_score;

            // update best solution and best score
            if step_result.best_score < best_score {
                best_solution.replace(step_result.best_solution);
                best_score = step_result.best_score;
                stagnation_counter = 0;
            }

            let n_accepted = step_result.accepted_transitions.len();
            accepted_counter += n_accepted;
            accepted_transitions.extend(step_result.accepted_transitions);
            rejected_transitions.extend(step_result.rejected_transitions);

            // check patience
            if stagnation_counter >= self.return_iter {
                current_solution = (*best_solution.borrow()).clone();
                current_score = best_score;
            }
            if stagnation_counter >= self.patience {
                break;
            }

            // update temperature
            current_temperature *= self.cooling_rate;

            // update time limit
            let elapsed = now.elapsed();
            if elapsed >= time_limit {
                break;
            }
            remaining_time_limit = time_limit - elapsed;

            // update iter
            iter += self.update_frequency;

            // run callback
            let progress = OptProgress {
                iter,
                accepted_count: accepted_counter,
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
            accepted_transitions,
            rejected_transitions,
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
