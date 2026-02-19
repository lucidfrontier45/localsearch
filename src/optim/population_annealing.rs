use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::{RngExt as _, distr::weighted::WeightedIndex, prelude::Distribution};
use rayon::prelude::*;

use super::{
    LocalSearchOptimizer, metropolis, metropolis::tune_temperature,
    simulated_annealing::tune_cooling_rate,
};
use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

/// Optimizer that implements the population annealing algorithm
/// It runs multiple simulated annealing processes and periodically updates the population
/// by discarding bad candidates and copying good ones.
pub struct PopulationAnnealingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Number of iterations without improvement before reverting to the best solution
    return_iter: usize,
    /// Initial inverse temperature
    initial_beta: f64,
    /// Cooling rate
    cooling_rate: f64,
    /// Number of steps to run each simulated annealing before updating the population
    update_frequency: usize,
    /// Number of simulated annealing processes to run in parallel
    population_size: usize,
}

impl PopulationAnnealingOptimizer {
    /// Constructor of PopulationAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `initial_beta` : initial inverse temperature
    /// - `cooling_rate` : cooling rate
    /// - `update_frequency` : number of steps to run each simulated annealing before updating the population
    /// - `population_size` : number of simulated annealing processes to run in parallel
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_beta: f64,
        cooling_rate: f64,
        update_frequency: usize,
        population_size: usize,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            initial_beta,
            cooling_rate,
            update_frequency,
            population_size,
        }
    }

    /// Tune initial inverse temperature by drawing random trials
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

    /// Tune cooling rate to reach high inverse temperature (beta ~ 1e2) at the end of optimization
    pub fn tune_cooling_rate(self, n_iter: usize) -> Self {
        let cooling_rate =
            tune_cooling_rate(self.initial_beta, 1e2, n_iter / self.update_frequency);

        Self {
            cooling_rate,
            ..self
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M>
    for PopulationAnnealingOptimizer
{
    /// Start optimization
    ///
    /// - `model`: the model to optimize
    /// - `initial_solution`: the initial solution to start optimization
    /// - `initial_score`: the initial score of the initial solution
    /// - `n_iter`: maximum iterations
    /// - `time_limit`: maximum iteration time
    /// - `callback`: callback function that will be invoked at the end of each iteration
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> (M::SolutionType, M::ScoreType) {
        let start_time = Instant::now();
        let mut rng = rand::rng();

        // Initialize population with random solutions or copies of the initial solution
        let mut population: Vec<(M::SolutionType, M::ScoreType)> =
            Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            // Generate a random solution for other members
            let (solution, _, score) =
                model.generate_trial_solution(initial_solution.clone(), initial_score, &mut rng);
            population.push((solution, score));
        }

        // Track the best solution found
        let best_solution = Rc::new(RefCell::new(initial_solution.clone()));
        let mut best_score = initial_score;

        // Update the best solution if we found a better one in the initial population
        for (solution, score) in &population {
            if *score < best_score {
                best_solution.replace(solution.clone());
                best_score = *score;
            }
        }

        let mut current_beta = self.initial_beta;
        let mut iter = 0;
        // Separate counters for return-to-best and patience
        let mut return_stagnation_counter = 0;
        let mut patience_stagnation_counter = 0;

        // Main optimization loop
        while iter < n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }

            let metropolis = metropolis::MetropolisOptimizer::new(
                self.patience,
                self.n_trials,
                self.return_iter,
                current_beta,
            );

            // Process each member of the population
            let step_results = population
                .par_iter()
                .map(|(solution, score)| {
                    // Run SA for n_population_update steps
                    let temp_callback =
                        &mut |_progress: OptProgress<M::SolutionType, M::ScoreType>| {};

                    metropolis.step(
                        model,
                        solution.clone(),
                        *score,
                        self.update_frequency,
                        time_limit.saturating_sub(duration),
                        temp_callback,
                    )
                })
                .collect::<Vec<_>>();

            // 1. Update time and iteration counters
            iter += self.update_frequency;

            // 2. Update best solution and score
            let best_step_result = step_results.iter().min_by_key(|r| r.best_score).unwrap();
            if best_step_result.best_score < best_score {
                best_score = best_step_result.best_score;
                best_solution.replace(best_step_result.best_solution.clone());
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            } else {
                return_stagnation_counter += self.update_frequency;
                patience_stagnation_counter += self.update_frequency;
            }

            // 3. Update accepted counter
            let acceptance_ratio = step_results
                .iter()
                .map(|r| r.acceptance_counter.acceptance_ratio())
                .sum::<f64>()
                / self.population_size as f64;

            // 4. Update current solution and score (population updated in algo specific)

            // 5. Check and handle return to best
            if return_stagnation_counter >= self.return_iter {
                // randomly select a member to revert
                let idx = rng.random_range(0..self.population_size);
                population[idx] = ((*best_solution.borrow()).clone(), best_score);
                return_stagnation_counter = 0;
            }

            // 6. Check patience
            if patience_stagnation_counter >= self.patience {
                break;
            }

            // 7. Update algorithm-specific state
            current_beta *= self.cooling_rate;
            let new_population: Vec<(M::SolutionType, M::ScoreType)> = step_results
                .into_iter()
                .map(|r| (r.last_solution, r.last_score))
                .collect();

            // Population update: resample based on Boltzmann distribution weights
            // Calculate weights for each solution based on the current temperature
            let mut weights = Vec::new();
            for &(_, score) in &new_population {
                // Boltzmann factor: exp(-beta * score)
                let boltzmann_factor = (-current_beta * score.into_inner()).exp().max(1e-8);
                weights.push(boltzmann_factor);
            }

            // normalize weights
            let weight_sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= weight_sum;
            }

            // Use stochastic universal sampling or roulette wheel sampling
            let slice_sampler = WeightedIndex::new(&weights).unwrap();
            (0..self.population_size).for_each(|i| {
                let idx = slice_sampler.sample(&mut rng);
                population[i] = new_population[idx].clone();
            });

            // 8. Invoke callback
            let progress =
                OptProgress::new(iter, acceptance_ratio, best_solution.clone(), best_score);
            callback(progress);
        }

        let final_best_solution = (*best_solution.borrow()).clone();
        (final_best_solution, best_score)
    }
}
