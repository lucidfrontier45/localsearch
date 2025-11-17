use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::{distr::weighted::WeightedIndex, prelude::Distribution};
use rayon::prelude::*;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{
    LocalSearchOptimizer, metropolis,
    simulated_annealing::{tune_cooling_rate, tune_initial_temperature},
};

/// Optimizer that implements the population annealing algorithm
/// It runs multiple simulated annealing processes and periodically updates the population
/// by discarding bad candidates and copying good ones.
pub struct PopulationAnnealingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Initial temperature
    initial_temperature: f64,
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
    /// - `initial_temperature` : initial temperature
    /// - `cooling_rate` : cooling rate
    /// - `update_frequency` : number of steps to run each simulated annealing before updating the population
    /// - `population_size` : number of simulated annealing processes to run in parallel
    pub fn new(
        patience: usize,
        n_trials: usize,
        initial_temperature: f64,
        cooling_rate: f64,
        update_frequency: usize,
        population_size: usize,
    ) -> Self {
        Self {
            patience,
            n_trials,
            initial_temperature,
            cooling_rate,
            update_frequency,
            population_size,
        }
    }

    /// Tune initial temperature by drawing random trials
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

    /// Tune cooling rate to reach near-zero temperature at the end of optimization
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
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M>
    for PopulationAnnealingOptimizer
{
    /// Start optimization
    ///
    /// - `model`: the model to optimize
    /// - `initial_solution`: the initial solution to start optimization. If None, a random solution will be generated.
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
        let mut accepted_counter = 0;

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

        let mut current_temperature = self.initial_temperature;
        let mut iter = 0;
        let mut stagnation_counter = 0;

        // Main optimization loop
        while iter < n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }

            let metropolis = metropolis::MetropolisOptimizer::new(
                usize::MAX,
                self.n_trials,
                usize::MAX,
                current_temperature,
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

            // Update best solution and score
            let best_step_result = step_results.iter().min_by_key(|r| r.best_score).unwrap();
            if best_step_result.best_score < best_score {
                best_score = best_step_result.best_score;
                best_solution.replace(best_step_result.best_solution.clone());
                stagnation_counter = 0;
            }

            // Update accepted counter
            let n_accepted: usize = step_results
                .iter()
                .map(|r| r.accepted_transitions.len())
                .sum();
            accepted_counter += n_accepted / self.population_size;

            // Update stagnation counter
            stagnation_counter += self.update_frequency;

            // Update algorithm-specific state
            current_temperature *= self.cooling_rate;
            let new_population: Vec<(M::SolutionType, M::ScoreType)> = step_results
                .into_iter()
                .map(|r| (r.last_solution, r.last_score))
                .collect();

            // Population update: resample based on Boltzmann distribution weights
            // Calculate weights for each solution based on the current temperature
            let mut weights = Vec::new();
            for &(_, score) in &new_population {
                // Boltzmann factor: exp(-score / temperature)
                let boltzmann_factor = (-score.into_inner() / current_temperature).exp().max(1e-8);
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

            // Check patience
            if stagnation_counter >= self.patience {
                break;
            }

            // Update iteration counter
            iter += self.update_frequency;

            // Invoke callback
            let progress =
                OptProgress::new(iter, accepted_counter, best_solution.clone(), best_score);
            callback(progress);
        }

        let final_best_solution = (*best_solution.borrow()).clone();
        (final_best_solution, best_score)
    }
}
