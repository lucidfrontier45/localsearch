use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::{distr::weighted::WeightedIndex, prelude::Distribution};

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

use super::{LocalSearchOptimizer, simulated_annealing::SimulatedAnnealingOptimizer};

/// Optimizer that implements the population annealing algorithm
/// It runs multiple simulated annealing processes and periodically updates the population
/// by discarding bad candidates and copying good ones.
pub struct PopulationAnnealingOptimizer {
    /// Base Simulated Annealing Optimizer
    pub base_sa: SimulatedAnnealingOptimizer,
    /// Number of simulated annealing processes to run in parallel
    pub n_population: usize,
    /// Number of steps to run each simulated annealing before updating the population
    pub n_population_update: usize,
}

impl PopulationAnnealingOptimizer {
    /// Constructor of PopulationAnnealingOptimizer
    /// - `base_sa`: base Simulated Annealing Optimizer
    /// - `n_population`: number of simulated annealing processes to run in parallel
    /// - `n_population_update`: number of steps to run each simulated annealing before updating the population
    pub fn new(
        base_sa: SimulatedAnnealingOptimizer,
        n_population: usize,
        n_population_update: usize,
    ) -> Self {
        Self {
            base_sa,
            n_population,
            n_population_update,
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

        // Initialize population with random solutions or copies of the initial solution
        let mut population: Vec<(M::SolutionType, M::ScoreType)> =
            Vec::with_capacity(self.n_population);

        for _ in 0..self.n_population {
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

        let mut current_temperature = self.base_sa.initial_temperature;
        let mut iter = 0;
        let accepted_counter = 0;
        let mut patience_counter = 0;

        // Main optimization loop
        while iter < n_iter {
            let duration = Instant::now().duration_since(start_time);
            if duration > time_limit {
                break;
            }

            // Run each simulated annealing process for n_population_update steps
            let mut new_population = Vec::with_capacity(self.n_population);

            // Process each member of the population
            let mut next_temperature = current_temperature;
            for (solution, score) in population.iter() {
                // Run SA for n_population_update steps
                let temp_callback = &mut |_progress: OptProgress<M::SolutionType, M::ScoreType>| {};

                let (updated_solution, updated_score, final_temperature) =
                    self.base_sa.optimize_with_temperature_and_cooling_rate(
                        model,
                        solution.clone(),
                        *score,
                        self.n_population_update,
                        time_limit.saturating_sub(duration),
                        temp_callback,
                        current_temperature,
                        self.base_sa.cooling_rate,
                    );
                next_temperature = final_temperature;
                new_population.push((updated_solution, updated_score));
            }
            current_temperature = next_temperature;

            // Update the best solution if needed
            patience_counter += self.n_population_update;
            for (solution, score) in &new_population {
                if *score < best_score {
                    best_score = *score;
                    best_solution.replace(solution.clone());
                    patience_counter = 0;
                }
            }

            if patience_counter >= self.base_sa.patience {
                break;
            }

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
            (0..self.n_population).for_each(|i| {
                let idx = slice_sampler.sample(&mut rng);
                population[i] = new_population[idx].clone();
            });

            // Invoke callback with progress information
            iter += self.n_population_update;
            let progress =
                OptProgress::new(iter, accepted_counter, best_solution.clone(), best_score);
            callback(progress);
        }

        let final_best_solution = (*best_solution.borrow()).clone();
        (final_best_solution, best_score)
    }
}
