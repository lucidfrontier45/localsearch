use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;
use rand::Rng as _;
use rayon::prelude::*;

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
    optim::metropolis::MetropolisOptimizer,
};

use super::metropolis::{calculate_temperature_from_acceptance_prob, gather_energy_diffs};
use super::{LocalSearchOptimizer, generic::StepResult};

/// Parallel Tempering (Replica Exchange) optimizer
/// Runs multiple Metropolis replicas at different inverse temperatures (betas).
pub struct ParallelTemperingOptimizer {
    /// The optimizer will give up if there is no improvement of the score after this number of iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each Metropolis step
    n_trials: usize,
    /// Returns to the best solution if there is no improvement after this number of iterations
    return_iter: usize,
    /// Vector of inverse temperatures (beta) for replicas
    betas: Vec<f64>,
    /// Number of Metropolis steps to run per replica between exchange attempts
    update_frequency: usize,
}

impl ParallelTemperingOptimizer {
    /// Create a ParallelTemperingOptimizer with explicit beta ladder
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        betas: Vec<f64>,
        update_frequency: usize,
    ) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            betas,
            update_frequency,
        }
    }

    /// Helper to create geometric spaced betas
    ///
    /// Creates `n_replicas` betas geometrically spaced between `beta_min` and `beta_max`.
    pub fn with_geometric_betas(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        n_replicas: usize,
        beta_min: f64,
        beta_max: f64,
        update_frequency: usize,
    ) -> Self {
        let mut betas = Vec::with_capacity(n_replicas);
        if n_replicas == 0 {
            panic!("n_replicas must be >= 1");
        }
        if n_replicas == 1 {
            betas.push(beta_min);
        } else {
            let ratio = (beta_max / beta_min).powf(1.0 / (n_replicas as f64 - 1.0));

            let mut b = beta_min;

            for _ in 0..n_replicas {
                betas.push(b);

                b *= ratio;
            }
        }

        Self::new(patience, n_trials, return_iter, betas, update_frequency)
    }

    /// Tune betas based on initial solution and target acceptance probabilities
    ///
    /// - `model`: the optimization model
    /// - `initial_solution`: the initial solution and score to use for tuning. If None, a random solution will be generated.
    /// - `n_warmup`: number of warmup iterations to gather energy differences
    /// - `target_max_prob`: target acceptance probability for the highest beta (coldest replica)
    /// - `target_min_prob`: target acceptance probability for the lowest beta (hottest replica)
    pub fn tune_temperature<M: OptModel<ScoreType = NotNan<f64>>>(
        self,
        model: &M,
        initial_solution: Option<(M::SolutionType, M::ScoreType)>,
        n_warmup: usize,
        target_max_prob: f64,
        target_min_prob: f64,
    ) -> Self {
        let energy_diffs = gather_energy_diffs(model, initial_solution, n_warmup);
        if energy_diffs.is_empty() {
            return self;
        }
        let beta_max = calculate_temperature_from_acceptance_prob(&energy_diffs, target_max_prob);
        let beta_min = calculate_temperature_from_acceptance_prob(&energy_diffs, target_min_prob);
        let n_replicas = self.betas.len();
        Self::with_geometric_betas(
            self.patience,
            self.n_trials,
            self.return_iter,
            n_replicas,
            beta_min,
            beta_max,
            self.update_frequency,
        )
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for ParallelTemperingOptimizer {
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

        let n_replicas = self.betas.len();

        // Initialize replicas: first replica uses provided initial solution
        let mut replicas: Vec<(M::SolutionType, M::ScoreType)> =
            vec![(initial_solution.clone(), initial_score,); n_replicas];

        let best_solution = Rc::new(RefCell::new(initial_solution.clone()));
        let mut best_score = initial_score;
        for (s, sc) in &replicas {
            if *sc < best_score {
                best_solution.replace(s.clone());
                best_score = *sc;
            }
        }

        let mut iter: usize = 0;
        let mut return_stagnation_counter: usize = 0;
        let mut patience_stagnation_counter: usize = 0;

        while iter < n_iter {
            let elapsed = Instant::now().duration_since(start_time);
            if elapsed > time_limit {
                break;
            }

            // Run Metropolis on each replica in parallel
            let n_trials = self.n_trials;
            let update_freq = self.update_frequency;
            let time_remaining = time_limit.saturating_sub(elapsed);

            // Keep a clone of current replicas for parallel processing
            let step_results: Vec<StepResult<M::SolutionType, M::ScoreType>> = replicas
                .par_iter()
                .enumerate()
                .map(|(idx, (sol, score))| {
                    let m = MetropolisOptimizer::new(
                        self.patience,
                        n_trials,
                        self.return_iter,
                        self.betas[idx],
                    );
                    let mut cb = &mut |_p: OptProgress<M::SolutionType, M::ScoreType>| {};
                    m.step(
                        model,
                        sol.clone(),
                        *score,
                        update_freq,
                        time_remaining,
                        &mut cb,
                    )
                })
                .collect();

            // 1. Update time and iteration counters
            iter = iter.saturating_add(self.update_frequency);

            // 2. Update best solution and score based on step_results
            let best_step_result = step_results.iter().min_by_key(|r| r.best_score).unwrap();
            if best_step_result.best_score < best_score {
                best_score = best_step_result.best_score;
                best_solution.replace(best_step_result.best_solution.clone());
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            } else {
                return_stagnation_counter =
                    return_stagnation_counter.saturating_add(self.update_frequency);
                patience_stagnation_counter =
                    patience_stagnation_counter.saturating_add(self.update_frequency);
            }

            // 3. Compute acceptance ratio
            let acceptance_ratio = {
                let mut sum = 0.0;
                for r in step_results.iter() {
                    sum += r.acceptance_counter.acceptance_ratio();
                }
                sum / n_replicas as f64
            };

            // 4. Update current solution and score from step results
            for (i, r) in step_results.into_iter().enumerate() {
                replicas[i] = (r.last_solution, r.last_score);
            }

            // 5. Check and handle return to best
            if return_stagnation_counter >= self.return_iter {
                let idx = rng.random_range(0..n_replicas);
                replicas[idx] = ((*best_solution.borrow()).clone(), best_score);
                return_stagnation_counter = 0;
            }

            // 6. Check patience
            if patience_stagnation_counter >= self.patience {
                break;
            }

            // 7. Algorithm-specific updates: attempt exchanges between adjacent replicas
            for i in 0..(n_replicas - 1) {
                let sc_i = replicas[i].1;
                let sc_j = replicas[i + 1].1;
                // p_swap = exp((beta_j - beta_i) * (E_j - E_i))
                let exponent = (self.betas[i + 1] - self.betas[i]) * (sc_j - sc_i).into_inner();
                let p_swap = exponent.exp();
                let accept = p_swap >= 1.0 || rng.random::<f64>() < p_swap;
                if accept {
                    replicas.swap(i, i + 1);
                }
            }

            // 8. Invoke callback
            let progress =
                OptProgress::new(iter, acceptance_ratio, best_solution.clone(), best_score);
            callback(progress);
        }

        (best_solution.borrow().clone(), best_score)
    }
}
