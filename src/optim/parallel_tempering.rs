use std::{cell::RefCell, num::NonZero, rc::Rc};

use ordered_float::NotNan;
use rand::RngExt as _;
use rayon::prelude::*;

use super::{
    LocalSearchLoop, LocalSearchOptimizer, Metropolis, calculate_temperature_from_acceptance_prob,
    search_loop::{derive_seed, make_master_rng},
};

use crate::{
    Duration, Instant, OptModel,
    callback::{OptCallbackFn, OptProgress},
    optim::StepResult,
};

/// Parallel Tempering (Replica Exchange) optimizer
/// Runs multiple Metropolis replicas at different inverse temperatures (betas).
pub struct ParallelTemperingOptimizer {
    patience: usize,
    n_trials: usize,
    return_iter: usize,
    betas: Vec<f64>,
    update_frequency: NonZero<usize>,
    /// RNG seed for bit-reproducible runs. `None` (default) preserves the
    /// entropy-driven behavior; set via [`Self::with_seed`].
    seed: Option<u64>,
}

impl ParallelTemperingOptimizer {
    /// Create a ParallelTemperingOptimizer with explicit beta ladder
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        betas: Vec<f64>,
        update_frequency: NonZero<usize>,
    ) -> Self {
        if betas.is_empty() {
            panic!("betas must contain at least one replica");
        }
        Self {
            patience,
            n_trials,
            return_iter,
            betas,
            update_frequency,
            seed: None,
        }
    }

    /// Private constructor used by both [`Self::new`] (which carries no seed)
    /// and the seed-preserving `tune_temperature` rebuilds. Lets every code
    /// path route through a single `Self { ... }` literal that does not
    /// silently drop the seed.
    fn new_with_seed(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        betas: Vec<f64>,
        update_frequency: NonZero<usize>,
        seed: Option<u64>,
    ) -> Self {
        if betas.is_empty() {
            panic!("betas must contain at least one replica");
        }
        Self {
            patience,
            n_trials,
            return_iter,
            betas,
            update_frequency,
            seed,
        }
    }

    /// Pin the RNG seed so [`Self::optimize`] (and the seed-aware tune
    /// helpers) yield bit-identical `(solution, score)` across calls with
    /// the same inputs.
    ///
    /// `None` (the default) keeps the historical entropy-driven behavior.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

/// Build `n_replicas` betas geometrically spaced between `beta_min` and
/// `beta_max`. Single-replica case collapses to `beta_min`. `n_replicas == 0`
/// is a precondition violation shared by every caller.
fn geometric_betas(n_replicas: usize, beta_min: f64, beta_max: f64) -> Vec<f64> {
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
    betas
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
        update_frequency: NonZero<usize>,
    ) -> Self {
        let betas = Self::geometric_betas(n_replicas, beta_min, beta_max);
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
        // salt 4: warmup trial stream is decorrelated from any opt-phase
        // stream; flows into the derived beta ladder so the LHS remains
        // reproducible when `seed` is set.
        let energy_diffs = super::gather_energy_diffs(
            model,
            initial_solution,
            n_warmup,
            self.seed.map(|s| derive_seed(s, 4)),
        );
        if energy_diffs.is_empty() {
            return self;
        }
        let beta_max = calculate_temperature_from_acceptance_prob(&energy_diffs, target_max_prob);
        let beta_min = calculate_temperature_from_acceptance_prob(&energy_diffs, target_min_prob);
        let n_replicas = self.betas.len();
        let betas = Self::geometric_betas(n_replicas, beta_min, beta_max);
        Self::new_with_seed(
            self.patience,
            self.n_trials,
            self.return_iter,
            betas,
            self.update_frequency,
            self.seed,
        )
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for ParallelTemperingOptimizer {
    fn rng_seed(&self) -> Option<u64> {
        self.seed
    }

    /// Start optimization
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
        // salt 2: master RNG for replica swap and return-to-best decisions.
        let mut rng = make_master_rng(self.seed.map(|s| derive_seed(s, 2)));

        let n_replicas = self.betas.len();

        // Initialize replicas: first replica uses provided initial solution
        let mut replicas: Vec<(M::SolutionType, M::ScoreType)> =
            vec![(initial_solution.clone(), initial_score); n_replicas];

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
            let update_freq = self.update_frequency.get();
            let time_remaining = time_limit.saturating_sub(elapsed);

            // Keep a clone of current replicas for parallel processing
            type ReplicaResult<M> = (
                StepResult<<M as OptModel>::SolutionType, <M as OptModel>::ScoreType>,
                Metropolis,
            );
            let step_results: Vec<ReplicaResult<M>> = replicas
                .par_iter()
                .enumerate()
                .map(|(idx, (sol, score))| {
                    // double-salt: per-replica loop RNG is decorrelated from
                    // both the outer master (salt 2) AND from the warmup trial
                    // stream (salt 4). Removing either downstream salt would not
                    // silently alias replica 0 onto warmup.
                    let loop_seed = self
                        .seed
                        .map(|s| derive_seed(derive_seed(s, 4), idx as u64));
                    let opt = match loop_seed {
                        Some(s) => LocalSearchLoop::new(self.patience, n_trials, self.return_iter)
                            .with_seed(s),
                        None => LocalSearchLoop::new(self.patience, n_trials, self.return_iter),
                    };
                    let mut cb = &mut |_p: OptProgress<M::SolutionType, M::ScoreType>| {};
                    opt.step(
                        model,
                        sol.clone(),
                        *score,
                        update_freq,
                        time_remaining,
                        &mut cb,
                        Metropolis::new(self.betas[idx]),
                    )
                })
                .collect();

            // 1. Update time and iteration counters
            iter = iter.saturating_add(update_freq);

            // 2. Update best solution and score based on step_results
            let best_step_result = step_results.iter().min_by_key(|r| r.0.best_score).unwrap();
            if best_step_result.0.best_score < best_score {
                best_score = best_step_result.0.best_score;
                best_solution.replace(best_step_result.0.best_solution.clone());
                return_stagnation_counter = 0;
                patience_stagnation_counter = 0;
            } else {
                return_stagnation_counter = return_stagnation_counter.saturating_add(update_freq);
                patience_stagnation_counter =
                    patience_stagnation_counter.saturating_add(update_freq);
            }

            // 3. Compute acceptance ratio
            let acceptance_ratio = {
                let mut sum = 0.0;
                for r in step_results.iter() {
                    sum += r.0.acceptance_counter.acceptance_ratio();
                }
                sum / n_replicas as f64
            };

            // 4. Update current solution and score from step results
            for (i, r) in step_results.into_iter().enumerate() {
                replicas[i] = (r.0.last_solution, r.0.last_score);
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
