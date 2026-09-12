use std::num::NonZero;

use ordered_float::NotNan;

use super::{GenericLocalSearchOptimizer, LocalSearchOptimizer, SimulatedAnnealing};
use crate::{
    Duration, OptModel, callback::OptCallbackFn, optim::metropolis::tune_temperature,
};

/// Tune cooling rate based on initial and final inverse temperatures and number of iterations
/// initial beta will be cooled to final beta after n_iter iterations
/// - `initial_beta` : initial inverse temperature
/// - `final_beta` : final inverse temperature
/// - `n_iter` : number of iterations
/// - returns : cooling rate
pub fn tune_cooling_rate(initial_beta: f64, final_beta: f64, n_iter: usize) -> f64 {
    (final_beta / initial_beta).powf(1.0 / n_iter as f64)
}

/// Optimizer that implements the simulated annealing algorithm
#[derive(Clone, Copy)]
#[allow(dead_code)] // fields document config for callers building the handler
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
    /// Non-zero number of steps after which temperature is updated
    update_frequency: NonZero<usize>,
}

impl SimulatedAnnealingOptimizer {
    /// Constructor of SimulatedAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter` : returns to the current best solution if there is no improvement after this number of iterations.
    /// - `initial_beta` : initial inverse temperature
    /// - `cooling_rate` : cooling rate
    /// - `update_frequency` : non-zero number of steps after which inverse temperature (beta) is updated
    pub const fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        initial_beta: f64,
        cooling_rate: f64,
        update_frequency: NonZero<usize>,
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
            tune_cooling_rate(self.initial_beta, 1e2, n_iter / self.update_frequency.get());

        Self {
            cooling_rate,
            ..self
        }
    }
}

impl<M, H> LocalSearchOptimizer<M, H> for SimulatedAnnealingOptimizer
where
    M: OptModel<ScoreType = NotNan<f64>>,
    H: Into<SimulatedAnnealing> + From<SimulatedAnnealing>,
{
    fn optimize(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        initial_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
        handler: H,
    ) -> (M::SolutionType, M::ScoreType, H) {
        let h: SimulatedAnnealing = handler.into();
        let opt = GenericLocalSearchOptimizer::new(self.patience, self.n_trials, self.return_iter);
        let (solution, score, h) = opt.optimize_with_handler(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            callback,
            h,
        );
        (solution, score, H::from(h))
    }
}
