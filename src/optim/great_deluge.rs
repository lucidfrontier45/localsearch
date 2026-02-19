use std::{cell::RefCell, rc::Rc};

use ordered_float::NotNan;

use super::{GenericLocalSearchOptimizer, base::LocalSearchOptimizer};
use crate::{
    Duration, OptModel,
    callback::{OptCallbackFn, OptProgress},
};

/// Optimizer that implements the Great Deluge Algorithm (GDA).
/// Unlike probabilistic methods like simulated annealing, GDA uses a deterministic
/// threshold ("water level") that decreases adaptively over iterations.
/// A trial solution is accepted if its score is below or equal to the current water level.
#[derive(Clone, Copy)]
pub struct GreatDelugeOptimizer {
    /// Patience: the optimizer will give up if there is no improvement after this many iterations
    patience: usize,
    /// Number of trial solutions to generate and evaluate at each iteration
    n_trials: usize,
    /// Return to the current best solution if there is no improvement after this many iterations
    return_iter: usize,
    /// Factor to initialize the water level as initial_score * level_factor
    level_factor: f64,
}

impl GreatDelugeOptimizer {
    /// Constructor for GreatDelugeOptimizer
    ///
    /// - `patience`: the optimizer will give up if there is no improvement after this many iterations
    /// - `n_trials`: number of trial solutions to generate and evaluate at each iteration
    /// - `return_iter`: returns to the current best solution if there is no improvement after this many iterations
    /// - `level_factor`: multiplier for initial water level (e.g., 1.1 for 10% above initial score)
    pub fn new(patience: usize, n_trials: usize, return_iter: usize, level_factor: f64) -> Self {
        Self {
            patience,
            n_trials,
            return_iter,
            level_factor,
        }
    }
}

impl<M: OptModel<ScoreType = NotNan<f64>>> LocalSearchOptimizer<M> for GreatDelugeOptimizer {
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
        // Initialize water level
        let initial_level = initial_score.into_inner() * self.level_factor;
        let water_level = Rc::new(RefCell::new(initial_level));

        let transition_fn = {
            // Clone the Rc so the closure owns its water_level reference, similar to SimulatedAnnealingOptimizer
            let water_level = Rc::clone(&water_level);
            move |_current: NotNan<f64>, trial: NotNan<f64>| -> f64 {
                let wl = *water_level.borrow();
                if trial.into_inner() < wl { 1.0 } else { 0.0 }
            }
        };

        let optimizer = GenericLocalSearchOptimizer::new(
            self.patience,
            self.n_trials,
            self.return_iter,
            transition_fn,
        );

        let mut wrapped_callback = |progress: OptProgress<M::SolutionType, M::ScoreType>| {
            // Update water level using the current best score from progress
            let progress_ratio = (progress.iter as f64) / (n_iter as f64);
            let best_f = progress.score.into_inner();
            let new_level = initial_level - (initial_level - best_f) * progress_ratio;
            water_level.replace(new_level);

            // Call the original callback
            callback(progress);
        };

        optimizer.optimize(
            model,
            initial_solution,
            initial_score,
            n_iter,
            time_limit,
            &mut wrapped_callback,
        )
    }
}
