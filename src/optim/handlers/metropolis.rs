use ordered_float::NotNan;
use rayon::prelude::*;

use crate::{
    OptModel,
    optim::transition::{TransitionHandler, UpdateCtx},
};

/// Classic Metropolis acceptance with a constant inverse temperature `beta`.
#[derive(Clone, Copy, Debug)]
pub struct Metropolis {
    /// Inverse temperature.
    pub beta: f64,
}

impl Metropolis {
    /// Constructor.
    pub const fn new(beta: f64) -> Self {
        Self { beta }
    }
}

/// Tune inverse temperature `beta` based on initial random trials.
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

/// Collect positive energy differences from warmup trials.
pub(crate) fn gather_energy_diffs<M: OptModel<ScoreType = NotNan<f64>>>(
    model: &M,
    initial_solution_and_score: Option<(M::SolutionType, M::ScoreType)>,
    n_warmup: usize,
) -> Vec<f64> {
    let mut rng = rand::rng();
    let (current_solution, current_score) =
        initial_solution_and_score.unwrap_or(model.generate_random_solution(&mut rng).unwrap());

    (0..n_warmup)
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
        .collect()
}

/// Calculate target based on target_prob.
///
/// `p = exp(-beta * ds)` => `beta = -ln(p) / ds`.
/// Average across all energy differences.
pub(crate) fn calculate_temperature_from_acceptance_prob(
    energy_diffs: &[f64],
    target_acceptance_prob: f64,
) -> f64 {
    let average_energy_diff = energy_diffs.iter().sum::<f64>() / energy_diffs.len() as f64;
    let ln_prob = target_acceptance_prob.ln().clamp(-100.0, -0.01);
    -ln_prob / average_energy_diff.clamp(0.01, 100.0)
}

/// Calculate Metropolis acceptance probability for a transition.
pub(crate) fn metropolis_probability(beta: f64, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
    let ds = trial - current;
    if ds <= NotNan::new(0.0).unwrap() {
        1.0
    } else {
        (-beta * ds.into_inner()).exp()
    }
}

impl TransitionHandler<NotNan<f64>> for Metropolis {
    fn update(&mut self, _ctx: &UpdateCtx<'_, NotNan<f64>>) {}

    fn evaluate(&self, current: NotNan<f64>, trial: NotNan<f64>) -> f64 {
        metropolis_probability(self.beta, current, trial)
    }
}
