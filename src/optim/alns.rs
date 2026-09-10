//! Adaptive Large Neighborhood Search (ALNS) segment runner.
//!
//! This module provides a segment-level ALNS demo built on top of
//! [`GenericLocalSearchOptimizer`](super::GenericLocalSearchOptimizer).
//! Destroy/repair (LNS) logic stays model-side inside
//! [`OptModel::generate_trial_solution`](crate::OptModel::generate_trial_solution);
//! this runner only adapts operator weights from segment feedback and reports
//! per-operator statistics via [`StepResult::output`](super::StepResult).
//!
//! [`TransitionType`](crate::OptModel::TransitionType) keeps its move-level
//! semantics (used by Tabu search); ALNS statistics never flow through it.

use super::{generic::StepResult, GenericLocalSearchOptimizer, TransitionProbabilityFn};
use crate::{callback::OptCallbackFn, Duration, OptModel};

/// Per-segment operator statistics returned as [`StepResult::output`](super::StepResult).
#[derive(Clone, Debug, Default)]
pub struct AlnsStatistics {
    /// Number of trial proposals attributed to each operator during the segment.
    pub operator_uses: Vec<usize>,
    /// Credit accumulated by each operator during the segment.
    pub operator_scores: Vec<f64>,
}

/// Model-side hook for ALNS operator bookkeeping.
///
/// Implementations select among `n_operators()` destroy/repair operators inside
/// [`OptModel::generate_trial_solution`](crate::OptModel::generate_trial_solution),
/// using the weights installed via [`set_operator_weights`](Self::set_operator_weights),
/// and count per-operator uses plus improving proposals (trials scoring strictly
/// better than the current score at proposal time).
///
/// Counters must use thread-safe interior mutability (e.g. atomics), because trial
/// generation runs in parallel via Rayon. [`drain_operator_stats`](Self::drain_operator_stats)
/// returns deltas since the previous drain and resets the counters.
pub trait AlnsOperatorModel: OptModel {
    /// Number of destroy/repair operators.
    fn n_operators(&self) -> usize;
    /// Install the current operator weights for roulette-wheel selection.
    fn set_operator_weights(&self, weights: &[f64]);
    /// Drain `(uses, improvements)` deltas per operator since the last drain.
    fn drain_operator_stats(&self) -> (Vec<usize>, Vec<usize>);
}

/// Segment-level ALNS optimizer wrapping [`GenericLocalSearchOptimizer`].
///
/// Each [`run_segment`](Self::run_segment) call installs the current weights,
/// runs one inner [`step`](GenericLocalSearchOptimizer::step) with `O = ()`,
/// classifies the segment outcome (new global best / improved current /
/// accepted / rejected) from the score deltas plus the acceptance counter,
/// credits operators, and updates weights with exponential smoothing
/// `w = (1 - r) * w + r * (score / uses)`.
pub struct AlnsOptimizer<ST: Ord + Sync + Send + Copy, FT: TransitionProbabilityFn<ST>> {
    inner: GenericLocalSearchOptimizer<ST, FT>,
    weights: Vec<f64>,
    reaction: f64,
    sigma_new_best: f64,
    sigma_improved: f64,
    sigma_accepted: f64,
}

impl<ST: Ord + Sync + Send + Copy, FT: TransitionProbabilityFn<ST>> AlnsOptimizer<ST, FT> {
    /// Create an ALNS runner with uniform weights and classic-style defaults
    /// (`reaction = 0.1`, `sigma_new_best = 10.0`, `sigma_improved = 5.0`,
    /// `sigma_accepted = 1.0`).
    ///
    /// - `patience` / `n_trials` / `return_iter` / `score_func` are forwarded to the inner optimizer.
    /// - `n_operators` is the number of destroy/repair operators.
    /// - `reaction` is the exponential smoothing factor in `[0, 1]`.
    pub fn new(
        patience: usize,
        n_trials: usize,
        return_iter: usize,
        score_func: FT,
        n_operators: usize,
        reaction: f64,
    ) -> Self {
        Self {
            inner: GenericLocalSearchOptimizer::new(patience, n_trials, return_iter, score_func),
            weights: vec![1.0; n_operators],
            reaction,
            sigma_new_best: 10.0,
            sigma_improved: 5.0,
            sigma_accepted: 1.0,
        }
    }

    /// Override the default score increments
    /// (`sigma_new_best`, `sigma_improved`, `sigma_accepted`).
    pub fn with_sigmas(
        mut self,
        sigma_new_best: f64,
        sigma_improved: f64,
        sigma_accepted: f64,
    ) -> Self {
        self.sigma_new_best = sigma_new_best;
        self.sigma_improved = sigma_improved;
        self.sigma_accepted = sigma_accepted;
        self
    }

    /// Current operator weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Run one ALNS segment and adapt operator weights from segment feedback.
    ///
    /// - `current_solution` / `current_score` seed the inner search.
    /// - `best_score` is the incoming global best, used to detect the
    ///   new-global-best category.
    /// - Returns the inner step result augmented with [`AlnsStatistics`] output.
    #[allow(clippy::too_many_arguments)]
    pub fn run_segment<M: AlnsOperatorModel<ScoreType = ST>>(
        &mut self,
        model: &M,
        current_solution: M::SolutionType,
        current_score: M::ScoreType,
        best_score: M::ScoreType,
        n_iter: usize,
        time_limit: Duration,
        callback: &mut dyn OptCallbackFn<M::SolutionType, M::ScoreType>,
    ) -> StepResult<M::SolutionType, M::ScoreType, AlnsStatistics> {
        if model.n_operators() != self.weights.len() {
            panic!(
                "AlnsOptimizer configured for {} operators but model reports {}",
                self.weights.len(),
                model.n_operators()
            );
        }
        model.set_operator_weights(&self.weights);

        let inner: StepResult<M::SolutionType, M::ScoreType, ()> = self.inner.step(
            model,
            current_solution,
            current_score,
            n_iter,
            time_limit,
            callback,
        );
        let (uses, improvements) = model.drain_operator_stats();

        let new_best = inner.best_score < best_score;
        let improved = inner.last_score < current_score;
        let accepted = inner.acceptance_counter.acceptance_ratio() > 0.0;

        let total_improvements: usize = improvements.iter().sum();
        let total_uses: usize = uses.iter().sum();
        let mut operator_scores = vec![0.0; self.weights.len()];
        for (i, score) in operator_scores.iter_mut().enumerate() {
            let improving = improvements[i] as f64;
            *score += self.sigma_improved * improving;
            if new_best && total_improvements > 0 {
                *score += self.sigma_new_best * improving / total_improvements as f64;
            } else if accepted && !improved && total_uses > 0 {
                // Accepted without improvement: participation credit spread over uses.
                *score += self.sigma_accepted * uses[i] as f64 / total_uses as f64;
            }
        }

        for (i, weight) in self.weights.iter_mut().enumerate() {
            let average = operator_scores[i] / uses[i].max(1) as f64;
            *weight = (1.0 - self.reaction) * *weight + self.reaction * average;
        }

        StepResult {
            best_solution: inner.best_solution,
            best_score: inner.best_score,
            last_solution: inner.last_solution,
            last_score: inner.last_score,
            acceptance_counter: inner.acceptance_counter,
            output: AlnsStatistics {
                operator_uses: uses,
                operator_scores,
            },
        }
    }
}
