//! Adaptive Large Neighborhood Search (ALNS).
//!
//! This module implements the trial-generator side of ALNS as a
//! [`TrialGenerator`](super::search_loop::TrialGenerator). The acceptance
//! side stays in the regular
//! [`TransitionHandler`](super::transition::TransitionHandler) machinery;
//! ALNS only contributes the destroy/repair operator selection and the
//! adaptive weight updates.
//!
//! # Overview
//!
//! Each iteration of the search loop calls
//! [`TrialGenerator::generate_trial`](super::search_loop::TrialGenerator::generate_trial).
//! The generator:
//!
//! 1. picks a destroy operator via roulette-wheel selection over its
//!    weights,
//! 2. picks a repair operator the same way (independently),
//! 3. applies `destroy` to produce a partial state `P`,
//! 4. applies `repair` to turn `P` back into a full
//!    `(solution, score)` pair.
//!
//! Once the loop knows what happened to that trial, it calls
//! [`TrialGenerator::feedback`](super::search_loop::TrialGenerator::feedback)
//! with a [`TrialOutcome`](super::search_loop::TrialOutcome). The selected
//! destroy and repair operators accumulate the corresponding reward and the
//! trial counter advances. At every segment boundary the operator weights
//! are recomputed with the reaction-factor blending rule from
//! Ropke & Pisinger (2006).
//!
//! See `src/tests/test_alns.rs` for a complete example.

use rand::{RngExt as _, rngs::StdRng};

use super::search_loop::{TrialGenerator, TrialOutcome};
use crate::OptModel;

/// Per-outcome reward table for ALNS operator credit.
///
/// Defaults follow the classic Ropke & Pisinger scheme (33/9/13/0): a new
/// global best is worth the most, an improvement over the current solution
/// comes next, and an accepted non-improving trial still earns a reward —
/// note the classic scheme rates an accepted move above a merely improving
/// one — while a rejected trial contributes nothing.
#[derive(Debug, Clone, Copy)]
pub struct Rewards {
    /// Reward for a trial that produced a new global best.
    pub new_best: f64,
    /// Reward for a trial that improved over the current solution but did
    /// not become the new global best.
    pub improved: f64,
    /// Reward for a trial that was accepted without improving the current
    /// solution.
    pub accepted: f64,
    /// Reward for a trial that was rejected by the acceptance criterion.
    pub rejected: f64,
}

impl Rewards {
    /// Construct a reward table from the four outcome values.
    pub const fn new(new_best: f64, improved: f64, accepted: f64, rejected: f64) -> Self {
        Self {
            new_best,
            improved,
            accepted,
            rejected,
        }
    }

    /// Look up the reward for a given [`TrialOutcome`].
    fn reward_for(&self, outcome: TrialOutcome) -> f64 {
        match outcome {
            TrialOutcome::NewBest => self.new_best,
            TrialOutcome::Improved => self.improved,
            TrialOutcome::Accepted => self.accepted,
            TrialOutcome::Rejected => self.rejected,
        }
    }
}

impl Default for Rewards {
    fn default() -> Self {
        // Classic Ropke & Pisinger values (33/9/13/0). Note `accepted`
        // intentionally exceeds `improved` in this scheme.
        Self {
            new_best: 33.0,
            improved: 9.0,
            accepted: 13.0,
            rejected: 0.0,
        }
    }
}

/// Adaptive weight and bookkeeping state for a single operator.
#[derive(Debug, Clone, Copy)]
pub struct OperatorStats {
    /// Roulette-wheel weight used for selection.
    pub weight: f64,
    /// Accumulated reward for the current segment.
    pub accumulated_score: f64,
    /// Number of times the operator was applied during the current segment.
    pub usage_count: usize,
}

impl OperatorStats {
    const fn new(weight: f64) -> Self {
        Self {
            weight,
            accumulated_score: 0.0,
            usage_count: 0,
        }
    }
}

/// Destroy or repair operator pool with its adaptive selection state.
///
/// Operators and their [`OperatorStats`] live in one struct so selection,
/// reward credit, and weight updates stay symmetric across both pools.
#[derive(Clone)]
struct OperatorPool<O> {
    operators: Vec<O>,
    stats: Vec<OperatorStats>,
    rewards: Rewards,
}

impl<O> OperatorPool<O> {
    /// Build a pool with every operator at weight `1.0`.
    ///
    /// # Panics
    ///
    /// Panics if `operators` is empty — an empty pool cannot select.
    fn new(operators: Vec<O>, rewards: Rewards) -> Self {
        assert!(!operators.is_empty(), "ALNS requires at least one operator");
        let stats = (0..operators.len())
            .map(|_| OperatorStats::new(1.0))
            .collect();
        Self {
            operators,
            stats,
            rewards,
        }
    }

    /// Pick an operator index by roulette-wheel sampling.
    fn select<R: rand::Rng>(&self, rng: &mut R) -> usize {
        select_by_weight(&self.stats, rng)
    }

    /// Accumulate the outcome reward for the operator at `idx`.
    fn credit(&mut self, idx: usize, outcome: TrialOutcome) {
        self.stats[idx].accumulated_score += self.rewards.reward_for(outcome);
        self.stats[idx].usage_count += 1;
    }

    /// Blend weights with the reaction factor at a segment boundary.
    fn update_weights(&mut self, reaction_factor: f64) {
        apply_weight_update(&mut self.stats, reaction_factor);
    }
}

/// Plug-in destroy operator for ALNS.
///
/// A destroy operator returns a *partial* state of type `P`. The partial
/// state is intentionally free-form — it can be a subset of indices that
/// were removed, a set of moves to undo, or anything the matching
/// [`RepairOperator`] knows how to interpret.
///
/// `solution` is borrowed so the loop can share it across rayon worker
/// threads; clone what you need. `rng` is a per-trial fork supplied by the
/// loop — use it for all randomness instead of creating your own, so runs
/// stay reproducible. The concrete [`StdRng`] type keeps the trait
/// object-safe so operators can be stored as `Box<dyn DestroyOperator<_, _>>`.
///
/// Object-safety is preserved by providing a [`Self::dyn_clone`] method
/// instead of adding [`Clone`] as a supertrait (which would force
/// `Self: Sized`). A blanket [`Clone`] impl on `Box<dyn DestroyOperator<_, _>>`
/// delegates to `dyn_clone`, so operators only need to be `Clone` for the
/// concrete type — see the example in `src/tests/test_alns.rs`.
pub trait DestroyOperator<M: OptModel, P>: Send + Sync {
    /// Apply the destroy step and produce a partial state.
    fn destroy(&self, model: &M, solution: &M::SolutionType, rng: &mut StdRng) -> P;
    /// Clone the operator into a `Box<dyn DestroyOperator>`. Implementors
    /// typically write `Box::new(self.clone())`.
    fn dyn_clone(&self) -> Box<dyn DestroyOperator<M, P>>;
}

/// Plug-in repair operator for ALNS.
///
/// A repair operator consumes the partial state produced by a destroy
/// operator and returns a complete `(solution, score)` pair. Like
/// [`DestroyOperator`], it receives the loop's per-trial [`StdRng`] fork —
/// use it for all randomness.
///
/// See [`DestroyOperator`] for the dyn-clone contract.
pub trait RepairOperator<M: OptModel, P>: Send + Sync {
    /// Apply the repair step and return a complete solution with its score.
    fn repair(&self, model: &M, partial: P, rng: &mut StdRng) -> (M::SolutionType, M::ScoreType);
    /// Clone the operator into a `Box<dyn RepairOperator>`.
    fn dyn_clone(&self) -> Box<dyn RepairOperator<M, P>>;
}

impl<M: OptModel, P> Clone for Box<dyn DestroyOperator<M, P>> {
    fn clone(&self) -> Self {
        self.dyn_clone()
    }
}

impl<M: OptModel, P> Clone for Box<dyn RepairOperator<M, P>> {
    fn clone(&self) -> Self {
        self.dyn_clone()
    }
}

/// Adaptive Large Neighborhood Search trial generator.
///
/// Holds destroy and repair operator pools together with their adaptive
/// weights. Each call to [`TrialGenerator::generate_trial`] picks one
/// destroy and one repair operator via independent roulette-wheel draws,
/// applies them in sequence, and returns the chosen indices as the trial
/// token so that [`TrialGenerator::feedback`] can credit the right
/// operators.
///
/// Generation takes `&self`, so the search loop can run `n_trials`
/// candidates in parallel on rayon worker threads. Only the winning
/// candidate is evaluated per iteration, so only the operator pair that
/// produced the winner is credited (winner-takes-all); the remaining
/// candidates are discarded without reward.
pub struct AlnsTrialGenerator<M: OptModel, P> {
    /// Destroy operators with their adaptive weights and reward table.
    destroy_pool: OperatorPool<Box<dyn DestroyOperator<M, P>>>,
    /// Repair operators with their adaptive weights and reward table.
    repair_pool: OperatorPool<Box<dyn RepairOperator<M, P>>>,
    /// Number of trials per segment — weights are recomputed whenever this
    /// threshold is reached.
    segment_size: usize,
    /// Number of feedback calls since the last weight update.
    trials_in_segment: usize,
    /// Reaction factor `r` in `[0, 1]`. New weight blends the previous
    /// weight and the segment performance as
    /// `w_new = (1 - r) * w_old + r * (segment_score / usage_count)`.
    reaction_factor: f64,
}

impl<M: OptModel, P> AlnsTrialGenerator<M, P> {
    /// Build a new ALNS generator from a pool of destroy and repair
    /// operators. All operators start with weight `1.0`; reward tables
    /// default to the classic Ropke & Pisinger scheme.
    ///
    /// # Panics
    ///
    /// Panics if `destroy_operators` or `repair_operators` is empty — ALNS
    /// cannot operate without at least one operator of each kind.
    pub fn new(
        destroy_operators: Vec<Box<dyn DestroyOperator<M, P>>>,
        repair_operators: Vec<Box<dyn RepairOperator<M, P>>>,
    ) -> Self {
        Self {
            destroy_pool: OperatorPool::new(destroy_operators, Rewards::default()),
            repair_pool: OperatorPool::new(repair_operators, Rewards::default()),
            segment_size: 100,
            trials_in_segment: 0,
            reaction_factor: 0.8,
        }
    }

    /// Replace the segment size (number of trials between weight updates).
    ///
    /// # Panics
    ///
    /// Panics if `segment_size == 0`.
    pub fn with_segment_size(mut self, segment_size: usize) -> Self {
        assert!(segment_size > 0, "segment_size must be positive");
        self.segment_size = segment_size;
        self
    }

    /// Replace the reaction factor `r` used when blending weights.
    ///
    /// Values outside `[0, 1]` are clamped into range.
    pub fn with_reaction_factor(mut self, reaction_factor: f64) -> Self {
        self.reaction_factor = reaction_factor.clamp(0.0, 1.0);
        self
    }

    /// Replace the reward table applied to both destroy and repair operators.
    pub fn with_rewards(mut self, rewards: Rewards) -> Self {
        self.destroy_pool.rewards = rewards;
        self.repair_pool.rewards = rewards;
        self
    }

    /// Adaptive selection state (weight, accumulated reward, usage) of the
    /// destroy-operator pool for the current segment.
    pub fn destroy_stats(&self) -> &[OperatorStats] {
        &self.destroy_pool.stats
    }

    /// Adaptive selection state (weight, accumulated reward, usage) of the
    /// repair-operator pool for the current segment.
    pub fn repair_stats(&self) -> &[OperatorStats] {
        &self.repair_pool.stats
    }

    /// Trials applied to operators during the current segment. Resets to
    /// `0` after each weight update.
    pub fn trials_in_segment(&self) -> usize {
        self.trials_in_segment
    }

    /// Credit one operator pair and advance the segment bookkeeping,
    /// updating weights at segment boundaries.
    fn credit(&mut self, d_idx: usize, r_idx: usize, outcome: TrialOutcome) {
        self.destroy_pool.credit(d_idx, outcome);
        self.repair_pool.credit(r_idx, outcome);
        self.trials_in_segment += 1;
        if self.trials_in_segment >= self.segment_size {
            self.destroy_pool.update_weights(self.reaction_factor);
            self.repair_pool.update_weights(self.reaction_factor);
            self.trials_in_segment = 0;
        }
    }
}

/// Roulette-wheel selection over an operator pool's weights.
fn select_by_weight<R: rand::Rng>(stats: &[OperatorStats], rng: &mut R) -> usize {
    let total: f64 = stats.iter().map(|s| s.weight).sum();
    debug_assert!(total > 0.0, "operator weights must remain positive");
    let target = rng.random::<f64>() * total;
    let mut cumulative = 0.0;
    for (idx, s) in stats.iter().enumerate() {
        cumulative += s.weight;
        if target < cumulative {
            return idx;
        }
    }
    stats.len() - 1
}

/// Recompute weights for one operator pool using the reaction-factor
/// blending rule from Ropke & Pisinger (2006). Operators unused during the
/// segment keep their previous weight.
fn apply_weight_update(stats: &mut [OperatorStats], reaction_factor: f64) {
    for s in stats.iter_mut() {
        if s.usage_count == 0 {
            s.accumulated_score = 0.0;
            continue;
        }
        let segment_avg = s.accumulated_score / s.usage_count as f64;
        s.weight = reaction_factor.mul_add(segment_avg, (1.0 - reaction_factor) * s.weight);
        // Avoid zero/negative weights so roulette-wheel sampling stays
        // well-defined even when every operator did poorly.
        if !s.weight.is_finite() || s.weight <= 0.0 {
            s.weight = 1e-3;
        }
        s.accumulated_score = 0.0;
        s.usage_count = 0;
    }
}

impl<M: OptModel, P> Clone for AlnsTrialGenerator<M, P> {
    fn clone(&self) -> Self {
        Self {
            destroy_pool: self.destroy_pool.clone(),
            repair_pool: self.repair_pool.clone(),
            segment_size: self.segment_size,
            trials_in_segment: self.trials_in_segment,
            reaction_factor: self.reaction_factor,
        }
    }
}

impl<M: OptModel, P> TrialGenerator<M> for AlnsTrialGenerator<M, P> {
    /// `(destroy_index, repair_index)` of the operators that produced the
    /// trial.
    type Token = (usize, usize);

    fn generate_trial(
        &self,
        model: &M,
        current_solution: &M::SolutionType,
        _current_score: M::ScoreType,
        rng: &mut StdRng,
    ) -> (M::SolutionType, M::ScoreType, Self::Token) {
        let d_idx = self.destroy_pool.select(rng);
        let r_idx = self.repair_pool.select(rng);
        let partial = self.destroy_pool.operators[d_idx].destroy(model, current_solution, rng);
        let (solution, score) = self.repair_pool.operators[r_idx].repair(model, partial, rng);
        (solution, score, (d_idx, r_idx))
    }

    fn feedback(&mut self, token: Self::Token, outcome: TrialOutcome) {
        self.credit(token.0, token.1, outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewards_lookup_matches_constructor() {
        let r = Rewards::new(10.0, 5.0, 2.0, 0.5);
        assert_eq!(r.reward_for(TrialOutcome::NewBest), 10.0);
        assert_eq!(r.reward_for(TrialOutcome::Improved), 5.0);
        assert_eq!(r.reward_for(TrialOutcome::Accepted), 2.0);
        assert_eq!(r.reward_for(TrialOutcome::Rejected), 0.5);
    }

    #[test]
    fn select_by_weight_skewed_distribution() {
        let stats = vec![
            OperatorStats {
                weight: 0.0,
                accumulated_score: 0.0,
                usage_count: 0,
            },
            OperatorStats {
                weight: 1.0,
                accumulated_score: 0.0,
                usage_count: 0,
            },
        ];
        let mut rng = rand::rng();
        let mut picks = [0usize, 0usize];
        for _ in 0..1000 {
            let idx = select_by_weight(&stats, &mut rng);
            picks[idx] += 1;
        }
        // The first operator has weight 0 so should never be picked.
        assert_eq!(picks[0], 0);
        assert_eq!(picks[1], 1000);
    }

    #[test]
    fn apply_weight_update_blends_with_segment_average() {
        let mut stats = vec![OperatorStats {
            weight: 1.0,
            accumulated_score: 10.0,
            usage_count: 2,
        }];
        apply_weight_update(&mut stats, 0.5);
        // (1 - 0.5) * 1.0 + 0.5 * (10.0 / 2.0) = 0.5 + 2.5 = 3.0
        assert!((stats[0].weight - 3.0).abs() < 1e-12);
        // Bookkeeping is reset.
        assert_eq!(stats[0].accumulated_score, 0.0);
        assert_eq!(stats[0].usage_count, 0);
    }

    #[test]
    fn apply_weight_update_clamps_non_positive_weights() {
        let mut stats = vec![OperatorStats {
            weight: 0.5,
            accumulated_score: -100.0,
            usage_count: 1,
        }];
        apply_weight_update(&mut stats, 1.0);
        // Would be -100.0 without the floor — but the clamp keeps it positive.
        assert!(stats[0].weight > 0.0);
    }

    #[test]
    fn apply_weight_update_keeps_weight_for_unused_operators() {
        let mut stats = vec![
            OperatorStats {
                weight: 2.5,
                accumulated_score: 0.0,
                usage_count: 0,
            },
            OperatorStats {
                weight: 1.0,
                accumulated_score: 10.0,
                usage_count: 2,
            },
        ];
        apply_weight_update(&mut stats, 0.5);
        // Unused operator keeps its weight; used one blends toward its average.
        assert!((stats[0].weight - 2.5).abs() < 1e-12);
        assert!((stats[1].weight - 3.0).abs() < 1e-12);
    }

    // Minimal stub model so token-routed `feedback` can be tested deterministically.
    #[derive(Clone)]
    struct StubModel;

    impl crate::OptModel for StubModel {
        type SolutionType = ();
        type TransitionType = ();
        type ScoreType = i32;

        fn generate_random_solution<R: rand::Rng>(
            &self,
            _rng: &mut R,
        ) -> Result<(Self::SolutionType, Self::ScoreType), crate::LocalsearchError> {
            Ok(((), 0))
        }

        fn generate_trial_solution<R: rand::Rng>(
            &self,
            current_solution: Self::SolutionType,
            current_score: Self::ScoreType,
            _rng: &mut R,
        ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
            (current_solution, (), current_score)
        }
    }

    #[derive(Clone)]
    struct StubDestroy;

    impl DestroyOperator<StubModel, ()> for StubDestroy {
        fn destroy(&self, _model: &StubModel, _solution: &(), _rng: &mut StdRng) {}

        fn dyn_clone(&self) -> Box<dyn DestroyOperator<StubModel, ()>> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct StubRepair;

    impl RepairOperator<StubModel, ()> for StubRepair {
        fn repair(&self, _model: &StubModel, _partial: (), _rng: &mut StdRng) -> ((), i32) {
            ((), 0)
        }

        fn dyn_clone(&self) -> Box<dyn RepairOperator<StubModel, ()>> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn feedback_credits_token_holder() {
        let mut generator = AlnsTrialGenerator::<StubModel, ()>::new(
            vec![Box::new(StubDestroy), Box::new(StubDestroy)],
            vec![Box::new(StubRepair), Box::new(StubRepair)],
        )
        .with_segment_size(usize::MAX)
        .with_rewards(Rewards::new(1.0, 0.0, 0.0, 0.0));
        // The token — not recency — decides who is credited: tokens carry
        // the operator pair, so the loop can generate trials in parallel
        // and credit only the winner.
        generator.feedback((1, 1), TrialOutcome::NewBest);
        assert_eq!(generator.destroy_stats()[1].usage_count, 1);
        assert_eq!(generator.repair_stats()[1].usage_count, 1);
        assert_eq!(generator.destroy_stats()[1].accumulated_score, 1.0);
        assert_eq!(generator.repair_stats()[1].accumulated_score, 1.0);
    }
}
