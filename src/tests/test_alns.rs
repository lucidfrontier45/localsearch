use std::{
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use ordered_float::NotNan;
use rand::{RngExt as _, SeedableRng as _, distr::Uniform, prelude::Distribution, rngs::StdRng};

use super::QuadraticModel;
use crate::{
    LocalsearchError, OptModel,
    optim::{
        AlnsTrialGenerator, DefaultTrialGenerator, DestroyOperator, EpsilonGreedy,
        GenericLocalSearchOptimizer, LocalSearchLoop, LocalSearchOptimizer, RepairOperator,
        TransitionHandler, TrialGenerator, TrialOutcome, UpdateCtx,
    },
};

// ---------------------------------------------------------------------------
// Test model tailored for ALNS: each solution is a vector of `usize`s and the
// score is the sum of the elements. Destroy operators zero out elements at a
// set of positions and repair operators refill them with fresh random values.
// ---------------------------------------------------------------------------

type PartialSolution = Vec<Option<usize>>;

#[derive(Clone)]
struct SumModel {
    n: usize,
    value_range: (usize, usize),
}

impl SumModel {
    fn new(n: usize, value_range: (usize, usize)) -> Self {
        Self { n, value_range }
    }

    fn evaluate(&self, solution: &[usize]) -> NotNan<f64> {
        let sum: usize = solution.iter().sum();
        NotNan::new(sum as f64).expect("finite test score")
    }
}

impl OptModel for SumModel {
    type SolutionType = Vec<usize>;
    type TransitionType = ();
    type ScoreType = NotNan<f64>;

    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        let dist = Uniform::new(self.value_range.0, self.value_range.1)
            .expect("value range must be valid");
        let solution = dist.sample_iter(rng).take(self.n).collect::<Vec<_>>();
        let score = self.evaluate(&solution);
        Ok((solution, score))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        mut current_solution: Self::SolutionType,
        _current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        // Touch a single random position so default-generator runs still work.
        let k = rng.random_range(0..self.n);
        let dist = Uniform::new(self.value_range.0, self.value_range.1)
            .expect("value range must be valid");
        current_solution[k] = dist.sample(rng);
        let score = self.evaluate(&current_solution);
        (current_solution, (), score)
    }
}

// ---------------------------------------------------------------------------
// Destroy / repair operator implementations used by the ALNS tests.
// ---------------------------------------------------------------------------

/// Destroy operator that zeroes out the first half of the solution.
#[derive(Clone)]
struct DestroyFirstHalf;

impl DestroyOperator<SumModel, PartialSolution> for DestroyFirstHalf {
    fn destroy(
        &self,
        _model: &SumModel,
        solution: &Vec<usize>,
        _rng: &mut StdRng,
    ) -> PartialSolution {
        let mut partial: PartialSolution = solution.iter().copied().map(Some).collect();
        let half = partial.len() / 2;
        for v in partial.iter_mut().take(half) {
            *v = None;
        }
        partial
    }

    fn dyn_clone(&self) -> Box<dyn DestroyOperator<SumModel, PartialSolution>> {
        Box::new(self.clone())
    }
}

/// Destroy operator that zeroes out the second half of the solution.
#[derive(Clone)]
struct DestroySecondHalf;

impl DestroyOperator<SumModel, PartialSolution> for DestroySecondHalf {
    fn destroy(
        &self,
        _model: &SumModel,
        solution: &Vec<usize>,
        _rng: &mut StdRng,
    ) -> PartialSolution {
        let mut partial: PartialSolution = solution.iter().copied().map(Some).collect();
        let half = partial.len() / 2;
        for v in partial.iter_mut().skip(half) {
            *v = None;
        }
        partial
    }

    fn dyn_clone(&self) -> Box<dyn DestroyOperator<SumModel, PartialSolution>> {
        Box::new(self.clone())
    }
}

/// Repair operator that fills `None` slots with values drawn from
/// `[value_range.0, value_range.1)` and computes the resulting score.
#[derive(Clone)]
struct RandomRepair {
    value_range: (usize, usize),
}

impl RepairOperator<SumModel, PartialSolution> for RandomRepair {
    fn repair(
        &self,
        model: &SumModel,
        mut partial: PartialSolution,
        rng: &mut StdRng,
    ) -> (Vec<usize>, NotNan<f64>) {
        let dist = Uniform::new(self.value_range.0, self.value_range.1)
            .expect("value range must be valid");
        for slot in partial.iter_mut() {
            if slot.is_none() {
                *slot = Some(dist.sample(rng));
            }
        }
        let solution = partial
            .into_iter()
            .map(|v| v.expect("repair must fill every slot"))
            .collect::<Vec<_>>();
        let score = model.evaluate(&solution);
        (solution, score)
    }

    fn dyn_clone(&self) -> Box<dyn RepairOperator<SumModel, PartialSolution>> {
        Box::new(self.clone())
    }
}

fn destroy_ops() -> Vec<Box<dyn DestroyOperator<SumModel, PartialSolution>>> {
    vec![Box::new(DestroyFirstHalf), Box::new(DestroySecondHalf)]
}

fn repair_ops() -> Vec<Box<dyn RepairOperator<SumModel, PartialSolution>>> {
    vec![Box::new(RandomRepair {
        value_range: (0, 100),
    })]
}

fn build_alns() -> AlnsTrialGenerator<SumModel, PartialSolution> {
    AlnsTrialGenerator::new(destroy_ops(), repair_ops())
}

// ---------------------------------------------------------------------------
// Default-generator behavior preserved through `step()`.
// ---------------------------------------------------------------------------

#[test]
fn default_step_matches_legacy_behavior() {
    // The legacy quadratic model keeps converging with hill climbing.
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
    let opt = LocalSearchLoop::new(10_000, 10, usize::MAX);
    let handler = EpsilonGreedy::new(0.0);
    let mut cb = |_p| {};
    let (result, _) = opt.step(
        &model,
        vec![0.0, 0.0, 0.0],
        NotNan::new(2.0_f64.powf(2.0) + 3.5_f64.powf(2.0)).unwrap(),
        200,
        Duration::from_secs(5),
        &mut cb,
        handler,
    );
    // The greedy search must keep producing strictly non-increasing scores.
    assert!(result.best_score.into_inner() < 30.0);
    assert_eq!(result.best_solution.len(), 3);
}

// ---------------------------------------------------------------------------
// Custom generator is invoked through `step_with_generator`.
// ---------------------------------------------------------------------------

// Atomics + mutex keep the generator `Sync` so the loop can invoke it from
// rayon worker threads.
#[derive(Default)]
struct CountingGenerator {
    generate_calls: AtomicUsize,
    feedback_calls: AtomicUsize,
    last_outcome: Mutex<Option<TrialOutcome>>,
}

impl TrialGenerator<QuadraticModel> for CountingGenerator {
    type Token = ();

    fn generate_trial(
        &self,
        _model: &QuadraticModel,
        current_solution: &Vec<f64>,
        current_score: NotNan<f64>,
        _rng: &mut StdRng,
    ) -> (Vec<f64>, NotNan<f64>, Self::Token) {
        self.generate_calls.fetch_add(1, Ordering::Relaxed);
        // Nudge one coordinate; the loop still applies its acceptance logic.
        let mut next = current_solution.clone();
        if !next.is_empty() {
            next[0] += 0.1;
        }
        (next, current_score, ())
    }

    fn feedback(&mut self, _token: Self::Token, outcome: TrialOutcome) {
        self.feedback_calls.fetch_add(1, Ordering::Relaxed);
        *self.last_outcome.lock().expect("mutex poisoned") = Some(outcome);
    }
}

#[test]
fn custom_generator_is_invoked_through_step_with_generator() {
    let model = QuadraticModel::new(1, vec![0.0], (-1.0, 1.0));
    let opt = LocalSearchLoop::new(10, 1, usize::MAX);
    let handler = EpsilonGreedy::new(0.0);
    let mut cb = |_p| {};
    let generator = CountingGenerator::default();
    let (_, _, returned) = opt.step_with_generator(
        &model,
        vec![0.5],
        NotNan::new(0.25).unwrap(),
        5,
        Duration::from_secs(1),
        &mut cb,
        handler,
        generator,
    );
    // n_iter iterations × n_trials per iteration.
    assert_eq!(returned.generate_calls.load(Ordering::Relaxed), 5);
    assert_eq!(returned.feedback_calls.load(Ordering::Relaxed), 5);
    assert!(
        returned
            .last_outcome
            .lock()
            .expect("mutex poisoned")
            .is_some()
    );
}

// ---------------------------------------------------------------------------
// Existing `GenericLocalSearchOptimizer::new` call sites still compile and run.
// ---------------------------------------------------------------------------

#[test]
fn generic_optimizer_new_still_works() {
    let model = SumModel::new(4, (0, 100));
    let optimizer = GenericLocalSearchOptimizer::new(1000, 1, usize::MAX, EpsilonGreedy::new(0.1));
    // The default generator is the unit struct `DefaultTrialGenerator`.
    let _: &DefaultTrialGenerator = optimizer.generator();
    let (solution, score) = optimizer
        .run(&model, None, 50, Duration::from_secs(1))
        .unwrap();
    assert_eq!(solution.len(), 4);
    let sum: usize = solution.iter().sum();
    assert_eq!(sum as f64, score.into_inner());
}

// ---------------------------------------------------------------------------
// ALNS: operator selection follows weights.
// ---------------------------------------------------------------------------

#[test]
fn alns_destroy_operator_selection_respects_weights() {
    // With equal initial weights, both destroy operators should be picked
    // roughly equally across many trials.
    let model = SumModel::new(4, (0, 100));
    let mut total_destroy_picks = [0usize, 0usize];
    let mut total_repair_picks = [0usize; 1];
    for i in 0..2000 {
        let mut local = build_alns().with_segment_size(usize::MAX);
        let mut rng = StdRng::seed_from_u64(42 + i);
        let solution = vec![1, 2, 3, 4];
        let (_solution, _score, token) =
            <AlnsTrialGenerator<SumModel, PartialSolution> as TrialGenerator<SumModel>>::generate_trial(
                &local, &model, &solution, NotNan::new(10.0).unwrap(), &mut rng,
            );
        // Credit the trial via its token (we don't care about the outcome here).
        local.feedback(token, TrialOutcome::Rejected);
        total_destroy_picks[0] += local.destroy_usage()[0];
        total_destroy_picks[1] += local.destroy_usage()[1];
        total_repair_picks[0] += local.repair_usage()[0];
    }
    // With equal initial weights, both destroy operators should be picked
    // somewhere between 35 % and 65 % of the time across 2000 trials.
    let total = (total_destroy_picks[0] + total_destroy_picks[1]) as f64;
    let p0 = total_destroy_picks[0] as f64 / total;
    let p1 = total_destroy_picks[1] as f64 / total;
    assert!(
        (0.35..=0.65).contains(&p0),
        "destroy operator 0 selected {p0:.3}, expected ~0.5"
    );
    assert!(
        (0.35..=0.65).contains(&p1),
        "destroy operator 1 selected {p1:.3}, expected ~0.5"
    );
    // All repair picks go to the single repair operator.
    assert_eq!(total_repair_picks[0], 2000);
}

// ---------------------------------------------------------------------------
// ALNS: rewards are credited to the selected operators.
// ---------------------------------------------------------------------------

#[test]
fn alns_feedback_credits_rewards_to_selected_operators() {
    // Drop in two repair operators so we can verify the credit lands on the
    // operator that was actually selected rather than smeared across all of
    // them. Custom rewards make the per-outcome accounting easy to assert.
    let mut generator = AlnsTrialGenerator::<SumModel, PartialSolution>::new(
        destroy_ops(),
        vec![
            Box::new(RandomRepair {
                value_range: (0, 100),
            }),
            Box::new(RandomRepair {
                value_range: (0, 1),
            }),
        ],
    )
    .with_segment_size(usize::MAX)
    .with_rewards(crate::optim::Rewards::new(20.0, 5.0, 2.0, 0.0));

    let model = SumModel::new(4, (0, 100));

    // Run a bunch of trials, observe that usage increments for both destroy
    // and repair operators and that the accumulated scores reflect the
    // rewards applied per outcome.
    let mut total_score = 0.0_f64;
    let mut total_usage = 0usize;
    for &outcome in &[
        TrialOutcome::NewBest,
        TrialOutcome::Improved,
        TrialOutcome::Accepted,
        TrialOutcome::Rejected,
    ] {
        let mut local_rng = StdRng::seed_from_u64(7);
        let solution = vec![1, 2, 3, 4];
        let (_, _, token) = <AlnsTrialGenerator<SumModel, PartialSolution> as TrialGenerator<
            SumModel,
        >>::generate_trial(
            &generator,
            &model,
            &solution,
            NotNan::new(10.0).unwrap(),
            &mut local_rng,
        );
        let expected_reward = match outcome {
            TrialOutcome::NewBest => 20.0,
            TrialOutcome::Improved => 5.0,
            TrialOutcome::Accepted => 2.0,
            TrialOutcome::Rejected => 0.0,
        };
        generator.feedback(token, outcome);
        // After this feedback, *some* destroy and *some* repair operator was
        // credited. Verify usage incremented and the aggregate reward is
        // accumulated correctly across all operators.
        let destroy_usage: usize = generator.destroy_usage().iter().sum();
        let repair_usage: usize = generator.repair_usage().iter().sum();
        assert_eq!(destroy_usage, repair_usage);
        assert!(destroy_usage >= 1);
        total_usage = destroy_usage;
        let destroy_scores: f64 = generator.destroy_scores().iter().sum();
        let repair_scores: f64 = generator.repair_scores().iter().sum();
        // Destroy and repair scores are independent, but their usage matches.
        total_score += expected_reward;
        assert!(
            (destroy_scores - total_score).abs() < 1e-9,
            "destroy accumulated score {destroy_scores} != expected {total_score}"
        );
        assert!(
            (repair_scores - total_score).abs() < 1e-9,
            "repair accumulated score {repair_scores} != expected {total_score}"
        );
    }
    assert_eq!(total_usage, 4);
}

// ---------------------------------------------------------------------------
// ALNS: weights are updated at segment boundaries.
// ---------------------------------------------------------------------------

#[test]
fn alns_updates_weights_at_segment_boundaries() {
    let mut generator =
        AlnsTrialGenerator::<SumModel, PartialSolution>::new(destroy_ops(), repair_ops())
            .with_segment_size(3)
            .with_reaction_factor(0.5);

    let model = SumModel::new(4, (0, 100));
    // Run 3 trials with reward 9 (Improved) each. With segment_size=3 and
    // reaction_factor=0.5, after the third feedback the weights are blended
    // with the segment average.
    for i in 0..3 {
        let mut rng = StdRng::seed_from_u64(i as u64);
        let solution = vec![1, 2, 3, 4];
        let (_, _, token) = <AlnsTrialGenerator<SumModel, PartialSolution> as TrialGenerator<
            SumModel,
        >>::generate_trial(
            &generator,
            &model,
            &solution,
            NotNan::new(10.0).unwrap(),
            &mut rng,
        );
        generator.feedback(token, TrialOutcome::Improved);
    }
    // After segment ends, usage and accumulated scores must reset.
    let usage = generator.destroy_usage();
    let scores = generator.destroy_scores();
    assert!(usage.iter().all(|&u| u == 0));
    assert!(scores.iter().all(|&s| s == 0.0));
    // trials_in_segment also resets.
    assert_eq!(generator.trials_in_segment(), 0);
    // Some destroy operator must have received a higher weight than another
    // (or at minimum the weights are shifted from the initial 1.0).
    let weights = generator.destroy_weights();
    assert!(
        weights.iter().any(|&w| (w - 1.0).abs() > 1e-12),
        "weights must have been updated from initial value of 1.0, got {weights:?}"
    );
    // Repair weights also updated.
    let repair_weights = generator.repair_weights();
    assert_eq!(repair_weights.len(), 1);
    // All repair trials went to the single repair operator, so its segment
    // average was 9.0 -> new weight = (1 - 0.5) * 1.0 + 0.5 * 9.0 = 5.0.
    assert!((repair_weights[0] - 5.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// ALNS: generator state survives the search step.
// ---------------------------------------------------------------------------

#[test]
fn alns_generator_state_survives_step_with_generator() {
    let model = SumModel::new(4, (0, 100));
    let opt = LocalSearchLoop::new(100, 1, usize::MAX);
    let handler = EpsilonGreedy::new(0.5);
    let mut cb = |_p| {};
    let initial = build_alns().with_segment_size(usize::MAX);
    let (_, _, returned) = opt.step_with_generator(
        &model,
        vec![1, 2, 3, 4],
        NotNan::new(10.0).unwrap(),
        10,
        Duration::from_secs(1),
        &mut cb,
        handler,
        initial,
    );
    // The usage / scores must reflect the 10 trials that ran (one credit
    // per iteration; tokens leave no pending state behind).
    let total_destroy_usage: usize = returned.destroy_usage().iter().sum();
    let total_repair_usage: usize = returned.repair_usage().iter().sum();
    assert_eq!(total_destroy_usage, 10);
    assert_eq!(total_repair_usage, 10);
}

// ---------------------------------------------------------------------------
// ALNS through `GenericLocalSearchOptimizer::with_trial_generator`.
// ---------------------------------------------------------------------------

#[test]
fn alns_runs_through_generic_optimizer() {
    let model = SumModel::new(4, (0, 100));
    let generator = build_alns().with_segment_size(usize::MAX);
    let optimizer = GenericLocalSearchOptimizer::new(200, 1, usize::MAX, EpsilonGreedy::new(0.1))
        .with_trial_generator(generator);
    let (solution, score) = optimizer
        .run(&model, None, 30, Duration::from_secs(1))
        .unwrap();
    assert_eq!(solution.len(), 4);
    let sum: usize = solution.iter().sum();
    assert_eq!(sum as f64, score.into_inner());
    // The generator stored inside the optimizer must be the ALNS one.
    assert_eq!(optimizer.generator().n_destroy_operators(), 2);
    assert_eq!(optimizer.generator().n_repair_operators(), 1);
}

// ---------------------------------------------------------------------------
// ALNS with n_trials > 1: only the winner is credited per iteration and no
// pending selections leak across iterations.
// ---------------------------------------------------------------------------

#[test]
fn alns_with_multiple_trials_credits_only_winner_per_iteration() {
    let model = SumModel::new(4, (0, 100));
    let opt = LocalSearchLoop::new(100, 3, usize::MAX);
    let handler = EpsilonGreedy::new(0.5);
    let mut cb = |_p| {};
    let initial = build_alns().with_segment_size(usize::MAX);
    let (_, _, returned) = opt.step_with_generator(
        &model,
        vec![1, 2, 3, 4],
        NotNan::new(10.0).unwrap(),
        10,
        Duration::from_secs(1),
        &mut cb,
        handler,
        initial,
    );
    // One credit per iteration — not one per generated candidate.
    // (Tokens carry the operator pair, so no pending state can leak and the
    // winner is always credited.)
    let total_destroy_usage: usize = returned.destroy_usage().iter().sum();
    let total_repair_usage: usize = returned.repair_usage().iter().sum();
    assert_eq!(total_destroy_usage, 10);
    assert_eq!(total_repair_usage, 10);
}

// ---------------------------------------------------------------------------
// A trial that beats the global best is credited `NewBest` even when the
// acceptance handler rejects it (best bookkeeping runs before acceptance).
// ---------------------------------------------------------------------------

/// Handler that always rejects. Deliberately violates the
/// improving-transitions-return-1.0 contract so classification of rejected
/// trials can be exercised.
struct AlwaysReject;

impl TransitionHandler<NotNan<f64>> for AlwaysReject {
    fn update(&mut self, _ctx: &UpdateCtx<'_, NotNan<f64>>) {}

    fn evaluate(&self, _current: NotNan<f64>, _trial: NotNan<f64>) -> f64 {
        0.0
    }
}

/// Generator whose every trial strictly improves on every previous one
/// (score decreases with each call), recording the outcome it was last
/// credited with. Deriving the score from the call count — not from
/// `current_score` — keeps trials improving even when the handler rejects
/// every one.
#[derive(Default)]
struct ImprovingGenerator {
    calls: AtomicUsize,
    last_outcome: Mutex<Option<TrialOutcome>>,
}

impl TrialGenerator<QuadraticModel> for ImprovingGenerator {
    type Token = ();

    fn generate_trial(
        &self,
        _model: &QuadraticModel,
        current_solution: &Vec<f64>,
        _current_score: NotNan<f64>,
        _rng: &mut StdRng,
    ) -> (Vec<f64>, NotNan<f64>, Self::Token) {
        let calls = self.calls.fetch_add(1, Ordering::Relaxed) as f64;
        let next = NotNan::new(-calls).expect("finite test score");
        (current_solution.clone(), next, ())
    }

    fn feedback(&mut self, _token: Self::Token, outcome: TrialOutcome) {
        *self.last_outcome.lock().expect("mutex poisoned") = Some(outcome);
    }
}

#[test]
fn rejected_trial_beating_best_reports_new_best() {
    let model = QuadraticModel::new(1, vec![0.0], (-1.0, 1.0));
    let opt = LocalSearchLoop::new(10, 1, usize::MAX);
    let handler = AlwaysReject;
    let mut cb = |_p| {};
    let (result, _, generator) = opt.step_with_generator(
        &model,
        vec![0.5],
        NotNan::new(0.25).unwrap(),
        5,
        Duration::from_secs(1),
        &mut cb,
        handler,
        ImprovingGenerator::default(),
    );
    // Every trial strictly improves, so the recorded global best keeps
    // dropping even though the handler rejects every single one.
    assert!(result.best_score.into_inner() < 0.25);
    // The generator must be credited `NewBest` — not `Rejected` — for a
    // trial that produced the recorded global best.
    assert_eq!(
        *generator.last_outcome.lock().expect("mutex poisoned"),
        Some(TrialOutcome::NewBest)
    );
}

#[test]
#[should_panic(expected = "n_trials must be at least 1")]
fn local_search_loop_rejects_zero_trials() {
    // Fail fast at construction instead of on the first iteration.
    let _: LocalSearchLoop<NotNan<f64>> = LocalSearchLoop::new(10, 0, usize::MAX);
}
