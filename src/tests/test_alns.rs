use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

use std::time::Duration;

use ordered_float::NotNan;
use rand::RngExt as _;

use super::QuadraticModel;
use crate::{
    optim::{AlnsOperatorModel, AlnsOptimizer, GenericLocalSearchOptimizer, StepResult},
    AcceptanceCounter, LocalsearchError, OptModel, OptProgress,
};

#[test]
fn step_result_default_output_is_unit_for_backward_compat() {
    let model = QuadraticModel::new(2, vec![1.0, -1.0], (-5.0, 5.0));
    let opt = GenericLocalSearchOptimizer::new(100, 4, 10, |_: NotNan<f64>, _: NotNan<f64>| 0.0);
    // Safety: test-only RNG access; `QuadraticModel` always returns `Ok` here.
    let (initial, score) = model.generate_random_solution(&mut rand::rng()).unwrap();
    let mut callback = |_: OptProgress<Vec<f64>, NotNan<f64>>| {};
    // Old two-parameter spelling must keep compiling via the `O = ()` default.
    let result: StepResult<Vec<f64>, NotNan<f64>> = opt.step(
        &model,
        initial,
        score,
        20,
        Duration::from_secs(1),
        &mut callback,
    );
    assert_eq!(result.output, ());
}

#[test]
fn step_result_carries_custom_output() {
    // A custom `O: Default` flows through `step()` as `O::default()`.
    let model = QuadraticModel::new(2, vec![1.0, -1.0], (-5.0, 5.0));
    let opt = GenericLocalSearchOptimizer::new(100, 4, 10, |_: NotNan<f64>, _: NotNan<f64>| 0.0);
    // Safety: test-only RNG access; `QuadraticModel` always returns `Ok` here.
    let (initial, score) = model.generate_random_solution(&mut rand::rng()).unwrap();
    let mut callback = |_: OptProgress<Vec<f64>, NotNan<f64>>| {};
    let result: StepResult<Vec<f64>, NotNan<f64>, crate::optim::AlnsStatistics> = opt.step(
        &model,
        initial,
        score,
        20,
        Duration::from_secs(1),
        &mut callback,
    );
    assert_eq!(result.output.operator_uses.len(), 0);
    assert_eq!(result.output.operator_scores.len(), 0);

    // An arbitrary payload can be attached by direct construction.
    let carried = StepResult {
        best_solution: vec![1.0],
        // Safety: literal is finite.
        best_score: NotNan::new(0.0).unwrap(),
        last_solution: vec![1.0],
        // Safety: literal is finite.
        last_score: NotNan::new(0.0).unwrap(),
        acceptance_counter: AcceptanceCounter::new(10),
        output: vec![7usize, 42],
    };
    assert_eq!(carried.output, vec![7usize, 42]);
}

/// Two-operator stub: operator 0 glides toward the target (reliably improving),
/// operator 1 jumps uniformly at random (usually worsening).
struct TwoOperatorModel {
    target: f64,
    weights: Mutex<Vec<f64>>,
    uses: [AtomicUsize; 2],
    improvements: [AtomicUsize; 2],
}

impl TwoOperatorModel {
    fn new(target: f64) -> Self {
        Self {
            target,
            weights: Mutex::new(vec![1.0, 1.0]),
            uses: [AtomicUsize::new(0), AtomicUsize::new(0)],
            improvements: [AtomicUsize::new(0), AtomicUsize::new(0)],
        }
    }

    fn score(&self, x: f64) -> NotNan<f64> {
        // Safety: inputs stay finite by construction, so `abs()` is never NaN.
        NotNan::new((x - self.target).abs()).unwrap()
    }

    fn select_operator(&self, rng: &mut impl rand::Rng) -> usize {
        // Safety: lock is never poisoned; we never panic while holding it.
        let weights = self.weights.lock().unwrap();
        let total: f64 = weights.iter().sum();
        let mut x = rng.random_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            x -= *w;
            if x <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

impl OptModel for TwoOperatorModel {
    type SolutionType = f64;
    // Move-level descriptor only: which operator produced the move.
    // ALNS statistics travel via `StepResult::output`, never here.
    type TransitionType = usize;
    type ScoreType = NotNan<f64>;

    fn generate_random_solution<R: rand::Rng>(
        &self,
        _rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        Ok((0.0, self.score(0.0)))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        let operator = self.select_operator(rng);
        self.uses[operator].fetch_add(1, Ordering::Relaxed);
        let trial = match operator {
            0 => current_solution + 0.3 * (self.target - current_solution),
            _ => rng.random_range(-10.0..10.0),
        };
        let score = self.score(trial);
        if score < current_score {
            self.improvements[operator].fetch_add(1, Ordering::Relaxed);
        }
        (trial, operator, score)
    }
}

impl AlnsOperatorModel for TwoOperatorModel {
    fn n_operators(&self) -> usize {
        2
    }

    fn set_operator_weights(&self, weights: &[f64]) {
        // Safety: lock is never poisoned; we never panic while holding it.
        *self.weights.lock().unwrap() = weights.to_vec();
    }

    fn drain_operator_stats(&self) -> (Vec<usize>, Vec<usize>) {
        let stats = (0..2)
            .map(|i| {
                (
                    self.uses[i].swap(0, Ordering::Relaxed),
                    self.improvements[i].swap(0, Ordering::Relaxed),
                )
            })
            .collect::<Vec<_>>();
        (
            stats.iter().map(|(u, _)| *u).collect(),
            stats.iter().map(|(_, v)| *v).collect(),
        )
    }
}

#[test]
fn alns_segment_runner_adapts_weights_toward_winning_operator() {
    let model = TwoOperatorModel::new(5.0);
    // Pure hill-climbing acceptance: improving trials always accepted, worsening never.
    let mut opt = AlnsOptimizer::new(
        10_000,
        4,
        1000,
        |_: NotNan<f64>, _: NotNan<f64>| 0.0,
        2,
        0.1,
    );
    let mut callback = |_: OptProgress<f64, NotNan<f64>>| {};

    let (mut current, mut current_score) = (0.0, model.score(0.0));
    let mut best = current_score;
    for _ in 0..3 {
        let result = opt.run_segment(
            &model,
            current,
            current_score,
            best,
            50,
            Duration::from_secs(10),
            &mut callback,
        );
        // Best score never worsens across segments.
        assert!(result.best_score <= best);
        // Per-operator feedback arrives via `output`, keeping `TransitionType`
        // (tabu move semantics) untouched.
        assert_eq!(result.output.operator_uses.len(), 2);
        assert_eq!(result.output.operator_scores.len(), 2);
        assert!(result.output.operator_uses.iter().sum::<usize>() > 0);
        best = result.best_score;
        current = result.best_solution;
        current_score = result.best_score;
    }

    assert!(best < model.score(0.0));
    assert!(opt.weights()[0] > 1.0, "weights: {:?}", opt.weights());
    assert!(
        opt.weights()[0] > opt.weights()[1],
        "weights: {:?}",
        opt.weights()
    );
}
