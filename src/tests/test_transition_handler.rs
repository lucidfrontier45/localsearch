use std::{
    num::NonZero,
    sync::{Arc, Mutex},
    time::Duration,
};

use ordered_float::NotNan;

use crate::optim::{
    AdaptiveAnnealingHandler, AdaptiveScheduler, EpsilonGreedyHandler, GenericLocalSearchOptimizer,
    GreatDelugeHandler, LogisticAnnealingHandler, MetropolisHandler, RelativeAnnealingHandler,
    SimulatedAnnealingHandler, TargetAccScheduleMode, TransitionHandler, TsallisHandler, UpdateCtx,
};
use crate::{LocalsearchError, OptModel};

fn score(value: f64) -> NotNan<f64> {
    NotNan::new(value).expect("finite test score")
}

fn ctx<'a>(
    iter: usize,
    total: usize,
    acc: f64,
    best: &'a NotNan<f64>,
) -> UpdateCtx<'a, NotNan<f64>> {
    UpdateCtx {
        iter,
        total,
        acc,
        best,
    }
}

#[test]
fn handler_formulas_match_existing_algorithms() {
    let current = score(1.0);
    let trial = score(1.1);

    assert_eq!(EpsilonGreedyHandler::new(0.2).evaluate(current, trial), 0.2);
    assert_abs_diff(
        MetropolisHandler::new(10.0).evaluate(current, trial),
        (-1.0_f64).exp(),
    );
    assert_abs_diff(
        RelativeAnnealingHandler::new(10.0).evaluate(current, trial),
        (-1.0_f64).exp(),
    );
    assert_abs_diff(
        LogisticAnnealingHandler::new(10.0).evaluate(current, trial),
        2.0 / (1.0 + 1.0_f64.exp()),
    );

    let tsallis = TsallisHandler::new(
        0.0,
        10.0,
        1.5,
        1.0,
        AdaptiveScheduler::default(),
        NonZero::new(2).expect("non-zero frequency"),
    );
    assert_abs_diff(tsallis.evaluate(current, trial), 1.25_f64.powf(-2.0));
    assert_eq!(
        GreatDelugeHandler::new(2.0).evaluate(current, score(1.5)),
        1.0
    );
}

#[test]
fn handlers_update_owned_state() {
    let best = score(4.0);
    let mut simulated =
        SimulatedAnnealingHandler::new(2.0, 0.5, NonZero::new(2).expect("non-zero frequency"));
    simulated.update(&ctx(1, 10, 0.0, &best));
    assert_eq!(simulated.beta, 2.0);
    simulated.update(&ctx(2, 10, 0.0, &best));
    assert_eq!(simulated.beta, 1.0);

    let scheduler = AdaptiveScheduler::new(0.5, 0.5, TargetAccScheduleMode::Constant, 0.05);
    let mut adaptive =
        AdaptiveAnnealingHandler::new(2.0, scheduler, NonZero::new(2).expect("non-zero frequency"));
    adaptive.update(&ctx(2, 10, 0.0, &best));
    assert!(adaptive.beta < 2.0);

    let mut deluge = GreatDelugeHandler::new(10.0);
    deluge.update(&ctx(5, 10, 0.0, &best));
    assert_eq!(deluge.level, 7.0);

    let mut tsallis = TsallisHandler::new(
        0.0,
        1.0,
        1.5,
        1.0,
        AdaptiveScheduler::default(),
        NonZero::new(2).expect("non-zero frequency"),
    );
    tsallis.update(&ctx(1, 10, 0.0, &best));
    assert_eq!(tsallis.offset, 4.0);
}

#[test]
fn generic_step_passes_acceptance_ratio_to_update_context() {
    #[derive(Clone, Copy)]
    struct IncreasingModel;

    impl OptModel for IncreasingModel {
        type SolutionType = i32;
        type TransitionType = ();
        type ScoreType = i32;

        fn generate_random_solution<R: rand::Rng>(
            &self,
            _rng: &mut R,
        ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
            Ok((0, 0))
        }

        fn generate_trial_solution<R: rand::Rng>(
            &self,
            current_solution: Self::SolutionType,
            current_score: Self::ScoreType,
            _rng: &mut R,
        ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
            (current_solution + 1, (), current_score - 1)
        }
    }

    #[derive(Clone)]
    struct RecordingHandler {
        updates: Arc<Mutex<Vec<f64>>>,
    }

    impl TransitionHandler<i32> for RecordingHandler {
        fn update(&mut self, context: &UpdateCtx<'_, i32>) {
            self.updates
                .lock()
                .expect("recording mutex not poisoned")
                .push(context.acc);
        }

        fn evaluate(&self, _current: i32, _trial: i32) -> f64 {
            0.0
        }
    }

    let updates = Arc::new(Mutex::new(Vec::new()));
    let handler = RecordingHandler {
        updates: Arc::clone(&updates),
    };
    let optimizer = GenericLocalSearchOptimizer::new(10, 1, usize::MAX, handler);
    let mut callback = |_progress| {};
    let result = optimizer.step(
        &IncreasingModel,
        0,
        0,
        2,
        Duration::from_secs(1),
        &mut callback,
    );

    assert_eq!(result.acceptance_counter.acceptance_ratio(), 1.0);
    assert_eq!(
        *updates.lock().expect("recording mutex not poisoned"),
        vec![0.0, 1.0]
    );
}

fn assert_abs_diff(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}
