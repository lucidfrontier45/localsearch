use std::{
    num::NonZero,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use ordered_float::NotNan;

use super::{LocalsearchError, OptModel};
use crate::optim::{
    EpsilonGreedy, LocalSearchLoop, LocalSearchOptimizer, PopulationAnnealingOptimizer, TabuList,
    TabuSearchOptimizer, gather_energy_diffs,
};

struct BatchOnlyModel {
    batch_calls: AtomicUsize,
}

impl BatchOnlyModel {
    fn new() -> Self {
        Self {
            batch_calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.batch_calls.load(Ordering::Relaxed)
    }
}

impl OptModel for BatchOnlyModel {
    type ScoreType = NotNan<f64>;
    type SolutionType = i32;
    type TransitionType = ();

    fn generate_random_solution<R: rand::Rng>(
        &self,
        _rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        Ok((0, NotNan::new(0.0).expect("zero is a valid NotNan value")))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        _current_solution: Self::SolutionType,
        _current_score: Self::ScoreType,
        _rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        panic!("single-trial generation is unsupported in this batch-only model")
    }

    fn generate_trial_solutions<R: rand::Rng + Send>(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rngs: &mut [R],
    ) -> Vec<(Self::SolutionType, Self::TransitionType, Self::ScoreType)> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        let next_score = current_score - NotNan::new(1.0).expect("one is a valid NotNan value");
        rngs.iter_mut()
            .map(|_rng| (current_solution + 1, (), next_score))
            .collect()
    }
}

#[test]
fn default_loop_uses_native_batch_once_per_iteration() {
    let model = BatchOnlyModel::new();
    let optimizer = LocalSearchLoop::new(10, 4, usize::MAX).with_seed(42);
    let mut callback = |_| {};

    let (result, _) = optimizer.step(
        &model,
        0,
        NotNan::new(0.0).expect("zero is a valid NotNan value"),
        3,
        Duration::from_secs(1),
        &mut callback,
        EpsilonGreedy::new(0.0),
    );

    assert_eq!(model.calls(), 3);
    assert_eq!(result.best_solution, 3);
    assert_eq!(
        result.best_score,
        NotNan::new(-3.0).expect("finite values are NotNan")
    );
}

#[derive(Default)]
struct NoTabu;

impl TabuList for NoTabu {
    type Item = ();

    fn set_size(&mut self, _size: usize) {}

    fn contains(&self, _transition: &Self::Item) -> bool {
        false
    }

    fn append(&mut self, _transition: Self::Item) {}
}

#[test]
fn non_alns_multi_trial_paths_use_the_batch_api() {
    let model = BatchOnlyModel::new();
    let initial_score = NotNan::new(0.0).expect("zero is a valid NotNan value");

    let _ = gather_energy_diffs(&model, Some((0, initial_score)), 3, Some(7));
    assert_eq!(model.calls(), 1);

    let model = BatchOnlyModel::new();
    let tabu = TabuSearchOptimizer::<NoTabu>::new(10, 3, usize::MAX, 4);
    let result = tabu.run(&model, Some((0, initial_score)), 1, Duration::from_secs(1));
    assert!(result.is_ok());
    assert_eq!(model.calls(), 1);

    let model = BatchOnlyModel::new();
    let population = PopulationAnnealingOptimizer::new(
        10,
        3,
        usize::MAX,
        1.0,
        0.99,
        NonZero::new(1).expect("one is non-zero"),
        4,
    );
    let result = population.run(&model, Some((0, initial_score)), 0, Duration::from_secs(1));
    assert!(result.is_ok());
    assert_eq!(model.calls(), 1);
}
