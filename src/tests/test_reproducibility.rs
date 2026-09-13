//! Reproducibility tests for issue #100.
//!
//! Verify that bit-identical `(solution, score)` pairs are produced across
//! two calls of the same optimizer when `with_seed` is used, across the
//! full set of single-chain + multi-chain optimizers on a deterministic
//! `QuadraticModel`. Also verify thread-count independence via
//! `ThreadPool::install`, and reproducibility of the tuning helpers.

use std::num::NonZero;
use std::time::Duration;

use ordered_float::NotNan;
use rayon::ThreadPoolBuilder;

use super::{QuadraticModel, TransitionType};
use crate::{
    optim::{
        AdaptiveAnnealingOptimizer, EpsilonGreedyOptimizer, LocalSearchOptimizer,
        ParallelTemperingOptimizer, PopulationAnnealingOptimizer, SimulatedAnnealingOptimizer,
        TabuList, TabuSearchOptimizer, tune_temperature,
    },
    utils::RingBuffer,
};

const N_ITER: usize = 200;

fn run_pair<R: LocalSearchOptimizer<QuadraticModel>>(opt_a: R, opt_b: R) {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
    let (sa, ia) = opt_a
        .run(&model, None, N_ITER, Duration::from_secs(30))
        .expect("run a");
    let (sb, ib) = opt_b
        .run(&model, None, N_ITER, Duration::from_secs(30))
        .expect("run b");
    assert_eq!(ia, ib, "score differs across same-seed runs");
    assert_eq!(sa, sb, "solution differs across same-seed runs");
}

#[test]
fn sa_same_seed_is_bit_identical() {
    let make = || {
        SimulatedAnnealingOptimizer::new(
            100,
            10,
            10,
            1.0,
            0.99,
            NonZero::new(1).expect("update_frequency >= 1"),
        )
        .with_seed(42)
    };
    run_pair(make(), make());
}

#[test]
fn epsilon_greedy_same_seed_is_bit_identical() {
    let make = || EpsilonGreedyOptimizer::new(100, 10, 10, 0.1).with_seed(7);
    run_pair(make(), make());
}

#[test]
fn adaptive_annealing_same_seed_is_bit_identical() {
    let make = || {
        AdaptiveAnnealingOptimizer::new(
            100,
            10,
            10,
            1.0,
            crate::optim::AdaptiveScheduler::default(),
            NonZero::new(1).expect("update_frequency >= 1"),
        )
        .with_seed(123)
    };
    run_pair(make(), make());
}

#[test]
fn tabu_same_seed_is_bit_identical() {
    // Simple identity tabu list: never considers anything tabu so the
    // optimizer behaves like a stochastic best-improvement search.
    #[derive(Debug, Default)]
    struct NoTabu;
    impl TabuList for NoTabu {
        type Item = TransitionType;
        fn set_size(&mut self, _: usize) {}
        fn contains(&self, _: &Self::Item) -> bool {
            false
        }
        fn append(&mut self, _: Self::Item) {}
    }
    let make = || TabuSearchOptimizer::<NoTabu>::new(100, 10, 10, 4).with_seed(99);
    run_pair(make(), make());
}

#[test]
fn parallel_tempering_same_seed_is_bit_identical() {
    let make = || {
        ParallelTemperingOptimizer::with_geometric_betas(
            100,
            10,
            10,
            4,
            0.1,
            5.0,
            NonZero::new(1).expect("update_frequency >= 1"),
        )
        .with_seed(0xDEAD_BEEFu64)
    };
    run_pair(make(), make());
}

#[test]
fn population_annealing_same_seed_is_bit_identical() {
    let make = || {
        PopulationAnnealingOptimizer::new(
            100,
            10,
            10,
            1.0,
            0.99,
            NonZero::new(1).expect("update_frequency >= 1"),
            4,
        )
        .with_seed(0xCAFE_F00Du64)
    };
    run_pair(make(), make());
}

#[test]
fn tune_temperature_seeded_is_bit_identical() {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
    let beta_a = tune_temperature(&model, None, 200, 0.8, Some(2024));
    let beta_b = tune_temperature(&model, None, 200, 0.8, Some(2024));
    assert_eq!(
        beta_a, beta_b,
        "tune_temperature must be deterministic for the same seed"
    );
    assert!(beta_a.is_finite() && beta_a > 0.0, "beta must be positive");
}

#[test]
fn thread_count_independent_for_epsilon_greedy() {
    // Same seed, different rayon thread counts → same result. Sequential
    // forks of the master RNG decouple the trial streams from worker
    // count.
    let _ = RingBuffer::<usize>::new(4); // keep import warm; remove if unused
    let model_factory = || QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));

    let run_under_pool = |threads: usize| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool");
        pool.install(|| {
            let model = model_factory();
            let opt = EpsilonGreedyOptimizer::new(100, 10, 10, 0.1).with_seed(7);
            opt.run(&model, None, N_ITER, Duration::from_secs(30))
                .expect("run")
        })
    };

    let (s1, i1) = run_under_pool(1);
    let (s4, i4) = run_under_pool(4);
    assert_eq!(
        i1, i4,
        "score must match across thread counts (1 vs 4): {} vs {}",
        i1, i4
    );
    assert_eq!(
        s1, s4,
        "best solution must match across thread counts (1 vs 4)"
    );
}

// Reference to NotNan so the helper signature compiles even on all targets
// where the test module isn't currently active.
#[allow(dead_code)]
fn _notnan_anchor(x: f64) -> NotNan<f64> {
    NotNan::new(x).unwrap()
}
