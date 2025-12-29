![LocalSearch Logo](logo_wide.png)

[![Crates.io](https://img.shields.io/crates/v/localsearch.svg)](https://crates.io/crates/localsearch)
[![Documentation](https://docs.rs/localsearch/badge.svg)](https://docs.rs/localsearch)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.92%2B-orange.svg)](https://www.rust-lang.org)

A high-performance Rust library for local search optimization algorithms (metaheuristics). Built with parallel execution using Rayon and a flexible trait-based design for implementing custom optimization problems.

## Features

This library implements 13+ local search optimization algorithms, all parallelized with Rayon:

### Local Search Algorithms
- **Random Search** - Baseline random sampling method
- **Hill Climbing** - Deterministic greedy ascent
- **Epsilon-Greedy** - Probabilistic acceptance of worse solutions
- **Metropolis** - MCMC method with fixed temperature
- **Tabu Search** - Memory-based search with forbidden move lists
- **Great Deluge** - Threshold-based acceptance with decreasing water levels

### Simulated Annealing Variants
- **Simulated Annealing** - Classic temperature-based acceptance with cooling schedules
- **Adaptive Annealing** - Dynamic temperature adaptation to target acceptance rates
- **Logistic Annealing** - Relative score differences with logistic acceptance curves
- **Relative Annealing** - Exponential acceptance based on relative score changes
- **Tsallis Relative Annealing** - Generalized acceptance using Tsallis statistics

### Population-Based Methods
- **Population Annealing** - Parallel simulated annealing with population resampling
- **Parallel Tempering** - Multiple chains at different temperatures with replica exchange


## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
localsearch = "0.23.0"
```

Requires Rust 1.92 or later.

## Quick Start

Implement the `OptModel` trait for your problem and choose an optimizer. Here's a quadratic function minimization example:

```rust
use std::time::Duration;

use localsearch::{
    LocalsearchError, OptModel,
    optim::HillClimbingOptimizer,
};
use ordered_float::NotNan;
use rand::distr::Uniform;

type SolutionType = Vec<f64>;
type ScoreType = NotNan<f64>;

#[derive(Clone)]
struct QuadraticModel {
    k: usize,
    centers: Vec<f64>,
    dist: Uniform<f64>,
}

impl QuadraticModel {
    fn new(k: usize, centers: Vec<f64>, value_range: (f64, f64)) -> Self {
        let (low, high) = value_range;
        let dist = Uniform::new(low, high).unwrap();
        Self { k, centers, dist }
    }

    fn evaluate_solution(&self, solution: &SolutionType) -> NotNan<f64> {
        let score = (0..self.k)
            .map(|i| (solution[i] - self.centers[i]).powf(2.0))
            .sum();
        NotNan::new(score).unwrap()
    }
}

impl OptModel for QuadraticModel {
    type SolutionType = SolutionType;
    type TransitionType = ();
    type ScoreType = ScoreType;

    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        let solution = self.dist.sample_iter(rng).take(self.k).collect::<Vec<_>>();
        let score = self.evaluate_solution(&solution);
        Ok((solution, score))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: Self::SolutionType,
        _current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, NotNan<f64>) {
        let k = rng.random_range(0..self.k);
        let v = self.dist.sample(rng);
        let mut new_solution = current_solution;
        new_solution[k] = v;
        let score = self.evaluate_solution(&new_solution);
        (new_solution, (), score)
    }
}

// Usage
let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
let opt = HillClimbingOptimizer::new(1000, 50);
let (solution, score) = opt
    .run(&model, None, 10000, Duration::from_secs(10))
    .unwrap();
```

## Advanced Examples

### Traveling Salesman Problem

The `examples/tsp_model.rs` demonstrates solving TSP using multiple algorithms with custom tabu lists and progress callbacks. It reads city coordinates from a file and compares performance against optimal routes.

Key features shown:
- Custom `TabuList` implementation for move prohibition
- Parallel optimizer comparison (Hill Climbing, Simulated Annealing, Tabu Search, etc.)
- Progress bars with acceptance ratio monitoring
- Optimal route validation

### Additional Capabilities

You can also add `preprocess_solution` and `postprocess_solution` to your model for setup and result formatting. See the examples for complete implementations.

## API Documentation

- [API Reference](https://docs.rs/localsearch) - Complete generated documentation
- [API Design](API.md) - Detailed trait and method documentation
- [Algorithm Reference](algorithm.md) - In-depth algorithm descriptions and parameters

## Contributing

Contributions are welcome! Please follow these guidelines:

- Follow the [repository conventions](AGENTS.md) for code style and documentation
- Ensure all public items have documentation (enforced by `#![forbid(missing_docs)]`)
- Run pre-commit checks: `cargo fmt`, `cargo clippy`, `cargo test`
- Add tests for new functionality

Report issues and feature requests at: [GitHub Issues](https://github.com/lucidfrontier45/localsearch/issues)

## Performance & Compatibility

- **Parallel Execution**: All algorithms leverage Rayon for CPU-parallel candidate evaluation
- **Thread Safety**: All types implement `Sync + Send` for concurrent use
- **WASM Support**: Conditional compilation available for web targets
- **Rust Version**: Requires Rust 1.92+

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.