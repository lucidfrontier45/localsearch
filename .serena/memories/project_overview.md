# Project Overview: localsearch

## Purpose
A Rust library for local search optimization algorithms implementing various metaheuristic optimization techniques. Provides parallelized implementations of algorithms like Hill Climbing, Simulated Annealing, Tabu Search, Epsilon Greedy Search, Relative Annealing, Logistic Annealing, Adaptive Annealing, Metropolis, Population Annealing, Tsallis Relative Annealing, Parallel Tempering, and Great Deluge.

## Tech Stack
- **Language**: Rust 2024 edition (minimum rustc 1.85)
- **Type**: Library crate
- **Parallelization**: Rayon for CPU-intensive operations
- **Key Dependencies**:
  - `rand` (0.9.0): Random number generation
  - `ordered-float` (5.0.0): Ordered floating-point types for scores
  - `rayon` (1.10.0): Parallel computation framework
  - `auto_impl` (1.2.0): Automatic trait implementations for smart pointers
  - `anyhow` (1.0.86): Error handling
- **Dev Dependencies**:
  - `approx` (0.5.1): Approximate floating-point comparisons in tests
  - `indicatif` (0.18.3): Progress bars for examples
- **WASM Support**: Conditional compilation for web targets using `web-time`

## Platform Support
- Primary targets: Linux, macOS, Windows
- WASM support with conditional compilation

## Key Features
- All algorithms are parallelized with Rayon
- Users implement `OptModel` trait to define their optimization problems
- Comprehensive set of local search metaheuristics
- Thread-safe design with `Sync + Send` requirements