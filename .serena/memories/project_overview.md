# LocalSearch Project Overview

## Purpose
The localsearch library is a Rust library for local search optimization that implements various metaheuristic algorithms. It provides implementations of 10 different optimization algorithms that are parallelized with Rayon.

## Algorithms Implemented
1. Hill Climbing
2. Tabu Search (requires problem-specific tabu list implementation)
3. Simulated Annealing
4. Epsilon Greedy Search
5. Relative Annealing
6. Logistic Annealing
7. Adaptive Annealing
8. Metropolis (Markov chain Monte Carlo)
9. Population Annealing
10. Tsallis Relative Annealing

## Tech Stack
- Language: Rust (edition 2024)
- Dependencies:
  - rand (0.9.0) - for random number generation
  - ordered-float (5.0.0) - for ordered floating point types
  - rayon (1.10.0) - for parallelization
  - auto_impl (1.2.0) - for automatic trait implementations
  - anyhow (1.0.86) - for error handling
  - web-time (1.1.0) - for WASM targets (target dependency)
- Dev Dependencies:
  - approx (0.5.1) - for approximate equality comparisons
  - indicatif (0.17.8) - for progress bars

## Core Architecture
- `OptModel` trait: Defines the interface for optimization models
- `LocalSearchOptimizer` trait: Defines the interface for optimization algorithms
- Modular structure in `src/optim/` with separate files for each algorithm

## Key Features
- Parallelized algorithms using Rayon
- Callback system for monitoring optimization progress
- Time limits and iteration limits for optimization runs
- Preprocessing and postprocessing hooks for solutions
- Consistent state update ordering across optimizers (defined in state_update_order.md)