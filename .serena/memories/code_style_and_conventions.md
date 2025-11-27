# Code Style and Conventions

## Rust Edition
- The project uses Rust edition 2024 with rust-version 1.85

## Code Organization
- Each optimization algorithm is in its own file in `src/optim/`
- Core traits are defined in `src/model.rs` and `src/optim/base.rs`
- Utility functions in `src/utils.rs`
- Callback handling in `src/callback.rs`
- Time handling in `src/time_wrapper.rs`

## Naming Conventions
- Structs and Traits: PascalCase (e.g., `HillClimbingOptimizer`, `LocalSearchOptimizer`, `OptModel`)
- Functions and Variables: snake_case (e.g., `generate_random_solution`, `run_with_callback`)
- Type Aliases: PascalCase (e.g., `ScoreType`, `SolutionType`)

## Trait Implementation
- Uses `auto_impl` crate to automatically implement traits for common smart pointer types
- All optimization algorithms implement the `LocalSearchOptimizer` trait
- All optimization problems implement the `OptModel` trait

## Documentation
- The project forbids missing docs (`#![forbid(missing_docs)]`)
- Documentation includes the README content via `#![doc = include_str!("../README.md")]`
- Public items need to be documented

## Clippy Lint Exceptions
- `#![allow(clippy::non_ascii_literal)]` - allows non-ASCII literals
- `#![allow(clippy::module_name_repetitions)]` - allows module name repetitions

## State Update Pattern
- All optimizers follow the unified update order defined in `state_update_order.md`
- This ensures consistency across all optimization algorithms
- The order is: time counters → best solution → accepted counter → current solution → return to best → patience check → algorithm-specific state → callback

## Generic Type Constraints
- Models must implement `Sync + Send` to support parallelization
- Score types must implement `Ord + Copy + Sync + Send`
- Solution and transition types must implement `Clone + Sync + Send`