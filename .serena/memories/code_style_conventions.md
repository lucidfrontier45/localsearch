# Code Style and Conventions for localsearch

## Naming Conventions
- **Functions/methods**: `snake_case` (e.g., `generate_random_solution`)
- **Variables/parameters**: `snake_case` (e.g., `current_solution`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `VERSION`)
- **Types/structs/enums**: `CamelCase` (e.g., `HillClimbingOptimizer`, `OptModel`)
- **Traits**: `CamelCase` (e.g., `LocalSearchOptimizer`)
- **Modules**: `snake_case` (e.g., `optim`, `utils`)
- **Type parameters**: Single uppercase letter (T, U, V) or descriptive names (e.g., `SolutionType`, `ScoreType`)
- **Associated types**: `CamelCase` with descriptive suffixes (e.g., `SolutionType`, `ScoreType`, `TransitionType`)

## Documentation Requirements
- **Strict enforcement**: `#![forbid(missing_docs)]` - ALL public items MUST be documented
- **Format**: Use `///` for items, `//!` for modules/crates
- **Function docs**: Document parameters, return values, and behavior
- **Example usage**: Include code examples where helpful
- **Intra-doc links**: Use `[`link`]()` syntax when referencing other items

## Import Organization
Imports are grouped into 4 different categories:
1. std
2. external crates
3. this crate
4. super

The 4 categories are sorted in that order. Different categories must be separated by 1 newline.

Imports of the same prefix must be merged. Within each category, sort imports alphabetically.

Use `cargo fmt` for final adjustment to ensure consistent formatting.

```rust
// 1. std
use std::{collections::HashMap, sync::Arc};

// 2. external crates
use ordered_float::NotNan;
use rand::{distr::Uniform, prelude::*};
use thiserror::Error;

// 3. this crate
use crate::{optim::{HillClimbingOptimizer, LocalSearchOptimizer}, Duration, OptModel};

// 4. super
use super::{SomeTrait, SomeType};
```

## Error Handling
- **Public APIs**: Use `Result<T, LocalsearchError>` for fallible operations (defined with thiserror)
- **Internal code**: Use `Result<T, LocalsearchError>` for consistency; avoid `unwrap()` and `expect()`
- **Library code**: Avoid `unwrap()` and `expect()` - return proper errors instead
- **Panic documentation**: Document any function that may panic
- **Error propagation**: Use `?` operator for clean error propagation

## Type System Usage
- **Public API types**: Prefer concrete types over generic bounds for stability
- **Internal ergonomics**: Use `impl Trait` for function parameters/return types
- **Auto-implementations**: Use `#[auto_impl(&, Box, Rc, Arc)]` for common smart pointer implementations
- **Trait bounds**: Use `Sync + Send` for types that may be used in parallel contexts
- **Associated types**: Use descriptive names (SolutionType, ScoreType, TransitionType)

## Safety and Performance
- **Panics**: Avoid panics in library code; document when they may occur
- **Thread safety**: All optimizers and models should be `Sync + Send`
- **Parallelization**: Leverage Rayon for CPU-intensive operations
- **Zero-cost abstractions**: Prefer compile-time polymorphism over runtime dispatch
- **Memory efficiency**: Use owned types where possible, avoid unnecessary allocations

## Testing Patterns
- **Unit tests**: Place next to modules using `#[cfg(test)]` attribute
- **Integration tests**: Place in `src/tests/` directory
- **Approximate comparisons**: Use `approx` crate for floating-point comparisons
- **Test organization**: Group related tests in modules with clear names

## Project Layout
- **Module files**: Prefer `a.rs` to `a/mod.rs`
- **Trait definitions**: Place in dedicated modules (e.g., `base.rs`, `model.rs`)
- **Algorithm implementations**: Group by category in `optim/` subdirectory
- **Utility functions**: Place in `utils.rs` or dedicated modules
- **Constants**: Define at module level with clear documentation
- **Type aliases**: Use when they improve readability (e.g., `type ScoreType = NotNan<f64>`)

## Clippy Allowances
- `clippy::non_ascii_literal` - Allowed for mathematical expressions
- `clippy::module_name_repetitions` - Allowed for clarity

## Design Patterns
- **Trait-based architecture**: Users implement `OptModel` trait to define optimization problems
- **Associated types**: Used extensively for type-safe generic programming
- **Builder pattern**: Some optimizers use builder-style construction
- **Callback pattern**: Optional progress callbacks during optimization