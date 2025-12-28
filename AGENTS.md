# Repository Agent Guidelines

This file documents repository conventions and guidance for automated agents and contributors.

## General

- **Rust edition**: `2024` (see `Cargo.toml`)
- **Rust toolchain**: minimum `rustc 1.92` per `Cargo.toml`
- **Project type**: Library crate for local search optimization algorithms
- **Parallelization**: Uses Rayon for parallel computations across all algorithms
- **Documentation**: `#![forbid(missing_docs)]` enforced - ALL public items must be documented

## Build & Test Commands

### Core Commands
- **Build (debug)**: `cargo build`
- **Build (release)**: `cargo build --release`
- **Run all tests**: `cargo test`
- **Run tests with output**: `cargo test -- --nocapture`
- **Format code**: `cargo fmt`
- **Check formatting**: `cargo fmt --check`
- **Lint with clippy**: `cargo clippy`
- **Lint autofix**: `cargo clippy --fix --allow-dirty`
- **Strict clippy check**: `cargo clippy --all-targets -- -D warnings`

### Running Specific Tests
- **Single unit test**: `cargo test <test_name>`
- **Single integration test file**: `cargo test --test <file_stem>`
- **Run tests in specific module**: `cargo test <module_name>::`
- **Run tests matching pattern**: `cargo test -- <pattern>`
- **Run tests with backtrace**: `cargo test -- --nocapture --backtrace`

### Advanced Testing
- **Run tests in parallel (default)**: `cargo test` (uses all available cores)
- **Run tests sequentially**: `cargo test -- --test-threads=1`
- **Run specific test with output**: `cargo test <test_name> -- --nocapture`
- **Benchmark tests**: `cargo test --release -- --bench` (if benchmarks exist)

## Code Style Guidelines

### Project Layout & Module Structure
- **Module files**: Prefer `a.rs` to `a/mod.rs` (repository convention)
- **Integration tests**: Place in `src/tests/` directory
- **Unit tests**: Place next to modules using `#[cfg(test)]` attribute
- **Examples**: Place in `examples/` directory with descriptive names
- **Documentation tests**: Enabled in `Cargo.toml` (doctest = false means disabled for lib, but examples can have them)

### Naming Conventions
- **Functions/methods**: `snake_case`
- **Variables/parameters**: `snake_case`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Types/structs/enums**: `CamelCase`
- **Traits**: `CamelCase`
- **Modules**: `snake_case`
- **Type parameters**: Single uppercase letter (T, U, V) or descriptive (SolutionType, ScoreType)
- **Associated types**: `CamelCase` with descriptive suffixes (SolutionType, ScoreType, TransitionType)

### Documentation Standards
- **Public items**: ALL public functions, structs, traits, enums, constants must have documentation
- **Documentation format**: Use triple slashes `///` for items, `//!` for modules/crates
- **Function docs**: Document parameters, return values, and behavior
- **Example usage**: Include code examples where helpful
- **Links**: Use intra-doc links with `[`link`]()` syntax when referencing other items

### Import Organization
```rust
// Group and order imports as:
// 1. Standard library imports
use std::collections::HashMap;
use std::sync::Arc;

// 2. External crate imports (alphabetical within group)
use anyhow::Result as AnyResult;
use ordered_float::NotNan;
use rand::prelude::*;

// 3. Local crate imports (prefer absolute paths)
use crate::optim::{HillClimbingOptimizer, LocalSearchOptimizer};
use crate::{Duration, OptModel};
```

### Error Handling
- **Public APIs**: Use `anyhow::Result<T>` for fallible operations
- **Internal code**: Use `anyhow::Result<T>` for convenience in tests and tooling
- **Library code**: Avoid `unwrap()` and `expect()` - return proper errors instead
- **Panic documentation**: Document any function that may panic with `#[must_use]` or panic docs
- **Error propagation**: Use `?` operator for clean error propagation

### Type System Usage
- **Public API types**: Prefer concrete types over generic bounds for stability
- **Internal ergonomics**: Use `impl Trait` for function parameters/return types
- **Auto-implementations**: Use `#[auto_impl(&, Box, Rc, Arc)]` for common smart pointer implementations
- **Trait bounds**: Use `Sync + Send` for types that may be used in parallel contexts
- **Associated types**: Use descriptive names (SolutionType, ScoreType, TransitionType)

### Safety & Performance
- **Panics**: Avoid panics in library code; document when they may occur
- **Thread safety**: All optimizers and models should be `Sync + Send`
- **Parallelization**: Leverage Rayon for CPU-intensive operations
- **Zero-cost abstractions**: Prefer compile-time polymorphism over runtime dispatch where appropriate
- **Memory efficiency**: Use owned types where possible, avoid unnecessary allocations

### Testing Patterns
- **Unit tests**: Test individual functions/structs in isolation
- **Integration tests**: Test end-to-end functionality in `src/tests/`
- **Property-based testing**: Use appropriate crates if complex invariants exist
- **Approximate comparisons**: Use `approx` crate for floating-point comparisons
- **Test organization**: Group related tests in modules with clear names

### Code Organization
- **Trait definitions**: Place in dedicated modules (e.g., `base.rs`, `model.rs`)
- **Algorithm implementations**: Group by category (annealing, local search, etc.)
- **Utility functions**: Place in `utils.rs` or dedicated modules
- **Constants**: Define at module level with clear documentation
- **Type aliases**: Use when they improve readability (e.g., `type ScoreType = NotNan<f64>`)

## Pre-push Local Checklist

- [ ] `cargo fmt` - Code is properly formatted
- [ ] `cargo clippy --fix --allow-dirty` - Lints are resolved
- [ ] `cargo build --release` - Code compiles without warnings
- [ ] `cargo test` - All tests pass
- [ ] `cargo doc --no-deps` - Documentation builds without warnings
- [ ] Public items have documentation (required by `#![forbid(missing_docs)]`)
- [ ] No `unwrap()`/`expect()` in library code
- [ ] Error handling follows `anyhow::Result` pattern
- [ ] Parallel safety: types implement `Sync + Send` where appropriate

## Agent Behavior and Interactions

### Command Execution
- **Preambles**: When planning shell commands or edits, include 1-2 sentence preamble describing immediate next steps
- **Command explanation**: Explain what each command does and why it's being run
- **Sequential operations**: Use `&&` for dependent commands, `;` for independent ones
- **Error handling**: Check command results and handle failures appropriately

### Code Editing Guidelines
- **Small, focused edits**: Prefer minimal changes over large refactors
- **Public API changes**: Require maintainer approval for breaking changes
- **Documentation updates**: Update docs when changing public interfaces
- **Import organization**: Maintain the std → external → crate grouping
- **Type consistency**: Follow existing patterns for associated types and generics

### Commit Guidelines
- **Agent commits**: Agents MUST NOT create commits unless explicitly requested by maintainer
- **Commit messages**: Follow conventional commit format when allowed
- **Staged changes**: Only commit what's been reviewed and tested

### Communication Style
- **Concise responses**: Keep responses under 4 lines unless detail is requested
- **Direct answers**: Answer questions directly without unnecessary preamble
- **Tool usage**: Explain non-trivial commands and their purpose
- **Error reporting**: Include relevant error output when tests/commands fail

## CI and Testing Guidance

### Test Execution
- **Local testing**: Run `cargo test` before submitting changes that affect behavior
- **Failing tests**: Include full failing test output when asking for help
- **Performance impact**: Test algorithm performance changes with appropriate benchmarks
- **Parallel testing**: Ensure tests work correctly with Rayon's parallel execution

### Code Quality Checks
- **Clippy warnings**: All warnings must be resolved before merging
- **Documentation**: Missing docs will cause compilation failure due to `#![forbid(missing_docs)]`
- **Formatting**: Code must pass `cargo fmt --check`
- **Dead code**: Remove unused code and dependencies

## Serena MCP (Model Context Protocol) Usage

Serena MCP provides semantic code analysis and editing capabilities for AI-assisted development. This project is configured to work optimally with Serena MCP tools.

### Available Serena Tools

#### Code Exploration Tools
- **serena_list_dir**: List files and directories in the project
- **serena_find_file**: Find files matching patterns
- **serena_search_for_pattern**: Search for arbitrary patterns in the codebase
- **serena_get_symbols_overview**: Get high-level overview of symbols in a file
- **serena_find_symbol**: Find symbols (functions, classes, etc.) by name pattern
- **serena_find_referencing_symbols**: Find all references to a specific symbol

#### Code Editing Tools
- **serena_replace_symbol_body**: Replace the entire body of a symbol (function, method, etc.)
- **serena_insert_after_symbol**: Insert code after a symbol definition
- **serena_insert_before_symbol**: Insert code before a symbol definition
- **serena_rename_symbol**: Rename a symbol throughout the codebase

#### Memory Management Tools
- **serena_write_memory**: Save project information to memory files
- **serena_read_memory**: Read previously saved memory files
- **serena_list_memories**: List available memory files
- **serena_edit_memory**: Edit existing memory files
- **serena_delete_memory**: Delete memory files

#### Project Management Tools
- **serena_activate_project**: Activate a project for Serena MCP usage
- **serena_get_current_config**: Get current Serena configuration
- **serena_check_onboarding_performed**: Check if project onboarding is complete
- **serena_onboarding**: Perform initial project onboarding
- **serena_think_about_collected_information**: Analyze collected information
- **serena_think_about_task_adherence**: Check task progress alignment
- **serena_think_about_whether_you_are_done**: Determine if a task is complete

### Best Practices for Serena MCP Usage

#### Code Exploration Workflow
1. **Start with overview**: Use `serena_get_symbols_overview` to understand file structure
2. **Find specific symbols**: Use `serena_find_symbol` with appropriate depth to locate functions/methods
3. **Explore relationships**: Use `serena_find_referencing_symbols` to understand symbol usage
4. **Search patterns**: Use `serena_search_for_pattern` for complex text searches

#### Code Editing Workflow
1. **Identify target**: Use exploration tools to locate the exact symbol to modify
2. **Choose appropriate tool**:
   - `serena_replace_symbol_body`: For complete function/method replacements
   - `serena_insert_after_symbol`: For adding code after existing symbols
   - `serena_insert_before_symbol`: For adding code before existing symbols
3. **Verify changes**: Use exploration tools to confirm modifications

#### Memory Management
- **Read existing memories**: Check `suggested_commands.md`, `project_overview.md`, `code_style_conventions.md`, and `task_completion_checklist.md` before starting work
- **Save project knowledge**: Use `serena_write_memory` to document important findings
- **Update conventions**: Edit memory files when project standards change

#### Project Setup
1. **Activate project**: Use `serena_activate_project /path/to/project` to initialize
2. **Check onboarding**: Use `serena_check_onboarding_performed` to verify setup
3. **Run onboarding**: Use `serena_onboarding` if setup is incomplete

### Serena MCP Integration Guidelines

#### When to Use Serena Tools
- **Complex code exploration**: When you need to understand relationships between symbols
- **Precise editing**: When modifying specific functions, methods, or classes
- **Large codebase navigation**: When working with extensive codebases that need semantic understanding
- **Refactoring**: When renaming symbols or understanding impact of changes

#### When to Use Standard Tools
- **Simple file operations**: Use regular file tools for basic reading/writing
- **Non-code files**: Use standard tools for documentation, config files, etc.
- **Bulk operations**: Use standard tools for operations affecting multiple files simultaneously

#### Performance Considerations
- **Token efficiency**: Serena tools are optimized for semantic understanding with minimal context usage
- **Selective reading**: Avoid reading entire files when symbol-level information suffices
- **Memory utilization**: Leverage memory files to persist project knowledge across sessions

## IDE / AI Assistant Configuration

### Current Status
- **Copilot instructions**: None currently defined
- **Cursor rules**: None currently defined
- **Repository-specific rules**: Not established

### Recommended Additions
If adding AI assistant configuration, consider:
- `.github/copilot-instructions.md` for GitHub Copilot guidance
- `.cursor/rules/` directory for Cursor-specific rules
- `.cursorrules` file for general Cursor configuration

## Public API Stability

### Breaking Changes
- **Result types**: Consult repo owner before changing public `Result<T, E>` types to `anyhow::Result<T>`
- **Trait methods**: Avoid changing trait method signatures without major version bump
- **Associated types**: Changes to associated types are breaking changes
- **Type parameters**: Adding/removing generic parameters is a breaking change

### Deprecation Strategy
- **Soft deprecation**: Mark deprecated items with `#[deprecated]` attribute
- **Migration guides**: Provide clear migration instructions in deprecation messages
- **Version planning**: Plan breaking changes for major version releases

## Dependencies and Compatibility

### Core Dependencies
- **rand**: Random number generation
- **ordered-float**: Ordered floating-point types for scores
- **rayon**: Parallel computation framework
- **auto_impl**: Automatic trait implementations for smart pointers
- **anyhow**: Error handling

### Development Dependencies
- **approx**: Approximate floating-point comparisons in tests
- **indicatif**: Progress bars for examples

### Platform Support
- **Primary targets**: Linux, macOS, Windows
- **WASM support**: Conditional compilation for web targets using `web-time`

## Contact and Resources

- **Repository**: https://github.com/lucidfrontier45/localsearch
- **Issues**: https://github.com/lucidfrontier45/localsearch/issues
- **Documentation**: Generated by `cargo doc --open`
- **Examples**: See `examples/` directory for usage patterns

(End)
