# Task Completion Checklist for localsearch

## Pre-push Local Checklist
Before committing or pushing changes, ensure all of these pass:

- [ ] `cargo fmt --check` - Code is properly formatted
- [ ] `cargo clippy --fix --allow-dirty` - Lints are resolved (fix automatically where possible)
- [ ] `cargo clippy --all-targets -- -D warnings` - Strict clippy check (warnings as errors)
- [ ] `cargo build --release` - Code compiles without warnings in release mode
- [ ] `cargo test` - All tests pass (both unit and integration)
- [ ] `cargo doc --no-deps` - Documentation builds without warnings
- [ ] Public items have documentation (required by `#![forbid(missing_docs)]`)
- [ ] No `unwrap()`/`expect()` in library code
- [ ] Error handling follows `anyhow::Result` pattern
- [ ] Parallel safety: types implement `Sync + Send` where appropriate

## Code Quality Requirements
- **Documentation**: ALL public items must be documented (enforced by `#![forbid(missing_docs)]`)
- **Formatting**: Must pass `cargo fmt --check`
- **Linting**: Must pass strict clippy checks
- **Testing**: All tests must pass
- **Error handling**: No unwrap/expect in library code
- **Thread safety**: Types used in parallel contexts must be `Sync + Send`

## CI Requirements
- **Formatting check**: `cargo fmt --check`
- **Clippy linting**: `cargo clippy --all-targets -- -D warnings`
- **Build check**: `cargo build --release`
- **Test suite**: `cargo test`
- **Documentation**: `cargo doc --no-deps`

## Performance Considerations
- **Parallelization**: Ensure changes don't break Rayon parallelization
- **Thread safety**: Maintain `Sync + Send` bounds where required
- **Memory efficiency**: Avoid unnecessary allocations
- **Zero-cost abstractions**: Preserve compile-time polymorphism benefits

## API Stability
- **Breaking changes**: Consult maintainer before changing public APIs
- **Trait signatures**: Avoid changing trait method signatures without major version bump
- **Associated types**: Changes to associated types are breaking
- **Deprecation**: Use `#[deprecated]` attribute for soft deprecation with migration guides

## Testing Requirements
- **Unit tests**: Test individual functions/structs in isolation
- **Integration tests**: Test end-to-end functionality
- **Floating-point tests**: Use `approx` crate for floating-point comparisons
- **Parallel testing**: Ensure tests work correctly with Rayon's parallel execution

## Documentation Updates
- **Public API changes**: Update documentation when changing public interfaces
- **Examples**: Keep code examples in README and docs up to date
- **Intra-doc links**: Use proper linking syntax for cross-references