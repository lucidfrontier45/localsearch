# Development Commands for localsearch

## Build Commands
- `cargo build` - Build the project in debug mode
- `cargo build --release` - Build the project in release mode for optimal performance

## Test Commands
- `cargo test` - Run all tests (unit and integration) in parallel
- `cargo test -- --nocapture` - Run tests with output visible
- `cargo test <test_name>` - Run a specific unit test
- `cargo test --test <file_stem>` - Run a specific integration test file
- `cargo test <module_name>::` - Run tests in a specific module
- `cargo test -- --test-threads=1` - Run tests sequentially (not parallel)
- `cargo test -- --nocapture --backtrace` - Run tests with output and backtrace on failures

## Code Quality Commands
- `cargo fmt` - Format all code according to Rust style guidelines
- `cargo fmt --check` - Check if code is properly formatted (CI use)
- `cargo clippy` - Run Clippy linter to check for common mistakes and style issues
- `cargo clippy --fix --allow-dirty` - Automatically fix clippy issues where possible
- `cargo clippy --all-targets -- -D warnings` - Strict clippy check treating warnings as errors

## Documentation Commands
- `cargo doc --no-deps` - Generate documentation and check for missing docs
- `cargo doc --open` - Generate and open documentation in browser

## Utility Commands
- `cargo check` - Quick compile check without generating binaries
- `cargo clean` - Remove build artifacts
- `cargo update` - Update dependencies to latest compatible versions

## Examples and Benchmarks
- `cargo run --example <example_name>` - Run an example (e.g., `cargo run --example quadratic_model`)
- `cargo test --release -- --bench` - Run benchmark tests (if any exist)

## Pre-commit Checklist Commands
Run these before committing:
1. `cargo fmt --check`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo build --release`
4. `cargo test`
5. `cargo doc --no-deps`