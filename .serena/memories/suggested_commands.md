# Suggested Commands for Development

## Building and Testing
- `cargo build` - Compile the project
- `cargo check` - Check for compilation errors without building
- `cargo test` - Run all tests
- `cargo run --example quadratic_model` - Run an example
- `cargo run --example tsp_model` - Run another example

## Code Quality
- `cargo fmt` - Format code according to Rust style guidelines
- `cargo clippy` - Check code for common mistakes and improvements
- `cargo doc` - Build documentation
- `cargo clean` - Remove build artifacts

## Dependency Management
- `cargo add <crate>` - Add a dependency to Cargo.toml
- `cargo remove <crate>` - Remove a dependency from Cargo.toml
- `cargo update` - Update dependencies

## Benchmarking and Performance
- `cargo bench` - Run benchmarks (if any exist)
- `cargo run --release` - Run with optimizations enabled

## Additional Commands
- `cargo tree` - Show dependency tree
- `cargo metadata` - Get metadata about the workspace
- `rustc --version` - Check Rust compiler version