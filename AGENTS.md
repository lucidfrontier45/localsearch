AGENTS - Agent Guidelines for this repository

Build / Lint / Test
- Build: `cargo build --all`.
- Run all tests: `cargo test --all`.
- Run a single test (by name filter): `cargo test <test_name> -- --nocapture`.
- Run a specific integration test binary: `cargo test --test <binary_name>`.
- Format check: `cargo fmt --all -- --check` or auto-format with `cargo fmt`.
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`.

Code style and conventions
- Formatting: use `rustfmt`/`cargo fmt`; submit code already formatted.
- Imports: order as `std::...`, external crates, `crate::...` or `super::...`; group and avoid wildcard imports.
- Types & signatures: prefer explicit types on public APIs; use `usize` for indexing and collection sizes.
- Naming: `snake_case` for functions/variables, `CamelCase` for types/traits/enums, `SCREAMING_SNAKE_CASE` for constants.
- Error handling: prefer `Result<T, E>` and `?` propagation; avoid `unwrap`/`expect` in library code (tests/main only if justified).
- Panics: avoid panicking in library code; return errors instead.
- Documentation: add `///` doc comments for public items and explain panics/errors.

Agent behaviour
- Follow repository AGENTS.md scope rules. No `.cursor` or Copilot instruction files detected in this repo.
- Keep changes minimal and focused; run tests locally after edits and include test commands in PR descriptions.