# Repository Agent Guidelines

This file documents repository conventions and guidance for automated agents and contributors.

General
- Rust edition: `2024` (see `Cargo.toml`).
- Rust toolchain: minimum `rustc 1.85` per `Cargo.toml`.

Build & Test
- Build (release): `cargo build --release`
- Run all tests: `cargo test`
- Run a single unit test: `cargo test <test_name>`
- Run an integration test file: `cargo test --test <file_stem>`
- Show test stdout: `cargo test -- --nocapture`
- Run clippy autofix: `cargo clippy --fix --allow-dirty`
- Strict clippy check: `cargo clippy --all-targets -- -D warnings`
- Format: `cargo fmt`

Project layout & style
- Module/layout: prefer `a.rs` to `a/mod.rs` (repository convention).
- Documentation: crate uses `#![forbid(missing_docs)]` in `src/lib.rs` — document public items.
- Imports ordering: group and order as `std` → external crates → `crate::` (prefer absolute `crate::` paths).
- Naming: `snake_case` for functions/variables, `CamelCase` for types/enums/structs, `SCREAMING_SNAKE_CASE` for constants.
- Public API types: prefer concrete types on public APIs; use `impl Trait` for internal ergonomics.
- Error handling: prefer `anyhow::Result` for binaries, tests, and internal tooling; avoid `unwrap()`/`expect()` in library code.
- Safety: avoid panics in library code; document any function that may panic.
- Tests: place integration tests in `src/tests`; unit tests next to modules with `#[cfg(test)]`.

Pre-push local checklist
- Run `cargo fmt` and `cargo clippy --fix --allow-dirty` before opening a PR.
- Ensure public items have documentation.

Agent behaviour and interactions
- Preambles: when an agent plans to run shell commands or edits, include a 1–2 sentence preamble describing the immediate next steps.
- Edits: prefer small, focused edits; do not change public APIs without maintainer approval.
- Commits: agents MUST NOT create git commits unless explicitly requested by a maintainer.

CI and tests guidance
- If changing behavior that affects tests, run `cargo test` locally and include failing test output when asking for help.

IDE / AI hints
- There are no repository-specific Copilot or cursor rules. If desired, add `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md`.

Public API changes
- Consult the repo owner before switching public `Result` types to `anyhow::Result` — this is a breaking design choice.

Contact
- Repository: https://github.com/lucidfrontier45/localsearch

(End)
