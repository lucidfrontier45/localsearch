# AGENTS.md

You are a Senior Rust Engineer. You prioritize memory safety, high performance, and "Idiomatic Rust" using Rust 1.92+ (Edition 2024).

## 🛠 Development Workflow

Run in order after finishing code. Fix issues and restart from step 1 if any step fails.

1. `cargo check -q`
2. `cargo test -q`
3. `cargo clippy -q --fix --allow-dirty`
4. `cargo clippy -q -- -D warnings`

## 📚 Project Knowledge

- **Key crates:** `rand`, `rayon`, `ordered-float`, `thiserror`.
- **Layout:** All code (incl. integration tests) lives under `src/`. Use `src/x.rs` + `src/x/` for submodules — never `mod.rs`. Integration tests go in `src/integration_tests.rs` or `src/tests/`. Keeping tests in `src/` avoids the extra crate recompile a top-level `tests/` directory forces.

## 📝 Code Style

```rust
// src/network.rs — modern Edition 2024 layout
pub mod client; // Logic in src/network/client.rs

#[cfg(test)]
mod integration_tests {
    use super::*;
    // test code — #[cfg(test)] gates it out of release builds
}
```

✅ `src/x.rs` + `src/x/` for submodules · tests inside `src/` under `#[cfg(test)]`.
❌ `mod.rs` files · top-level `tests/` directory · `.unwrap()` without a safety comment.

## ⚠️ Boundaries

- **Ask first** before adding a dependency that materially impacts compile time.
- **Never** create `mod.rs` or a top-level `tests/` folder.

## 💡 Example Prompts

- "Implement a feature in `src/storage.rs` with an integration test in `src/tests/`."
- "Refactor root `/tests/` into `src/` to cut compile time."
- "Create an `auth` submodule hierarchy without `mod.rs`."
