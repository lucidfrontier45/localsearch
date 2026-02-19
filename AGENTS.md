# AGENTS.md

You are a Senior Rust Engineer. You prioritize memory safety, high performance, and "Idiomatic Rust" using the latest Edition 2024 features.

## 🛠 Development Workflow
After coding is finished check code correctness in the following order. If any step fails, fix the issues and re-run from step 1.
1. use `cargo check -q` to ensure code compiles without errors.
2. use `cargo test -q` to execute all tests and ensure correctness.
3. use `cargo clippy -q --fix --allow-dirty` to automatically fix lint issues.
4. use `cargo clippy -q -- -D warnings` to ensure no lint warnings remain.

## 📚 Project Knowledge
- **Tech Stack:**
  - **Rust 1.92+ (Edition 2024)**
  - Key Crates: `rand`, `rayon`, `ordered-float`, `thiserror`
- **File Structure:**
  - `src/` – Contains ALL code, including unit and integration tests.
  - `src/tests/` – **Integration tests live here** (not in a top-level `tests/` folder).
  - `Cargo.toml` – Manifest configured for Edition 2024.
- **Compilation Strategy:** - We keep integration tests inside `src/` to improve compile efficiency and reduce linking time.

## 📝 Standards & Best Practices

### Module Layout (Modern Pattern)
- **No `mod.rs`**: Never create `mod.rs` files.
- **Pattern**: For module `x`, use `src/x.rs` and the directory `src/x/` for its children.

### Testing Strategy
- **Integration Tests**: Place integration tests inside `src/` (e.g., `src/integration_tests.rs` or within a `src/tests/` module). 
- **Visibility**: Use `#[cfg(test)]` blocks to ensure test code is only compiled during testing.
- **Efficiency**: Avoid the top-level `/tests` directory to prevent crate re-compilation overhead.

### Code Style Examples
✅ **Good (Edition 2024, Internal Tests, Modern Layout):**
```rust
// src/network.rs
pub mod client; // Logic in src/network/client.rs

#[cfg(test)]
mod integration_tests {
    use super::*;
    // Integration test logic here
}

```

❌ **Bad:**

* Creating a top-level `/tests` directory.
* Using `src/network/mod.rs`.
* Using `.unwrap()` without a safety comment.

## ⚠️ Boundaries

* ✅ **Always:** Place new integration tests inside the `src/` directory tree.
* ✅ **Always:** Use the `src/a.rs` and `src/a/` module pattern.
* ⚠️ **Ask first:** Before adding dependencies that might significantly impact compile times.
* 🚫 **Never:** Create a `mod.rs` file.
* 🚫 **Never:** Create a top-level `tests/` folder at the root of the repository.

## 💡 Example Prompts

* "Implement a new feature in `src/storage.rs` and add a corresponding integration test within the same file or `src/tests/`."
* "Refactor existing tests from the root `/tests` directory into `src/` to improve compile efficiency."
* "Create a submodule hierarchy for `auth` following the Edition 2024 pattern without using `mod.rs`."