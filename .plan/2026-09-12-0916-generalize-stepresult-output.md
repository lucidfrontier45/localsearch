# Plan: Generalize `StepResult` with algorithm-specific output (issue 96)

- **Date:** 2026-09-12 09:16
- **Status:** approved
- **Ref:** GitHub issue #96

## Goal

Add a generic output type parameter `O = ()` to `StepResult` with a `pub output: O`
field, preserving all existing usage (`StepResult<S, ST>` keeps working unchanged).

## Background

LNS needs no framework change — it can live entirely inside
`OptModel::generate_trial_solution()`. ALNS is different: it needs feedback from the
search process to adapt operator-selection weights (how often an operator was used,
whether its candidates produced a new global best, improved the current solution,
were accepted without improvement, or were rejected).

`TransitionType` must stay independent from this concern: it represents move-level
information (used by Tabu Search for tabu-list handling). ALNS statistics are
algorithm-level feedback and must not be encoded into `TransitionType`.

Current state:

- `StepResult<S, ST>` defined in `src/optim/generic.rs:15`
- Constructed only in `GenericLocalSearchOptimizer::step`
- Consumed by `metropolis.rs` and `parallel_tempering.rs` in type position only
  (no field-exhaustive construction outside `generic.rs`)
- Not re-exported from `src/optim.rs`

## Approach

1. `src/optim/generic.rs`: `pub struct StepResult<S, ST, O = ()>` + `pub output: O`;
   construction site in `step()` gets `output: ()` literal.
2. `src/optim.rs`: add `pub use generic::StepResult;`.
3. `#[cfg(test)]` test in `src/optim/generic.rs`: build
   `StepResult<S, ST, MyOut>` directly + transform `StepResult<S, ST>` into custom
   `O` via struct-update syntax (validates ergonomics without a helper API).
4. `API.md`: document the new type parameter and field.
5. Workflow: `cargo check -q` → `cargo test -q` → `cargo clippy -q --fix
   --allow-dirty` → `cargo clippy -q -- -D warnings`.

## Trade-offs

- No `map_output`/`with_output` helper — pub fields + struct-update syntax are
  sufficient; add when ALNS actually lands (YAGNI).
- Tabu search untouched — the `StepResult<S, ST, Vec<TransitionType>>` example in
  the issue is illustrative; any tabu refactor is a separate issue.
- Breaking change for downstream exhaustive struct literals (adding a `pub` field)
  — accepted; documented in `API.md`.

## Open questions

None.

## Next step

Edit `src/optim/generic.rs`.
