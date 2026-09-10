# Generalize `StepResult` with algorithm-specific output (issue #96)

## Goal

Add `StepResult<S, ST, O = ()>` with strict backward compat plus a segment-level ALNS demo reusing `GenericLocalSearchOptimizer` acceptance logic.

## Background

- Issue #96 (OPEN): LNS needs no framework change (it lives in `OptModel::generate_trial_solution()`); ALNS needs search feedback (operator uses, new-best / improved-current / accepted / rejected) to adapt weights, without overloading `TransitionType` (which stays move-level for Tabu in `src/optim/tabu_search.rs`).
- Current state (read-only inspection):
  - `src/optim/generic.rs:15` — `pub struct StepResult<S, ST>` with 5 fields, no `output`; `GenericLocalSearchOptimizer::step()` at `:74` returns it, constructs at `:168`.
  - `src/model.rs:22` — `generate_trial_solution() -> (Solution, TransitionType, Score)`; operator identity lives model-side.
  - `src/optim/metropolis.rs:103` — `MetropolisOptimizer::step()` wraps generic and returns `StepResult`; `src/optim/parallel_tempering.rs:173` collects `Vec<StepResult<…>>` from replicas.
  - `src/optim.rs:24` exports only `GenericLocalSearchOptimizer`, not `StepResult`; `src/lib.rs` exports `OptModel`, `AcceptanceCounter`.
  - Tests: `src/tests.rs` (`QuadraticModel`, `TransitionType = (usize, f64, f64)`) plus `src/tests/test_*.rs`; no top-level `tests/` per `AGENTS.md`.
- Confirmed scope: full ALNS demo, strict compat (`StepResult<S, ST>` keeps compiling), touch Generic + wrappers only, tests in `src/`.

## Approach

1. **Type generalization (`src/optim/generic.rs`)** — `pub struct StepResult<S, ST, O = ()> { …, pub output: O }`; keep field order, add `output` last; add `Clone`/`Debug` derives only if they don't break existing bounds; export via `src/optim.rs` (`pub use generic::{GenericLocalSearchOptimizer, StepResult}`).
2. **Default-producing `step()`** — make `GenericLocalSearchOptimizer::step()` generic over `O: Default`, returning `output: O::default()` (zero behavior change for `O = ()`); existing call sites keep compiling via default param. Same passthrough for `MetropolisOptimizer::step()` (`src/optim/metropolis.rs:103`). Leave `ParallelTempering` internals on `O = ()`; only update type annotations and struct literal (`output: ()`).
3. **ALNS module (`src/optim/alns.rs`, no `mod.rs`)** — new `pub struct AlnsStatistics { pub operator_uses: Vec<usize>, pub operator_scores: Vec<f64> }` (plus `Default`); new `AlnsOptimizer` (or `AlnsSegmentRunner`) holding a `GenericLocalSearchOptimizer` plus weight vector and score increments (e.g. new-best / improved / accepted / rejected); per segment: call inner `step()` with `O = ()`, compute segment delta from `best_score`/`last_score` plus `acceptance_counter`, build outer `StepResult<S, ST, AlnsStatistics>` by wrapping/augmenting (issue's "wrap" option). Operator selection/weights stay model-side or in the ALNS struct; LNS destroy/repair stays in `generate_trial_solution()`.
4. **Tests (`src/tests/test_alns.rs`, register in `src/tests.rs`)** — (a) `O = ()` old spelling still compiles and `output == ()`; (b) custom `O` (e.g. `Vec<TransitionType>`, `AlnsStatistics`) carries through; (c) ALNS demo: 2-operator stub model with interior-mutability counters, 2–3 segments, assert weights move toward winning operator and best score non-worsening; (d) tabu path untouched (`TransitionType` not overloaded).
5. **Docs** — update `API.md`/`algorithm.md` `StepResult`/`TransitionType` separation in same change; doc-comment on `output` as "algorithm-specific step output".
6. **Verify** per `AGENTS.md`: `cargo check -q` → `cargo test -q` → `cargo clippy -q --fix --allow-dirty` → `cargo clippy -q -- -D warnings`, fixing from step 1 on failure.

## Trade-offs

- `O = ()` default vs. no default / associated type on `LocalSearchOptimizer`: default chosen — preserves `StepResult<S, ST>` spelling; associated type rejected as larger breaking change across all optimizers.
- Wrapper/augment (inner `O = ()` → outer `AlnsStatistics`) vs. per-iteration collector/observer hook in `step()` now: wrapper chosen — closes the demo without threading operator IDs through parallel trial closures; full hook (closure param, per-trial outcome stream) deferred as follow-up.
- `O: Default` bound on `step()` vs. separate `step_with_output()` overload: `Default` chosen for minimal diff; overload rejected for now (extra API surface, unneeded for segment-level demo).
- Stats in `TransitionType` vs. `StepResult::output`: rejected per issue — keeps Tabu move semantics clean.

## Open questions

- Scoring weights for `AlnsStatistics` categories (values for new-best / improved / accepted / rejected?) and weight-update rule (exponential smoothing factor, segment length)?
- How does operator identity flow in the demo — stub model with `RefCell<Vec<usize>>` use-counts, or `TransitionType` carrying operator index? Former keeps `TransitionType` clean but needs interior mutability under `par_iter`.
- Does `StepResult` need `Clone`/`Debug` bounds on `O`, and should `AlnsStatistics` live in `optim/alns.rs` vs. test-only stub?

## Next step

Edit `src/optim/generic.rs:15,82,168` for `O = ()` plus `output`, re-export, then `cargo check -q`.
