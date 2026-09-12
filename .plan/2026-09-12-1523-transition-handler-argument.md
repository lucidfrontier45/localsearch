# Transition Handler as Per-Call Argument

## Goal
Make `TransitionHandler` (and `TabuList`) per-call arguments of `optimize` / `step` /
`run` / `run_with_callback`, mirroring the existing `optimize_with_tabu_list` pattern. Drop the handler-as-field pattern from `GenericLocalSearchOptimizer`.

## Background
- `GenericLocalSearchOptimizer<ST, H>` stores `handler: H` and clones it inside `step`. The handler is mutated across iterations via `update`.
- `optimize_with_tabu_list` on `TabuSearchOptimizer` already follows the desired pattern: no field, `mut tabu_list: T` arg, returns `(sol, score, T)`.
- 8 wrappers (Metropolis, SA, Tsallis, AdaptiveAnnealing, GreatDeluge, LogisticAnnealing, RelativeAnnealing, EpsilonGreedy) build the handler via `to_generic() -> GenericLocalSearchOptimizer<ST, H>` then call `.optimize()`.
- `parallel_tempering.rs:180` and `population_annealing.rs:168` call `.step()` on the generic.
- `tests/test_transition_handler.rs:149` constructs `GenericLocalSearchOptimizer::new(10, 1, usize::MAX, handler)`.
- 13 `impl LocalSearchOptimizer for ...` exist.

## Approach

### 1. Reshape `LocalSearchOptimizer` trait (`src/optim/base.rs`)
Replace `optimize/run/run_with_callback` signatures with handler-taking versions.
The trait itself does NOT constrain `H`; each impl re-declares the bound via method-level `where`.

```rust
pub trait LocalSearchOptimizer<M: OptModel> {
    fn optimize<H>(..., handler: H) -> (M::SolutionType, M::ScoreType, H);
    fn run_with_callback<H>(..., handler: H, callback: ...) -> Result<...>;
    fn run<H>(..., handler: H) -> Result<...> {
        self.run_with_callback(..., &mut |_| {}, handler)
    }
}
```

### 2. Drop handler field from `GenericLocalSearchOptimizer` (`src/optim/generic.rs`)
- Struct becomes `GenericLocalSearchOptimizer<ST>` — drop `H` type param and `handler: H` field.
- `new(patience, n_trials, return_iter)` (drop handler param).
- Add `step_with_handler<H: TransitionHandler<ST>>(..., mut handler: H) -> (StepResult<..., ...>, H)` (tuple return per Q B).
- Add `optimize_with_handler<H: TransitionHandler<ST>>(..., handler: H) -> (..., H)`.
- `impl<M, ST> LocalSearchOptimizer<M> for GenericLocalSearchOptimizer<ST>` where `M: OptModel<ScoreType = ST>`: `optimize<H> where H: TransitionHandler<ST>` delegates to `optimize_with_handler`; `run_with_callback<H> where H: TransitionHandler<ST>` preprocesses → calls `optimize` → drops handler → postprocesses.

### 3. Unify tabu (`src/optim/tabu_search.rs`)
- Drop `optimize_with_tabu_list`.
- Add `optimize_with_handler<H: TabuList<Item = M::TransitionType>>(&self, ..., mut handler: H) -> (..., H)` — body is the old `optimize_with_tabu_list`.
- Add convenience `optimize_default` that builds `T::default()` + sets size, calls `optimize_with_handler`.
- `impl<T, M> LocalSearchOptimizer<M> for TabuSearchOptimizer<T>`: `optimize<H> where H: TabuList<Item = M::TransitionType>` delegates; `run_with_callback<H>` same bound.
- Default `run<H>` from the trait body works.

### 4. Update 8 wrapper optimizers (Q A = A1)
Each wrapper implements `LocalSearchOptimizer` with bound `H: Into<ConcreteHandler> + From<ConcreteHandler>`:
```rust
fn optimize<H>(..., handler: H) -> (..., H)
where H: Into<Metropolis> + From<Metropolis> {
    let m: Metropolis = handler.into();
    let opt = GenericLocalSearchOptimizer::new(self.patience, self.n_trials, self.return_iter);
    let (sol, sc, m) = opt.optimize_with_handler(..., m);
    (sol, sc, H::from(m))  // round-trip preserves state
}
```
`run_with_callback<H>` follows the same preprocess → `optimize` → drop handler → postprocess pattern with the same bound.

Wrappers touched: Metropolis, SimulatedAnnealing, TsallisRelativeAnnealing, AdaptiveAnnealing, GreatDeluge, LogisticAnnealing, RelativeAnnealing, EpsilonGreedy.

`RandomSearchOptimizer` and `HillClimbingOptimizer` delegate to `EpsilonGreedyOptimizer`; they take the same `Into<EpsilonGreedy> + From<EpsilonGreedy>` bound.

### 5. Drop `to_generic()` (Q C)
Removed from every wrapper. Parallel/population wrappers stop using it.

### 6. Update parallel/population wrappers
- `parallel_tempering.rs`: replace `m.to_generic().step(...)` with
  ```rust
  let opt = GenericLocalSearchOptimizer::new(m.patience, n_trials, m.return_iter);
  let (step_result, _handler) = opt.step_with_handler(..., Metropolis::new(self.betas[idx]));
  ```
  Collect `Vec<(StepResult<...>, Metropolis)>`.
- `population_annealing.rs`: same pattern with `Metropolis::new(current_beta)`.

### 7. Update tests (`src/tests/test_transition_handler.rs`)
- Line 149: drop the handler arg from `GenericLocalSearchOptimizer::new`; pass it to a new `step_with_handler` (or `optimize_with_handler`) call.
- Add a `step_with_handler` returns-handler test.
- Add a `TabuSearchOptimizer::optimize_with_handler` test.

### 8. Re-exports (`src/optim.rs`)
No change — `StepResult`, `TransitionHandler`, `TabuList`, `GenericLocalSearchOptimizer`, `TabuSearchOptimizer` already exported.

## Trade-offs
- Breaking API change across `LocalSearchOptimizer` trait, `GenericLocalSearchOptimizer`, `TabuSearchOptimizer`, every wrapper. Acceptable because the previous refactor (commit `d751735`) was recent and there are no external callers.
- Wrappers take an `H: Into<Concrete> + From<Concrete>` even though only `H = Concrete` is convenient. Identity `From` is auto-derived for `Concrete -> Concrete`, so the common case is zero-cost. Other H types require user-implemented round-trip conversions — pragmatic cost.
- `step_with_handler` returns tuple `(StepResult, H)` rather than extending `StepResult`. Parallel/population wrappers unpack the tuple.

## Open questions
None — all answered.

## Next step
Edit in order: `base.rs` → `generic.rs` → `tabu_search.rs` → wrappers (8) → `random.rs` + `hill_climbing.rs` → `parallel_tempering.rs` → `population_annealing.rs` → `tests/test_transition_handler.rs`. Then `cargo check -q` → `cargo test -q` → `cargo clippy -q --fix --allow-dirty` → `cargo clippy -q -- -D warnings`.
