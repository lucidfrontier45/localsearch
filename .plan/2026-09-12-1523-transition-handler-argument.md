# Transition Handler as Internal Detail (revised)

## Goal

Keep `TransitionHandler` as an internal mechanism: `GenericLocalSearchOptimizer`
takes the handler per-call via `step` / `optimize_with_handler`, but the public
`LocalSearchOptimizer` trait takes NO handler argument. Each wrapper optimizer
builds its own concrete handler internally from its config and discards it.
This supersedes the earlier per-call-argument design (handler leaked into the
trait), which was reverted: callers could supply wrong/duplicated config
(e.g. `GreatDeluge::new(0.0)` vs `level_factor`, `TsallisAnnealing::new(0.0, …)`
vs seeded offset), and `run(None, …)` made correct handler construction
impossible since the initial score is unknown up front.

## Background

- `GenericLocalSearchOptimizer<ST>` stores no handler field. `step<M, H>` and
  `optimize_with_handler<M, H>` take `handler: H` per call and return the
  handler alongside the result (`(StepResult, H)` / `(sol, score, H)`), so
  per-iteration state survives replica/member loops.
- `LocalSearchOptimizer<M>` (`src/optim/base.rs`) is handler-free:
  `optimize(...) -> (Solution, Score)`, `run` / `run_with_callback` unchanged
  from the pre-refactor shape (2-tuple results).
- 8 direct wrappers (Metropolis, SA, Tsallis, AdaptiveAnnealing, GreatDeluge,
  LogisticAnnealing, RelativeAnnealing, EpsilonGreedy) own their config and
  construct the matching handler inside `optimize`.
- `parallel_tempering.rs` / `population_annealing.rs` call `Generic…::step`
  per replica/member with `Metropolis::new(beta)` built from their own ladder.
- `TabuSearchOptimizer` keeps `optimize_with_handler` (caller-supplied list)
  plus `optimize_default`; the trait impl builds `T::default()` internally.
- `tests/test_transition_handler.rs` exercises handlers + `step` directly.
- 13 `impl LocalSearchOptimizer for ...` exist; `GenericLocalSearchOptimizer`
  is NOT one of them.

## Approach

### 1. `LocalSearchOptimizer` trait stays handler-free (`src/optim/base.rs`)

No change from the pre-refactor shape:

```rust
pub trait LocalSearchOptimizer<M: OptModel> {
    fn optimize(&self, model: &M, initial_solution: M::SolutionType,
        initial_score: M::ScoreType, n_iter: usize, time_limit: Duration,
        callback: &mut dyn OptCallbackFn<...>) -> (M::SolutionType, M::ScoreType);
    fn run_with_callback(...) -> Result<(M::SolutionType, M::ScoreType), ...> { ... }
    fn run(...) -> Result<...> {
        self.run_with_callback(..., &mut |_| {})
    }
}
```

### 2. `GenericLocalSearchOptimizer` takes handler per-call, no trait impl (`src/optim/generic.rs`)

- Struct stays `GenericLocalSearchOptimizer<ST>` — no `H` type param, no
  `handler` field. `new(patience, n_trials, return_iter)`.
- `step<M, H: TransitionHandler<ST>>(..., mut handler: H) -> (StepResult, H)`.
- `optimize_with_handler<M, H: TransitionHandler<ST>>(..., handler: H) -> (sol, score, H)`.
- NO `impl LocalSearchOptimizer for GenericLocalSearchOptimizer`.

### 3. Tabu keeps both paths (`src/optim/tabu_search.rs`)

- Keep `optimize_with_handler<M, H: TabuList>(..., mut tabu_list: H) -> (..., H)`.
- Keep `optimize_default` (builds `T::default()` + sets size, delegates).
- `impl<T: TabuList, M: OptModel<TransitionType = T::Item>> LocalSearchOptimizer<M>`:
  delegate to `optimize_default`, discard the returned list.

### 4. 8 wrappers build handlers internally

Each wrapper implements `LocalSearchOptimizer<M>` (handler-free) and discards
the handler returned by `optimize_with_handler`:

```rust
fn optimize(&self, ...) -> (M::SolutionType, M::ScoreType) {
    let handler = Metropolis::new(self.beta); // from OWN config
    let opt = GenericLocalSearchOptimizer::new(self.patience, self.n_trials, self.return_iter);
    let (sol, sc, _) = opt.optimize_with_handler(..., handler);
    (sol, sc)
}
```

- GreatDeluge seeds `GreatDeluge::new(initial_score * self.level_factor)` from
  the run's actual initial score — never caller-supplied.
- Tsallis seeds `TsallisAnnealing::new(initial_score, ...)` offset the same way.
- SA / Adaptive / EpsilonGreedy / Logistic / Relative build from their own
  fields (`initial_beta`, `scheduler`, `epsilon`, `w`, …). No dead config.

`RandomSearchOptimizer` and `HillClimbingOptimizer` delegate to
`EpsilonGreedyOptimizer` with fixed epsilon (1.0 / 0.0) and implement the
handler-free trait.

### 5. Parallel/population wrappers

- `parallel_tempering.rs`: per replica, `GenericLocalSearchOptimizer::new(...)`
  + `step(..., Metropolis::new(self.betas[idx]))`; unpack `(StepResult, _)`,
  discard handler. Trait impl is handler-free, returns 2-tuple.
- `population_annealing.rs`: same pattern with `Metropolis::new(current_beta)`.

### 6. Tests + examples

- `src/tests/test_*.rs`: handlerless `run(&model, None, …)` — no handler
  construction in tests (except `test_transition_handler.rs`, which tests
  handlers + `step` directly and is unchanged).
- `examples/tsp_model.rs`, `examples/quadratic_model.rs`: drop all handler
  construction + handler imports; `run` / `run_with_callback` without handler arg.

### 7. Re-exports (`src/optim.rs`)

Keep exporting `StepResult`, `TransitionHandler`, `UpdateCtx`, `TabuList`,
`GenericLocalSearchOptimizer`, `TabuSearchOptimizer`, and handler types (needed
by `step`/`optimize_with_handler` callers and handler unit tests).

## Trade-offs

- Reverts the per-call-handler trait design: callers lose the ability to inject
  a custom handler through `optimize`/`run`. Accepted because handler state
  that derives from run-time data (`initial_score`-seeded GDA level / Tsallis
  offset) cannot be constructed correctly by callers, especially under
  `run(None, …)`; and duplicated config (optimizer field + handler field)
  silently diverges.
- `GenericLocalSearchOptimizer` not implementing the trait means it can't be
  boxed as `dyn LocalSearchOptimizer` — fine, since only complete optimizers
  (with owned config) are trait objects; generic stays a building block.

## Open questions

None.

## Next step

Edit in order: `base.rs` (already handler-free — verify) → `generic.rs`
(drop trait impl) → wrappers (8, build handlers internally) → `random.rs` +
`hill_climbing.rs` (delegate, handler-free trait) → `parallel_tempering.rs` →
`population_annealing.rs` → `tabu_search.rs` (trait impl via
`optimize_default`) → tests + examples (drop handler args). Then
`cargo check -q` → `cargo test -q` → `cargo clippy -q --fix --allow-dirty` →
`cargo clippy -q -- -D warnings`.
