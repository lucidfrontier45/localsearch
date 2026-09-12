# Refactor: TransitionHandler trait for local-search optimizers

## Goal

Replace per-optimizer transition closures + `Rc<RefCell>` mutation with a unified `TransitionHandler` trait. All 10 local-search optimizers become thin wrappers around `GenericLocalSearchOptimizer<H>` + a per-algorithm `XxxHandler`.

## Background

- 10 optimizers currently wrap `GenericLocalSearchOptimizer` (8 direct; `HillClimbingOptimizer` + `RandomSearchOptimizer` wrap `EpsilonGreedyOptimizer`).
- Annealing variants share `metropolis::metropolis_transition`; `great_deluge` is deterministic; `tsallis` + `adaptive_annealing` reuse `AdaptiveScheduler`.
- Shared mutable state (β, `water_level`, `offset`) lives in `Rc<RefCell<T>>` + callback wrapper today.
- Examples (`examples/tsp_model.rs`) + tests (`src/tests/test_*.rs`) reference structs by name. Breaking refactor.
- AGENTS.md: integration tests in `src/tests/`; Edition 2024; no `mod.rs`.

## Approach

### Step 1 — Define core types in `src/optim/transition.rs` (new file)

```rust
pub struct UpdateCtx<'a, ST> {
    pub iter: usize,
    pub total: usize,
    pub acc: f64,
    pub best: &'a ST,
}

pub trait TransitionHandler<ST: Ord + Send + Sync + Copy>: Send + Sync {
    fn update(&mut self, ctx: &UpdateCtx<'_, ST>);
    fn evaluate(&self, current: ST, trial: ST) -> f64;
}
```

- `update` called once per iter (before trials); `evaluate` called per trial.
- Handler owns state directly. No `Rc<RefCell>`.

### Step 2 — Re-wire `GenericLocalSearchOptimizer<ST, H>` in `src/optim/generic.rs`

- Replace `FT: TransitionProbabilityFn` bound with `H: TransitionHandler<ST>`.
- Field: `handler: H`.
- Internal loop: `handler.update(&ctx)` → for each trial: `if rng < handler.evaluate(...) { accept }`.
- Add `pub fn step<M>(&self, model, ...) -> StepResult<...>` (same semantics as today's `MetropolisOptimizer::step`).
- Keep `optimize()` as `while time_left { step(); callback(progress); }`.

### Step 3 — Implement handlers in `src/optim/handlers/` (new dir)

| Handler | Owns | `evaluate` | `update` |
|---|---|---|---|
| `EpsilonGreedy` | `epsilon: f64` | `trial<current ? 1 : epsilon` | no-op |
| `Metropolis` | `beta: f64` | `exp(-β·Δ)` | no-op |
| `SimulatedAnnealing` | `beta, cooling_rate, update_freq, iter_counter` | metropolis with current β | when `iter % freq == 0 && iter > 0`: `β *= cooling_rate` |
| `AdaptiveAnnealing` | `beta, scheduler, iter_counter, freq` | metropolis with current β | when `iter % freq == 0 && iter > 0`: `β = scheduler.update_temperature(β, iter, total, acc)` |
| `RelativeAnnealing` | `beta` | `exp(-β·(Δ/\|cur\|))` | no-op |
| `LogisticAnnealing` | `w` | `2/(1+exp(w·(Δ/\|cur\|)))` | no-op |
| `GreatDeluge` | `initial_level, level_factor` | `trial<level ? 1 : 0` | `level = initial_level - (initial_level - best) · iter/total` |
| `Tsallis` | `offset, beta, q, xi, scheduler, iter_counter, freq` | `max([1-(1-q)βd]^{1/(1-q)}, 0.01)`, `d=(trial-offset)/(cur-offset+xi)` | `offset = best`; β update via scheduler when `iter % freq == 0 && iter > 0` |

`AdaptiveScheduler` + `TargetAccScheduleMode` live in `src/optim/handlers/adaptive_annealing.rs`.

### Step 4 — Public API reshape in `src/optim.rs`

- Export: `TransitionHandler`, `UpdateCtx`, all `*Handler` types, `StepResult`.
- Each top-level `XxxOptimizer` struct holds `GenericLocalSearchOptimizer<ST, XxxHandler>`. `new(...)` signature preserved.
- `LocalSearchOptimizer<M>` impl on each `XxxOptimizer` delegates to inner `optimize()`/`step()`.
- `MetropolisOptimizer::step` removed (logic moved into handler `update` + `GenericLocalSearchOptimizer::step`).

### Step 5 — Factory optimizers

- `HillClimbingOptimizer` = `EpsilonGreedyOptimizer::new(patience, n_trials, usize::MAX, 0.0)`.
- `RandomSearchOptimizer` = `EpsilonGreedyOptimizer::new(patience, 1, usize::MAX, 1.0)`.
- No `HillClimbing`/`Random` handler structs. Just typed constructors over `EpsilonGreedy`.

### Step 6 — Migrate callers

- `examples/tsp_model.rs`: no signature change expected; verify still compiles.
- `src/tests/test_*.rs`: update any direct `GenericLocalSearchOptimizer::new(..., closure)` to handler form. Add `src/tests/test_transition_handler.rs`:
  - Each handler's `evaluate` matches existing formula (snapshot against current implementations).
  - `update` mutates state correctly (β cool, level interp, offset tracking).
  - `StepResult.acceptance_counter.acceptance_ratio` flows through to `UpdateCtx.acc`.

### Step 7 — Delete dead code

- Remove `transition_prob` free fns + `Rc<RefCell<...>>` wrappers from `epsilon_greedy.rs`, `metropolis.rs`, `relative_annealing.rs`, `logistic_annealing.rs`, `simulated_annealing.rs`, `adaptive_annealing.rs`, `great_deluge.rs`, `tsallis.rs`.
- Remove `metropolis::metropolis_transition` (now inside handler).
- Keep `metropolis::tune_temperature`, `gather_energy_diffs`, `calculate_temperature_from_acceptance_prob` (still used by `tune_initial_temperature` builders + parallel/population).
- Keep `simulated_annealing::tune_cooling_rate` (still used by `population_annealing`).

### Step 8 — Validate per AGENTS.md workflow

1. `cargo check -q`
2. `cargo test -q`
3. `cargo clippy -q --fix --allow-dirty`
4. `cargo clippy -q -- -D warnings`

## Trade-offs

- **Single `step(current, trial, ctx) -> f64` method** — rejected: forces per-trial mutation; current designs mutate once per iter.
- **Additive (old + new API)** — rejected (user wants breaking).
- **Stateless handler, push β/level into `UpdateCtx`** — rejected: per-algorithm state is non-trivial.
- **Drop step from Generic** — rejected (user wants step).
- **`Box<dyn TransitionHandler>`** vs generic `<H>` — generic preferred (monomorphization + `Copy`); trait object only as fallback if type inference breaks.

## Open questions

None.

## Next step

Execute Step 1 → 8 in order. First concrete action: create `src/optim/transition.rs` with `UpdateCtx` + `TransitionHandler` trait.
