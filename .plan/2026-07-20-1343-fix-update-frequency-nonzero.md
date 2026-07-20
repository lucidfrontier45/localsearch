# Fix #78 — `NonZero<usize>` for `update_frequency`

## Goal
Replace `update_frequency: usize` with `update_frequency: NonZero<usize>` in the 5 optimizers affected by #78, eliminating the panic / infinite-loop class of bugs at the type level.

## Background
Issue #78 catalogs 6 failure sites where `update_frequency == 0` either panics or hangs:

| File | Line | Expression | Result |
|------|------|------------|--------|
| `src/optim/simulated_annealing.rs` | 127 | `progress.iter % self.update_frequency` | panic (mod 0) |
| `src/optim/simulated_annealing.rs` | 92 | `n_iter / self.update_frequency` | panic (div 0) |
| `src/optim/adaptive_annealing.rs` | 210 | `progress.iter % self.update_frequency` | panic (mod 0) |
| `src/optim/population_annealing.rs` | 86 | `n_iter / self.update_frequency` | panic (div 0) |
| `src/optim/population_annealing.rs` | 181 | `iter += self.update_frequency` | infinite loop |
| `src/optim/parallel_tempering.rs` | 193 | `iter = iter.saturating_add(self.update_frequency)` | infinite loop |

`TsallisRelativeAnnealingOptimizer` (`src/optim/tsallis.rs:137`) guards at runtime with `update_frequency > 0 && ... % == 0` — redundant once the type guarantees non-zero. Simplify.

`MetropolisOptimizer::step(..., n_iter: usize, ...)` keeps `usize` — `NonZero<usize>` passes through via `.get()`.

## Approach

### 1. Optimizer source (5 files)

For each: add `std::num::NonZero` import, change field type, change `new` parameter type, convert at every arithmetic site with `.get()`.

- **`src/optim/simulated_annealing.rs`**
  - Field: `update_frequency: NonZero<usize>`
  - `new(..., update_frequency: NonZero<usize>, ...)`
  - `tune_cooling_rate`: `n_iter / self.update_frequency.get()`
  - Callback: `progress.iter % self.update_frequency.get() == 0`

- **`src/optim/adaptive_annealing.rs`**
  - Field + `new` param: `NonZero<usize>`
  - Callback: `progress.iter % self.update_frequency.get() == 0`

- **`src/optim/population_annealing.rs`**
  - Field + `new` param: `NonZero<usize>`
  - `tune_cooling_rate`: `n_iter / self.update_frequency.get()`
  - `metropolis.step(..., self.update_frequency.get(), ...)`
  - `iter += self.update_frequency.get();`
  - `return_stagnation_counter += self.update_frequency.get();`
  - `patience_stagnation_counter += self.update_frequency.get();`

- **`src/optim/parallel_tempering.rs`**
  - Field: `NonZero<usize>`
  - `new(..., update_frequency: NonZero<usize>, ...)`
  - `with_geometric_betas(..., update_frequency: NonZero<usize>, ...)`
  - `tune_temperature` forwards `self.update_frequency` by value (Copy) — no change
  - Inner: `let update_freq = self.update_frequency.get();` then pass `update_freq` to `metropolis.step`
  - `iter = iter.saturating_add(self.update_frequency.get());`
  - `return_stagnation_counter.saturating_add(self.update_frequency.get())`
  - `patience_stagnation_counter.saturating_add(self.update_frequency.get())`

- **`src/optim/tsallis.rs`**
  - Field + `new` param: `NonZero<usize>`
  - Drop `self.update_frequency > 0` runtime guard
  - Callback: `progress.iter % self.update_frequency.get() == 0`

### 2. Call sites — wrap literals

Use `NonZero::new(N).expect("update_frequency must be >= 1")` for known-good literals. (Standard pattern; `.expect` is safe because every literal here is a fixed non-zero constant in source.)

- `src/tests/test_simulated_annealing.rs`: literal `1`
- `src/tests/test_adaptive_annealing.rs`: literal `100`
- `src/tests/test_population_annealing.rs`: literal `100`
- `src/tests/test_parallel_tempering.rs`: literals `5` (both call sites)
- `src/tests/test_tsallis.rs`: literal `100`
- `examples/tsp_model.rs`: literals `100` (SA), `100` (Adaptive), `100` (Population), `10` (ParallelTempering), `100` (Tsallis)

### 3. Doc comments

Update each optimizer's `/// - update_frequency` doc line. Current wording says "number of steps after which ...". New wording notes the value is a non-zero `usize`. Keep terse — single sentence per field.

### 4. Verify (per AGENTS.md workflow)

1. `cargo check -q`
2. `cargo test -q`
3. `cargo clippy -q --fix --allow-dirty`
4. `cargo clippy -q -- -D warnings`

If any step fails, fix and re-run from step 1.

## Trade-offs

- **`Result<Self, Error>` return** — rejected: forces every caller into fallible ergonomics; NonZero gives identical safety at compile time.
- **Builder method `with_update_frequency(n) -> Option<Self>`** — rejected: pushes validation to the caller; NonZero makes `new` infallible and self-documenting.
- **Keep runtime `assert!(update_frequency >= 1)`** — rejected: dead weight once the type guarantees it.
- **`#[serde]` impact** — none; field has no serde derives. `#[derive(Clone, Copy)]` still works (`NonZero<usize>` is `Copy`).

## Open Questions

- `ParallelTemperingOptimizer::new` still panics on empty `betas`. Out of scope for #78; leave untouched unless asked.
- `metropolis.step(..., n_iter: usize, ...)` — NonZero could propagate here too, but `metropolis` is outside the bug scope; keep as-is.

## Next Step

Begin implementation at `src/optim/simulated_annealing.rs`. Verify after each file with `cargo check -q` to catch arithmetic-site oversights early. Then update tests + example in one batch and run the full AGENTS.md verification.

---

## Follow-up: review cleanup (concerns #3 + #4)

**Scope**: post-review of #78 surfaced two concerns in `src/optim/population_annealing.rs`. Concern #2 (boilerplate `.expect()`) intentionally left as-is per decision. No other files touched.

### F1 — Unify overflow policy to `saturating_add` (concern #3)

`parallel_tempering.rs` already uses `saturating_add` for iter/counter increments; `population_annealing.rs` uses plain `+=` (panics on overflow in debug, wraps in release). Make population match parallel.

Three sites in `population_annealing.rs` (current lines 181, 191, 192):
- `iter += self.update_frequency.get();` → `iter = iter.saturating_add(update_freq);`
- `return_stagnation_counter += self.update_frequency.get();` → `return_stagnation_counter = return_stagnation_counter.saturating_add(update_freq);`
- `patience_stagnation_counter += self.update_frequency.get();` → `patience_stagnation_counter = patience_stagnation_counter.saturating_add(update_freq);`

**Behavior change**: debug-build overflow panic → saturate at `usize::MAX`. Unreachable in practice (`iter` bounded by `n_iter`, counters reset on improvement). Net: consistent policy across sibling annealers, marginally safer.

### F2 — Hoist `.get()` once per loop iteration (concern #4)

Mirror `parallel_tempering.rs:169`. Insert `let update_freq = self.update_frequency.get();` at top of outer `while` loop body, before `population.par_iter()`. All four `.get()` sites then read the local:
- closure arg `metropolis.step(..., update_freq, ...)` (line 173)
- the three F1 increments (post-collect)

**Rationale**: one field read per outer iteration instead of four; the `par_iter` closure captures the `usize` copy rather than `self.update_frequency`.

### Verify (per AGENTS.md)

1. `cargo check -q`
2. `cargo test -q`
3. `cargo clippy -q --fix --allow-dirty`
4. `cargo clippy -q -- -D warnings`

### Trade-offs

- **`checked_add` + fallible loop** — rejected: `run` returns `Result`, but converting inner-loop counter math to fallible balloons the diff; `saturating_add` matches established parallel_tempering pattern.
- **Leave #3 plain `+=`** — rejected: user chose unification; consistency across sibling optimizers worth the 3-line change.

### Open questions

- `BUGS.md` (untracked) flags a *separate* `tune_cooling_rate` issue: when `n_iter < update_frequency`, integer division yields `0` → `cooling_rate = inf`, silently freezing SA. The #78 NonZero change prevents the div-by-zero *panic* but NOT this `0 → inf` path. Out of scope here; needs its own plan.

### Next step

Edit `src/optim/population_annealing.rs` lines ~169–192: hoist `update_freq`, convert three `+=` to `saturating_add`. Run 4-step verify.
