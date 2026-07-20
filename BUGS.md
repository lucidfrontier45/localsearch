# Bug Analysis Report

Analysis of `localsearch` crate (v0.24.0). Scanned all 37 files under `src/`,
cross-referenced against `algorithm.md` and `state_update_order.md`.
All 16 existing tests pass; `cargo clippy -D warnings` clean.

Findings ranked by severity.

---

## 🔴 BUG 1 — Integer underflow panic in Parallel Tempering exchange loop

**File:** `src/optim/parallel_tempering.rs:236`

```rust
for i in 0..(n_replicas - 1) {
```

`n_replicas = self.betas.len()`. If user constructs the optimizer with an **empty**
`betas` vector (`ParallelTemperingOptimizer::new(.., vec![], ..)`), `n_replicas` is `0`
and `n_replicas - 1` underflows → **panic on usize subtraction overflow** (in debug;
wraps to `usize::MAX` in release, looping ~18 quintillion times).

The earlier code already assumes `n_replicas > 0` (line 215 divides by it, line 196
`min_by_key().unwrap()` would panic on empty `step_results`), but the subtraction is the
first hard fault and the easiest to trigger.

**Fix:** guard at construction or at top of `optimize`:
```rust
if self.betas.is_empty() {
    return (initial_solution, initial_score); // or return Err / panic with message
}
```

---

## 🟠 BUG 2 — Division-by-zero / `inf` cooling rate when `n_iter < update_frequency`

**Files:**
- `src/optim/simulated_annealing.rs:92`  (`tune_cooling_rate`)
- `src/optim/population_annealing.rs:86` (`tune_cooling_rate`)

```rust
let cooling_rate = tune_cooling_rate(self.initial_beta, 1e2, n_iter / self.update_frequency);
```

`n_iter / self.update_frequency` is **integer division**. When `n_iter < update_frequency`
the third argument is `0`, and inside `tune_cooling_rate` (`simulated_annealing.rs:20`):

```rust
(final_beta / initial_beta).powf(1.0 / n_iter as f64)  // 1.0 / 0.0 = +inf
```

`powf(+inf)` → `+inf`, so `cooling_rate = inf`. Every temperature update then sets
`beta = inf`, freezing the search instantly. Silently breaks SA instead of erroring.

**Fix:** clamp the denominator, e.g.
```rust
let steps = (n_iter / self.update_frequency).max(1);
tune_cooling_rate(self.initial_beta, 1e2, steps)
```
or make `tune_cooling_rate` itself reject/ guard `n_iter == 0`.

---

## 🟠 BUG 3 — Great Deluge acceptance uses strict `<` instead of documented `<=`

**File:** `src/optim/great_deluge.rs:71`

```rust
if trial.into_inner() < wl { 1.0 } else { 0.0 }
```

`algorithm.md` documents the rule as:

> a trial is accepted if its score is **less than or equal to** the current water level

Code uses strict `<`. A trial exactly at the water level is rejected, contradicting the
spec. With floating-point scores exact equality is rare but reachable (e.g. integer-valued
objectives, or when `level_factor = 1.0`).

**Fix:** `if trial.into_inner() <= wl { 1.0 } else { 0.0 }`

---

## 🟡 BUG 4 — Population Annealing Boltzmann weights can overflow to `+inf`

**File:** `src/optim/population_annealing.rs:229`

```rust
let boltzmann_factor = (-current_beta * score.into_inner()).exp().max(1e-8);
```

The `.max(1e-8)` guards underflow but **not overflow**. This is a minimization problem;
good solutions have *low* (often negative) scores. When `score` is very negative,
`-beta * score` → `+inf`, `exp(+inf)` → `+inf`. With one infinite weight,
`WeightedIndex` (line 240) panics or the normalized distribution collapses to a single
member — destroying population diversity silently.

Standard population annealing uses the **shifted** form `exp(-beta * (score - score_min))`
to keep the exponent ≤ 0.

**Fix:**
```rust
let min_score = new_population.iter().map(|(_, s)| s.into_inner()).fold(f64::INFINITY, f64::min);
// ...
let boltzmann_factor = (-current_beta * (score.into_inner() - min_score)).exp().max(1e-8);
```

---

## 🟡 BUG 5 — `min_by_key().unwrap()` panics on `n_trials == 0`

**File:** `src/optim/generic.rs:112-113`

```rust
let (trial_solution, trial_score) = (0..self.n_trials)
    .into_par_iter()
    .map(...)
    .min_by_key(|(_, score)| *score)
    .unwrap();   // panics if n_trials == 0
```

If a user passes `n_trials = 0` to any optimizer that delegates to
`GenericLocalSearchOptimizer` (SA, Metropolis, relative/logistic annealing, epsilon-greedy,
great deluge, hill climbing, random search), `min_by_key` returns `None` and `.unwrap()`
panics. Same pattern at `parallel_tempering.rs:196` and `population_annealing.rs:184`
(though those panic earlier on empty replicas/population).

**Fix:** validate `n_trials >= 1` in `GenericLocalSearchOptimizer::new` (return `Result`
or `panic!` with a clear message), or guard before the loop.

---

## 🟢 NOTE A — Replica-exchange loop swaps sequentially, not even/odd

**File:** `src/optim/parallel_tempering.rs:236-246`

The exchange loop iterates `i = 0,1,2,…` and swaps adjacent pairs in place. A swap at `i`
then affects the pair evaluated at `i+1` in the *same* sweep. Standard parallel tempering
alternates **even** (`(0,1),(2,3),…`) and **odd** (`(1,2),(3,4),…`) sublattices each round
to keep swaps independent and detailed balance clean.

This is a known simplification, not a crash, but it biases exchange dynamics vs. the
textbook algorithm. Worth documenting or fixing if statistical correctness matters.

---

## 🟢 NOTE B — `utils::RingBuffer` has no `contains` / `set_size`

**File:** `src/utils.rs`

`RingBuffer<T>` exposes only `new`, `append`, `iter`. It does **not** implement the
`TabuList` trait (`contains`, `set_size`), despite its doc comment *"RingBuffer to be used
to implement a Tabu List"*. Every TabuList user (`examples/tsp_model.rs`,
`src/tests/test_tabu_search.rs`) wraps it in a custom struct and re-implements those
methods. The doc is misleading; either implement `TabuList` for `RingBuffer` directly or
fix the comment.

---

## Summary

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🔴 High   | `parallel_tempering.rs:236` | `n_replicas - 1` underflow panic on empty `betas` |
| 2 | 🟠 Med    | `simulated_annealing.rs:92`, `population_annealing.rs:86` | `n_iter/update_frequency` int div → `inf` cooling rate |
| 3 | 🟠 Med    | `great_deluge.rs:71` | `<` vs `<=` contradicts documented acceptance rule |
| 4 | 🟠 Med    | `population_annealing.rs:229` | Boltzmann weight overflow → `inf`, breaks resampling |
| 5 | 🟡 Low    | `generic.rs:113` | `.unwrap()` panics when `n_trials == 0` |
| A | 🟢 Note   | `parallel_tempering.rs:236` | Sequential vs even/odd exchange sweep |
| B | 🟢 Note   | `utils.rs` | `RingBuffer` doc claims TabuList usage it doesn't implement |

No incorrectness found in the core acceptance math (Metropolis, Tsallis, relative/logistic
transition probabilities all verified correct for minimization). State update ordering
across all optimizers matches `state_update_order.md`.
