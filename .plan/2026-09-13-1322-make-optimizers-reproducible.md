# Plan: Make Optimizers Reproducible (#100)

- **Issue:** https://github.com/lucidfrontier45/localsearch/issues/100 (open, enhancement)
- **Status:** planned, not implemented
- **API impact:** non-breaking additions only

## Goal

Seeded, bit-reproducible runs for all optimizers via non-breaking `with_seed(u64)`
builder; eliminate every entropy-sourced RNG inside library code.

## Background

Issue text: "pass seed to optimize/optimize_with_callback; reduce stochastic
behaviour as much as possible."

Clarification decisions:

- API shape **B** — non-breaking builder `with_seed(u64)`; `Option<u64>` field,
  `None` = current entropy behavior.
- Public seed param type: `u64`.
- Tune functions: seeded too.
- Time-limit nondeterminism: doc caveat, accepted.
- Sub-seed derivation: reuse master-rng → sequential-fork pattern already
  proven in `search_loop.rs:260-264`.

### Entropy RNG sites to eliminate

| Site | Problem |
|---|---|
| `src/optim/base.rs:38` | `rand::rng()` for initial solution in `run_with_callback` |
| `src/optim/search_loop.rs:225` | master `rand::rng()` for whole shared loop |
| `src/optim/tabu_search.rs:151` | thread-local `rand::rng()` **inside rayon par_iter** — worst offender |
| `src/optim/parallel_tempering.rs:128` | master `rand::rng()` |
| `src/optim/population_annealing.rs:106` | master `rand::rng()` |
| `src/optim/handlers/metropolis.rs:44,51` | `gather_energy_diffs` warmup — feeds all tune fns |

`src/optim/alns.rs:465` is test-only — leave as-is.

Residual nondeterminism that seeding cannot fix (doc-only): wall-clock
`time_limit` cutoff varies iteration count; user models with internal rng.

## Approach

1. **Seed derivation util** (`src/optim/search_loop.rs`, `pub(crate)`):
   - `make_master_rng(seed: Option<u64>) -> StdRng` —
     `Some(s) => StdRng::seed_from_u64(s)`, `None => StdRng::from_rng(rand::rng())`
     (entropy fallback).
   - `fork_seed(master: &mut StdRng) -> u64` — draws sub-seed.
   - `derive_seed(seed: u64, salt: u64) -> u64` — splitmix64-finalizer domain
     separation. Salts: initial solution ≠ loop ≠ warmup, avoiding identical
     `seed_from_u64` streams for distinct phases.
2. **Trait `LocalSearchOptimizer`** (`base.rs`): add provided method
   `fn rng_seed(&self) -> Option<u64> { None }`. `run_with_callback` uses it for
   initial-solution generation (salt 1). Non-breaking; external impls default
   to entropy.
3. **`LocalSearchLoop`**: field `seed: Option<u64>` (`new()` = `None`), builder
   `pub const fn with_seed(mut self, seed: u64) -> Self`. `step_with_generator`
   master rng from field (salt 2). Existing sequential-fork logic untouched.
4. **All optimizers** (SA, adaptive, relative, logistic, tsallis, metropolis,
   hill climbing, epsilon greedy, great deluge, random search, tabu, PT, PA,
   `GenericLocalSearchOptimizer`): field `seed: Option<u64>`,
   `pub const fn with_seed(...)` builder, override `rng_seed()`, construct loop
   with seed.
   - Rebuild pitfall: PT `tune_temperature`/`with_geometric_betas` reconstruct
     via `Self::new(...)` — must carry `seed` field through;
     `GenericLocalSearchOptimizer::with_trial_generator` spread keeps it.
     Audit every `Self {`/`Self::new` rebuild.
   - `RandomSearchOptimizer` delegates to epsilon greedy — pass seed through.
5. **`tabu_search.rs`**: replace per-trial `rand::rng()` in par_iter with
   master-seeded sequential forks (search_loop pattern). Makes tabu results
   thread-count independent.
6. **`parallel_tempering.rs` / `population_annealing.rs`**: master rng from
   `make_master_rng`.
7. **Handlers** (`handlers/metropolis.rs`):
   - `gather_energy_diffs_with_seed(model, initial, n_warmup, seed: Option<u64>)`
     — internal par_iter gets sequential forks.
   - `tune_temperature_with_seed` likewise.
   - Old fns delegate with `None`. Non-breaking.
   - Handler methods in `handlers/simulated_annealing.rs` +
     `handlers/adaptive_annealing.rs` get `_with_seed` variants.
8. **Optimizer tune fns** (SA/adaptive/PA/PT): signatures unchanged —
   internally route `self.seed` into `_with_seed` free fns. Non-breaking.
9. **Docs**: rustdoc on `with_seed`/`rng_seed`; API.md reproducibility section:
   - set `time_limit` generously so `n_iter` governs → bit-reproducible;
   - user models must draw only from supplied rng;
   - thread-count independence via sequential forks.
10. **Tests** (`src/tests/test_reproducibility.rs` + tune test where model
    lives):
    - same seed → identical `(solution, score)` across two runs (SA, tabu, PT,
      PA, epsilon greedy on `QuadraticModel`);
    - `tune_temperature_with_seed` twice → identical beta;
    - rayon 1-thread vs N-thread via scoped `ThreadPool::install` → identical
      result.
11. **Workflow** (AGENTS.md order): `cargo check -q` → `cargo test -q` →
    `cargo clippy -q --fix --allow-dirty` → `cargo clippy -q -- -D warnings`.
    No new deps.

## Trade-offs

- **Seed param on `optimize`/`run` (rejected)** — breaking for all trait impls
  + callers; issue text suggested it, but option B chosen in clarification.
- **`Option<u64>` public param (rejected)** — `u64` in `with_seed`; `None`
  semantics live only in the private field.
- **Storing `StdRng` in optimizers (rejected)** — breaks `&self`/`Copy`/reuse
  semantics; seed field + per-call fork keeps optimizers stateless.
- **Salt-based domain separation vs shared stream (chosen salt)** — plain
  `seed_from_u64(seed)` for both initial solution and loop would reuse the
  identical stream twice; splitmix64 salt is 3 lines, kills the correlation.

## Open questions

- None blocking. Version bump (0.25.0?) left to maintainer — non-breaking
  feature.

## Next step

Step 1: add seed-derivation helpers in `search_loop.rs`, then thread
`LocalSearchLoop.seed` (steps 2–3).
