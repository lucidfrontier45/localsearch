# OptModel Batch Trial Generation Design

## Goal

Add overridable batch trial generation to `OptModel`. Route all built-in model-based trial generation through the batch API while preserving ALNS's custom generation path.

## Background

`OptModel` currently requires `generate_trial_solution`, and several optimization paths call it repeatedly. Those call sites implement their own parallelization or remain sequential. This prevents models from using more efficient native batch algorithms.

The new API centralizes batching in `OptModel::generate_trial_solutions`. Existing models retain their single-trial implementation and receive parallel batch behavior by default. Batch-oriented models can override the method with a more efficient implementation.

## API design

`generate_trial_solution` remains mandatory:

```rust
fn generate_trial_solution<R: rand::Rng>(
    &self,
    current_solution: Self::SolutionType,
    current_score: Self::ScoreType,
    rng: &mut R,
) -> (
    Self::SolutionType,
    Self::TransitionType,
    Self::ScoreType,
);
```

Add an overridable batch method:

```rust
fn generate_trial_solutions<R: rand::Rng + Send>(
    &self,
    current_solution: Self::SolutionType,
    current_score: Self::ScoreType,
    rngs: &mut [R],
) -> Vec<(
    Self::SolutionType,
    Self::TransitionType,
    Self::ScoreType,
)>;
```

Default implementation uses Rayon over `rngs.par_iter_mut()`, clones `current_solution` for each candidate, and delegates to `generate_trial_solution`. Indexed parallel collection preserves RNG/input order in returned vector. Empty RNG slices return an empty vector.

### Batch-native models

A model with efficient native batching may override `generate_trial_solutions` and provide a dummy mandatory `generate_trial_solution`. Built-in optimizers using `DefaultTrialGenerator` call only the batch method, so they do not reach the dummy implementation.

Prefer an explicit panic over fabricated output:

```rust
fn generate_trial_solution<R: rand::Rng>(
    &self,
    _current_solution: Self::SolutionType,
    _current_score: Self::ScoreType,
    _rng: &mut R,
) -> (
    Self::SolutionType,
    Self::TransitionType,
    Self::ScoreType,
) {
    panic!("single-trial generation is unsupported; use generate_trial_solutions")
}
```

This is safe only when every consumer uses `generate_trial_solutions`. External callers invoking the single method directly will panic. Documentation must state this limitation.

ALNS uses `AlnsTrialGenerator` rather than model trial-generation methods. Its operator-based generation and feedback remain unchanged, so a dummy model single method is also acceptable when that model is used exclusively through ALNS.

## Approach

1. Extend `OptModel` with `generate_trial_solutions` and its Rayon-backed default implementation.
2. Keep `generate_trial_solution` mandatory; do not add strategy traits or mutually recursive defaults.
3. Update `DefaultTrialGenerator` and the shared optimization loop so default generation invokes `model.generate_trial_solutions` once per iteration.
4. Generate per-trial seeds sequentially from the master RNG, construct `StdRng`s in seed order, and pass the mutable RNG slice to the batch method.
5. Preserve current minimum-score selection, tie behavior, winner feedback, acceptance behavior, and seeded reproducibility.
6. Keep ALNS on `AlnsTrialGenerator::generate_trial` so operator tokens and adaptive feedback retain current semantics.
7. Replace repeated model single-trial calls with batch calls in all non-ALNS paths, including:
   - population annealing initialization;
   - tabu search candidate generation;
   - Metropolis temperature tuning;
   - any other model-based multi-trial call sites found during implementation.
8. Use a one-element RNG slice for non-ALNS one-off generation where enforcing batch-only built-in access is necessary.
9. Update API docs and optimizer comments to describe ordering, parallelism, override behavior, and dummy single implementations.

## Tests

1. Verify default batch implementation calls `generate_trial_solution` once per RNG.
2. Verify batch results preserve RNG/input order.
3. Verify an empty RNG slice returns an empty vector.
4. Add a batch-native model whose single method panics; prove optimization with `DefaultTrialGenerator` succeeds.
5. Verify a custom batch override is called once per default-loop iteration.
6. Verify seeded optimization remains reproducible.
7. Verify ALNS retains its custom generation and winner-feedback path.
8. Verify all non-ALNS built-in multi-trial paths use the batch method.

## Trade-offs

- Mandatory single generation keeps the trait straightforward and preserves compatibility for existing models.
- Batch-native models still need a dummy single method because Rust cannot directly require either one of two trait methods without more complex strategy types.
- Dummy implementations weaken direct-call safety. Clear documentation and exclusive batch use in built-in default generation mitigate this risk.
- `R: Send` is required because the default implementation mutates independent RNGs across Rayon workers.
- Native batch overrides can avoid cloning, improve locality, or use domain-specific vectorization.

## Scope boundaries

In scope:

- `OptModel` batch API;
- default Rayon fallback;
- non-ALNS built-in migration;
- deterministic RNG behavior;
- tests and API documentation.

Out of scope:

- changing ALNS operator selection or feedback;
- adding dependencies;
- changing acceptance or scoring semantics;
- guaranteeing safety for external direct calls to a documented dummy single implementation.

## Validation

Run in order, restarting from step 1 after any failure:

1. `cargo check -q`
2. `cargo test -q`
3. `cargo clippy -q --fix --allow-dirty`
4. `cargo clippy -q -- -D warnings`

## Next step

Implement `OptModel::generate_trial_solutions`, then migrate default and non-ALNS generation paths.