# TODO: 2026-07-20-1343-fix-update-frequency-nonzero

## Phase 1: NonZero migration (5 optimizers)
- [x] Read 5 optimizer files to confirm line numbers + arithmetic sites — already NonZero in all 5
- [x] Edit src/optim/simulated_annealing.rs — already migrated
- [x] Edit src/optim/adaptive_annealing.rs — already migrated
- [x] Edit src/optim/population_annealing.rs — already migrated
- [x] Edit src/optim/parallel_tempering.rs — already migrated
- [x] Edit src/optim/tsallis.rs — already migrated, runtime guard removed

## Phase 2: Call sites (5 tests + 1 example)
- [x] src/tests/test_simulated_annealing.rs — already wrapped
- [x] src/tests/test_adaptive_annealing.rs — already wrapped
- [x] src/tests/test_population_annealing.rs — already wrapped
- [x] src/tests/test_parallel_tempering.rs — already wrapped (2x)
- [x] src/tests/test_tsallis.rs — already wrapped
- [x] examples/tsp_model.rs — already wrapped (5x)

## Phase 3: Doc comments
- [x] Doc comments — already note "non-zero" in all 5 files

## Phase 4: Follow-up F1+F2 (population_annealing.rs only)
- [x] Hoist `let update_freq = self.update_frequency.get();` into outer loop body
- [x] Convert 3x `+=` to `saturating_add` (iter, return_stagnation, patience_stagnation)
- [x] Add `: usize` annotation to counter bindings (compiler ambiguity)

## Phase 5: Verify per AGENTS.md
- [x] cargo check -q — silent OK
- [x] cargo test -q — 21 passed
- [x] cargo clippy -q --fix --allow-dirty — silent OK
- [x] cargo clippy -q -- -D warnings — silent OK