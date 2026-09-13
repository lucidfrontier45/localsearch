# State Variable Update Order Rule

This document outlines the unified order for updating state variables after the step function in all optimizers within the `src/optim/` directory. This order ensures consistency, logical flow, and minimizes dependencies between updates across different optimization algorithms.

## Unified Update Order

After each iteration of the shared trial-and-accept loop (`LocalSearchLoop::step` / `step_with_generator` in `src/optim/search_loop.rs`), update state variables in the following sequence. Optimizers with their own outer loops (`tabu_search.rs`, `parallel_tempering.rs`, `population_annealing.rs`) apply the same order around their inner steps:

1. **Check the time budget** (elapsed vs `time_limit`; break if exceeded) and note the iteration index.
2. **Refresh proposal state before trials** via `handler.update(ctx)` with `ctx = { iter, total, acc, best }` (cooling schedules, water level, Tsallis offset, ...).
3. **Generate trials and keep the winner** (`n_trials` candidates in parallel via the `TrialGenerator`; best by score).
4. **Update best solution and score** if the winner improves on it (`if trial_score < best_score { ... }`), resetting both stagnation counters; otherwise increment them. Classify the `TrialOutcome` (`NewBest` / `Improved` / `Accepted` / `Rejected`) against the pre-iteration best and feed it back to the generator.
5. **Accept or reject** (`p = handler.evaluate(current, trial)`; accept iff `p > rand(0, 1)`) and enqueue the result in the sliding-window `AcceptanceCounter`.
6. **Update current solution and score** from the accepted trial (on reject, keep the current values).
7. **Check and handle return to best** (if return-stagnation reaches `return_iter`, reset current to best and clear that counter).
8. **Check patience** (if patience-stagnation reaches `patience`, break).
9. (optional) **Update outer-loop algorithm state** (e.g., tabu list append on accept, replica exchange, population resampling with `beta *= cooling_rate`).
10. **Invoke callback** with `OptProgress { iter, acceptance_ratio, best_solution, best_score }`.

## Rationale

- **Logical Flow**: Core state (current/best) is updated first, followed by counters, checks, and side effects.
- **Consistency**: All optimizers now follow the same pattern, reducing bugs from inconsistent ordering.
- **Dependencies**: Updates are sequenced to avoid using stale or partially updated values.
- **Preservation of Behavior**: outcome classification uses pre-update values where necessary (e.g., `NewBest` is judged against the pre-iteration best, and acceptance is judged against the pre-update current score).

## Implementation Notes

- `src/optim/search_loop.rs` is the reference implementation of this order; keep it in sync when changing the loop.
- Outer-loop optimizers (`parallel_tempering.rs`, `population_annealing.rs`) aggregate per-replica/member `StepResult`s and then apply the same steps, advancing stagnation counters by `update_frequency` and comparing with `>=`; `tabu_search.rs` appends the accepted transition to the tabu list at step 9.
- Ensure cloning is used where ownership moves occur to avoid borrow errors (e.g., in `tabu_search.rs`).