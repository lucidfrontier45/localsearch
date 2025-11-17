# State Variable Update Order Rule

This document outlines the unified order for updating state variables after the step function in all optimizers within the `src/optim/` directory. This order ensures consistency, logical flow, and minimizes dependencies between updates across different optimization algorithms.

## Unified Update Order

After calling the internal step function (e.g., `metropolis.step`, `generic.step`, etc.), update state variables in the following sequence:

1. **Update current solution and score** from the step result (e.g., `current_solution = step_result.last_solution; current_score = step_result.last_score;`).
2. **Update best solution and score** if the step result's best is an improvement (e.g., `if step_result.best_score < best_score { ... }`), and reset stagnation counter if improved.
3. **Update accepted counter and transitions** (e.g., accumulate accepted transitions and increment counter).
4. **Update stagnation counter** (increment it).
5. **Check and handle return to best** (if stagnation >= return_iter, reset current to best).
6. **Update algorithm-specific state** (e.g., temperature cooling, tabu list append, population resampling).
7. **Check patience** (if stagnation >= patience, break).
8. **Update time and iteration counters** (e.g., elapsed time checks, iter increments).
9. **Invoke callback** with progress.

## Rationale

- **Logical Flow**: Core state (current/best) is updated first, followed by counters, checks, and side effects.
- **Consistency**: All optimizers now follow the same pattern, reducing bugs from inconsistent ordering.
- **Dependencies**: Updates are sequenced to avoid using stale or partially updated values.
- **Preservation of Behavior**: Transition logging uses pre-update values where necessary (e.g., old current_score for accepted transitions).

## Implementation Notes

- For optimizers with their own loops (e.g., `generic.rs`), apply this order inside the iteration loop.
- For optimizers calling external steps (e.g., `simulated_annealing.rs`), apply after the step call.
- Ensure cloning is used where ownership moves occur to avoid borrow errors (e.g., in `tabu_search.rs`).</content>
<parameter name="filePath">state_update_order.md