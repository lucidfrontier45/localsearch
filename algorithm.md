# Overview of optimization algorithms in `src/optim`

This document summarizes the algorithms implemented under `src/optim`.
Each section names the optimizer, describes the core idea, the acceptance/transition probability, and important parameters or helper functions. File references point to the primary implementation locations.

## Base trait
  - `LocalSearchOptimizer` (`src/optim/base.rs`) — Defines the optimizer interface: `optimize` and `run/run_with_callback` helpers that handle generating an initial solution and pre/postprocessing.

## Generic Local Search
  - `GenericLocalSearchOptimizer<ST, H, G = DefaultTrialGenerator>` (`src/optim/generic.rs`) — Adapter that drives any `TransitionHandler` (plus an optional `TrialGenerator`) through the `LocalSearchOptimizer` interface on top of `LocalSearchLoop`.
  - Behavior:
    - Each iteration generates `n_trials` candidate solutions through the configured generator (`DefaultTrialGenerator` delegates to `model.generate_trial_solution`, parallelized with Rayon) and selects the best trial by score.
    - Accepts the winning trial with probability `p = handler.evaluate(current_score, trial_score)`, accepting if `p > rand(0,1)`. Improving trials return `1.0` from the handler itself.
    - Tracks `best_solution`, `return_iter` (periodically revert to best), and `patience` (early stop when stagnating).
    - Reports acceptance ratio via `AcceptanceCounter` and calls the provided callback with `OptProgress` each iteration.
    - Stores the handler/generator as blueprints and clones them per `optimize` call, so the optimizer is reusable across runs. Use `with_trial_generator` to plug in ALNS or another adaptive scheme.
  - Key files: `src/optim/generic.rs` (adapter), `src/optim/search_loop.rs` (step loop and acceptance logic).

## Metropolis
  - `MetropolisOptimizer` (`src/optim/metropolis.rs`) — Standard Metropolis algorithm with fixed inverse temperature `beta`.
  - Transition probability: `p = 1.0` if `trial <= current` else `p = exp(-beta * (trial - current))` implemented by `metropolis_probability` (`src/optim/handlers/metropolis.rs`, re-exported via `src/optim/handlers.rs`).
  - Helper: `tune_temperature` / `gather_energy_diffs` (`src/optim/handlers/metropolis.rs`) — estimate beta from warmup energy differences to target acceptance probability.

## Simulated Annealing (SA)
  - `SimulatedAnnealingOptimizer` (`src/optim/simulated_annealing.rs`) — Metropolis with time-varying temperature.
  - Uses Metropolis acceptance `exp(-beta * ΔE)` (same `metropolis_probability`) and multiplies inverse temperature `beta` by `cooling_rate` every `update_frequency` iterations (only when `iter > 0` is a multiple of it).
  - Helpers: `tune_initial_temperature` (via `handlers::metropolis::tune_temperature`) and `tune_cooling_rate(initial_beta, final_beta, n_iter)` (`src/optim/handlers/simulated_annealing.rs`).

## Adaptive Annealing
  - `AdaptiveAnnealingOptimizer` (`src/optim/adaptive_annealing.rs`) — Tries to adapt temperature to realize a scheduled target acceptance rate.
  - Scheduler `AdaptiveScheduler` (`src/optim/handlers/adaptive_annealing.rs`) supports `Linear`, `Exponential`, `Cosine` (default), and `Constant` target acceptance schedules. It updates `beta` using `beta *= exp(-gamma * (target_acc - acc)/target_acc)` every `update_frequency` iterations.
  - Can tune initial temperature via `tune_initial_temperature` which delegates to `handlers::metropolis::tune_temperature` (target probability taken from `scheduler.initial_target_acc`).

## Logistic Annealing
  - `LogisticAnnealingOptimizer` (`src/optim/logistic_annealing.rs`, handler in `src/optim/handlers/logistic_annealing.rs`) — Acceptance based on *relative* score difference using a logistic-like formula.
  - Transition probability: improving trials (`trial < current`) return `1.0`; otherwise `d = (trial - current) / |current|.max(EPSILON)`, `p = 2 / (1 + exp(w * d))`. Larger `w` makes the acceptance steeper.

## Relative Annealing
  - `RelativeAnnealingOptimizer` (`src/optim/relative_annealing.rs`, handler in `src/optim/handlers/relative_annealing.rs`) — Accepts using a relative-difference exponential: improving trials return `1.0`; otherwise `d = (trial - current)/|current|.max(EPSILON)`, `p = exp(-beta * d)`.

## Tsallis Relative Annealing
  - `TsallisRelativeAnnealingOptimizer` (`src/optim/tsallis.rs`, handler `TsallisAnnealing` in `src/optim/handlers/tsallis.rs`) — Generalizes relative annealing with Tsallis statistics (q-statistics).
  - Acceptance probability: `ΔE <= 0` returns `1.0`; otherwise with `d = ΔE / (current - offset + xi)`, `p = max([1 - (1-q) * beta * d]^{1/(1-q)}, 0.01)`.
  - Maintains a mutable offset tracking the best score (`update` sets `offset = ctx.best` each iteration; `optimize` seeds it from the initial score) and allows scheduling `beta` via an `AdaptiveScheduler` (optimizer default: `Constant(0.3)` with `gamma = 0.05`).

## Epsilon-Greedy
  - `EpsilonGreedyOptimizer` (`src/optim/epsilon_greedy.rs`, handler in `src/optim/handlers/epsilon_greedy.rs`) — Simple strategy: always accept improving moves; accept worsening moves with fixed probability `epsilon` (constructor clamps to `[0, 1]`).
  - Transition probability: `p = 1.0` if `trial < current` else `p = epsilon`.

## Hill Climbing
  - `HillClimbingOptimizer` (`src/optim/hill_climbing.rs`) — Deterministic greedy search implemented as `EpsilonGreedy` with `epsilon = 0.0` and `return_iter = usize::MAX` (effectively never revert to best).

## Random Search
  - `RandomSearchOptimizer` (`src/optim/random.rs`) — Repeatedly samples random trials and always accepts them (delegates to `EpsilonGreedy` with `epsilon = 1.0`, `n_trials = 1`, and `return_iter = usize::MAX`).

## Parallel Tempering (Replica Exchange)
  - `ParallelTemperingOptimizer` (`src/optim/parallel_tempering.rs`) — Runs multiple Metropolis replicas at different `betas` in parallel and occasionally attempts swaps between adjacent replicas.
  - Replica exchange acceptance between replicas i and j uses: `p_swap = exp((beta_j - beta_i) * (E_j - E_i))` and swaps if `p_swap >= 1` or with probability `p_swap` otherwise (`src/optim/parallel_tempering.rs`).
  - Provides helpers to tune a geometric ladder of betas (`with_geometric_betas`) or tune betas from warmup energy differences (`tune_temperature`) leveraging Metropolis warmup functions.

## Population Annealing
  - `PopulationAnnealingOptimizer` (`src/optim/population_annealing.rs`) — Maintains a population of candidate solutions, runs a batched simulated-annealing step on each member, then resamples the population according to Boltzmann weights.
  - After each population update the algorithm multiplies `beta` by the `cooling_rate`, computes weights `w_i = exp(-beta * score_i)`, normalizes them, and resamples the population with `WeightedIndex`.
  - Supports tuning initial beta (`tune_initial_temperature`) and tuning the cooling rate to reach a target final beta (`tune_cooling_rate`).
  - Boltzmann weights use a `1e-8` floor (`exp(-beta * score).max(1e-8)`) before normalizing for `WeightedIndex` resampling.

## Tabu Search
  - `TabuSearchOptimizer<T: TabuList>` (`src/optim/tabu_search.rs`) — Generates `n_trials` candidates, sorts them by score, then picks the first candidate that is either better than the current best (aspiration criterion) or not present in the tabu list.
  - Tabu mechanics:
    - Tabu list must implement `TabuList` trait (`src/optim/tabu_search.rs`) providing `contains`, `append`, and `set_size`.
    - When a transition is accepted its transition descriptor is appended to the tabu list to prevent recent moves being repeated.
    - If no acceptable candidate is found among samples, the iteration rejects and increases stagnation counters.
  - Entry points: `optimize_default` builds a fresh `T::default()` list sized by `default_tabu_size`; `optimize_with_handler` takes a caller-supplied list and returns it alongside the result.

## Great Deluge
  - `GreatDelugeOptimizer` (`src/optim/great_deluge.rs`, handler in `src/optim/handlers/great_deluge.rs`) — Implements the Great Deluge Algorithm (GDA), a threshold-driven local search.
  - Acceptance rule: improving trials (`trial < current`) are always accepted; otherwise a trial is accepted iff its score is strictly below the current water level (`trial < level`, deterministic 1/0 rather than a probability).
  - Water level update: the water level is initialized as `initial_score * level_factor` at the start of each `optimize` run and linearly interpolates toward the best-found score as `level = initial_level - (initial_level - best) * (iter / total)`.
  - Parameters:
    - `patience`: iterations without improvement before stopping
    - `n_trials`: number of candidate trials per iteration
    - `return_iter`: iterations without improvement before reverting to the best solution
    - `level_factor`: multiplier to set the initial water level above the initial score
  - Key file: `src/optim/great_deluge.rs`

Notes and common patterns

- Concrete optimizers each own their `TransitionHandler` as a blueprint and drive the shared `LocalSearchLoop::step` (Tabu, Parallel Tempering, and Population Annealing run their own outer loops around inner `LocalSearchLoop::step` calls). `GenericLocalSearchOptimizer` is the adapter for running an arbitrary handler through the standard interface, optionally with a custom `TrialGenerator` (e.g. ALNS).
- Several optimizers include helper tuning routines to set `beta` or cooling rates based on warmup sampling of energy differences: see `handlers::metropolis::gather_energy_diffs` and `tune_temperature` (`src/optim/handlers/metropolis.rs`) and `handlers::simulated_annealing::tune_cooling_rate` (`src/optim/handlers/simulated_annealing.rs`).
- Parallelism: candidate generation and many inner loops are parallelized with Rayon to speed up `n_trials` evaluations.