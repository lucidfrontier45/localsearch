# Overview of optimization algorithms in `src/optim`

This document summarizes the algorithms implemented under `src/optim`.
Each section names the optimizer, describes the core idea, the acceptance/transition probability, and important parameters or helper functions. File references point to the primary implementation locations.

## Base trait
  - `LocalSearchOptimizer` (`src/optim/base.rs`) — Defines the optimizer interface: `optimize` and `run/run_with_callback` helpers that handle generating an initial solution and pre/postprocessing.

## Generic Local Search
  - `GenericLocalSearchOptimizer<ST, FT>` (`src/optim/generic.rs`) — Core local-search engine used by many concrete optimizers.
  - Behavior:
    - Each iteration generates `n_trials` candidate solutions via `model.generate_trial_solution` (parallelized with Rayon) and selects the best trial by score.
    - Accepts a trial if it improves the current score. Otherwise, uses a provided transition probability function `FT: Fn(current_score, trial_score) -> f64` to compute acceptance probability `p`, then accepts if `p > rand(0,1)`.
    - Tracks `best_solution`, `return_iter` (periodically revert to best), and `patience` (early stop when stagnating).
    - Reports acceptance ratio via `AcceptanceCounter` and calls the provided callback with `OptProgress` each iteration.
  - Key file: `src/optim/generic.rs` (step loop and acceptance logic).

## Metropolis
  - `MetropolisOptimizer` (`src/optim/metropolis.rs`) — Standard Metropolis algorithm with fixed inverse temperature `beta`.
  - Transition probability: `p = 1.0` if `trial <= current` else `p = exp(-beta * (trial - current))` implemented by `metropolis_transition` (`src/optim/metropolis.rs`).
  - Helper: `tune_temperature` / `gather_energy_diffs` (`src/optim/metropolis.rs`) — estimate beta from warmup energy differences to target acceptance probability.

## Simulated Annealing (SA)
  - `SimulatedAnnealingOptimizer` (`src/optim/simulated_annealing.rs`) — Metropolis with time-varying temperature.
  - Uses Metropolis acceptance `exp(-beta * ΔE)` (same `metropolis_transition`) and updates inverse temperature `beta` multiplicatively by a `cooling_rate` every `update_frequency` iterations.
  - Helpers: `tune_initial_temperature` (via `metropolis::tune_temperature`) and `tune_cooling_rate(initial_beta, final_beta, n_iter)` (`src/optim/simulated_annealing.rs`).

## Adaptive Annealing
  - `AdaptiveAnnealingOptimizer` (`src/optim/adaptive_annealing.rs`) — Tries to adapt temperature to realize a scheduled target acceptance rate.
  - Scheduler `AdaptiveScheduler` (`src/optim/adaptive_annealing.rs`) supports `Linear`, `Exponential`, `Cosine`, and `Constant` target acceptance schedules. It updates `beta` using `beta *= exp(-gamma * (target_acc - acc)/target_acc)`.
  - Can tune initial temperature via `tune_initial_temperature` which delegates to `metropolis::tune_temperature`.

## Logistic Annealing
  - `LogisticAnnealingOptimizer` (`src/optim/logistic_annealing.rs`) — Acceptance based on *relative* score difference using a logistic-like formula.
  - Transition probability: `d = (trial - current) / current`, `p = 2 / (1 + exp(w * d))` (`src/optim/logistic_annealing.rs`). Larger `w` makes the acceptance steeper.

## Relative Annealing
  - `RelativeAnnealingOptimizer` (`src/optim/relative_annealing.rs`) — Accepts using a relative-difference exponential: `d = (trial - current)/current`, `p = exp(-beta * d)` (`src/optim/relative_annealing.rs`).

## Tsallis Relative Annealing
  - `TsallisRelativeAnnealingOptimizer` (`src/optim/tsallis.rs`) — Generalizes relative annealing with Tsallis statistics (q-statistics).
  - Acceptance probability for worse solutions (ΔE > 0): `p = [1 - (1-q) * beta * ΔE / (E - offset + xi)]^{1/(1-q)}` clamped to a minimum (implemented in `tsallis_transition_prob` in `src/optim/tsallis.rs`).
  - Maintains a mutable offset (current best score) and allows scheduling `beta` via an `AdaptiveScheduler`.

## Epsilon-Greedy
  - `EpsilonGreedyOptimizer` (`src/optim/epsilon_greedy.rs`) — Simple strategy: always accept improving moves; accept worsening moves with fixed probability `epsilon`.
  - Transition probability: `p = 1.0` if `trial < current` else `p = epsilon` (`src/optim/epsilon_greedy.rs`).

## Hill Climbing
  - `HillClimbingOptimizer` (`src/optim/hill_climbing.rs`) — Deterministic greedy search implemented as `EpsilonGreedy` with `epsilon = 0.0` and `return_iter = ∞`.

## Random Search
  - `RandomSearchOptimizer` (`src/optim/random.rs`) — Repeatedly samples random trials and always accepts them (wraps `EpsilonGreedy` with `epsilon = 1.0` and `n_trials = 1`).

## Parallel Tempering (Replica Exchange)
  - `ParallelTemperingOptimizer` (`src/optim/parallel_tempering.rs`) — Runs multiple Metropolis replicas at different `betas` in parallel and occasionally attempts swaps between adjacent replicas.
  - Replica exchange acceptance between replicas i and j uses: `p_swap = exp((beta_j - beta_i) * (E_j - E_i))` and swaps if `p_swap >= 1` or with probability `p_swap` otherwise (`src/optim/parallel_tempering.rs`).
  - Provides helpers to tune a geometric ladder of betas (`with_geometric_betas`) or tune betas from warmup energy differences (`tune_temperature`) leveraging Metropolis warmup functions.

## Population Annealing
  - `PopulationAnnealingOptimizer` (`src/optim/population_annealing.rs`) — Maintains a population of candidate solutions, runs a batched simulated-annealing step on each member, then resamples the population according to Boltzmann weights.
  - After each population update the algorithm multiplies `beta` by the `cooling_rate`, computes weights `w_i = exp(-beta * score_i)`, normalizes them, and resamples the population with `WeightedIndex`.
  - Supports tuning initial beta (`tune_initial_temperature`) and tuning the cooling rate to reach a target final beta (`tune_cooling_rate`).

## Tabu Search
  - `TabuSearchOptimizer<T: TabuList>` (`src/optim/tabu_search.rs`) — Generates `n_trials` candidates, sorts them by score, then picks the first candidate that is either better than the current best (aspiration criterion) or not present in the tabu list.
  - Tabu mechanics:
    - Tabu list must implement `TabuList` trait (`src/optim/tabu_search.rs`) providing `contains`, `append`, and `set_size`.
    - When a transition is accepted its transition descriptor is appended to the tabu list to prevent recent moves being repeated.
    - If no acceptable candidate is found among samples, the iteration rejects and increases stagnation counters.

Notes and common patterns

- Many optimizers are thin wrappers around `GenericLocalSearchOptimizer` with different `score_func` (transition probability) or with `Metropolis`-style transitions that depend on a `beta` parameter.
- Several optimizers include helper tuning routines to set `beta` or cooling rates based on warmup sampling of energy differences: see `metropolis::gather_energy_diffs` and `tune_temperature` (`src/optim/metropolis.rs` and `src/optim/metropolis.rs`) and `simulated_annealing::tune_cooling_rate` (`src/optim/simulated_annealing.rs`).
- Parallelism: candidate generation and many inner loops are parallelized with Rayon to speed up `n_trials` evaluations.