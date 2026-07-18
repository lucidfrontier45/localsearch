# Fix zero-division in logistic + relative annealing

## Goal
Eliminate NaN/inf propagation in `transition_prob` for both optimizers via epsilon-clamped denominator, no public API change, with regression tests for `current == 0`.

## Background
- Both files use `d = (trial - current) / current.abs()`.
- `src/optim/relative_annealing.rs` documents IEEE-754 NaN behavior at line 22 ("no panic").
- Tests at `src/tests/test_relative_annealing.rs:18` already evaluate an objective returning `0.0`, so `current_score == 0` is reachable in practice.
- Logistic has a latent bug beyond NaN: at `current=0, trial<0` (improvement), `d = -inf`, and the sigmoid `2.0/(1.0 + exp(w*d)) = 2.0/(1.0 + 0.0) = 2.0` (correct), but at `current=0, trial>0` we get `2.0/(1.0+inf) = 0` (rejection of worsening — correct), and at `trial == 0` we get `d = NaN` → `p = NaN` → `NaN > rand == false` → reject. The inf path is correct; the NaN path silently makes `current=trial=0` indifferent.

## Approach
1. **`src/optim/relative_annealing.rs:9`** — replace denominator with `current.abs().max(f64::EPSILON)`. Keep existing comment, note epsilon floor.
2. **`src/optim/logistic_annealing.rs:9`** — same epsilon clamp. Add doc comment matching the relative style.
3. **Unit tests (both `mod test` blocks)** — add `current=0` cases asserting `is_finite()` and acceptance direction.
4. **Integration tests** — adapt existing objective (already returns 0) to start from `initial_score=0` so the inner loop actually triggers `current_score == 0`. Assert finite final score, no panic.
5. Run per AGENTS.md: `cargo check -q && cargo test -q && cargo clippy -q --fix --allow-dirty && cargo clippy -q -- -D warnings`.

## Trade-offs
- `f64::EPSILON ≈ 2.22e-16` — safe for integer/continuous objectives. Continuous scores << 1 may saturate; acceptable vs NaN.
- Fallback to absolute diff rejected (loses relative-comparison property).
- Keep IEEE-754 rejected (leaves latent logistic improvement-rejection risk + undocumented).

## Next step
Edit `src/optim/{relative,logistic}_annealing.rs`, extend unit tests, adapt integration tests, then `cargo test`.
