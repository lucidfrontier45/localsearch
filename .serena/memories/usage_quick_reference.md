# Usage Quick Reference

## Implementing Your Own Model
1. Define your solution, score, and transition types
2. Create a struct that implements the `OptModel` trait
3. Implement the required methods:
   - `generate_random_solution()` - to create a random starting point
   - `generate_trial_solution()` - to create new candidate solutions
4. Optionally implement `preprocess_solution()` and `postprocess_solution()`

## Using an Optimizer
1. Create an instance of your model
2. Choose an optimizer (e.g., `HillClimbingOptimizer`, `SimulatedAnnealingOptimizer`, etc.)
3. Call `run()` or `run_with_callback()` with your parameters
4. Handle the result

## Available Optimizers
- `HillClimbingOptimizer::new(patience, n_trials)`
- `SimulatedAnnealingOptimizer::new(...)` - parameters vary by algorithm
- `EpsilonGreedyOptimizer::new(...)`
- `TabuSearchOptimizer::new(...)` - requires tabu list implementation
- And others in the `optim` module

## Common Parameters
- `n_iter`: maximum number of iterations
- `time_limit`: maximum time for optimization (std::time::Duration)
- `patience`: iterations to wait before giving up on improvement
- `callback`: function called after each iteration with progress info

## Example Structure
```rust
use localsearch::{OptModel, optim::{HillClimbingOptimizer, LocalSearchOptimizer}};

// Implement OptModel for your problem
struct MyModel { ... }
impl OptModel for MyModel { ... }

// Create model and optimizer
let model = MyModel::new(...);
let optimizer = HillClimbingOptimizer::new(patience, n_trials);

// Run optimization
let result = optimizer.run(&model, initial_solution, n_iter, time_limit);
```