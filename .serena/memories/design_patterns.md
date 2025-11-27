# Design Patterns and Architectural Patterns

## Trait-Based Architecture
- **Strategy Pattern**: The `LocalSearchOptimizer` trait allows different optimization algorithms to be used interchangeably
- **Template Method Pattern**: The base trait provides a default `run` and `run_with_callback` implementation that can be used by all optimizers

## Model-Algorithm Separation
- **Separation of Concerns**: The `OptModel` trait separates the problem domain (what to optimize) from the algorithm domain (how to optimize)
- Models implement the problem-specific logic while algorithms implement the optimization strategy

## Callback Pattern
- Progress monitoring and control through callback functions passed to optimization runs
- Allows for progress bars, logging, early termination, and other monitoring without changing core algorithm code

## State Management
- Consistent state update ordering as defined in `state_update_order.md`
- Centralized handling of time limits, iteration limits, patience, and stagnation
- Preprocessing and postprocessing hooks for solution manipulation

## Generic Programming
- Algorithms are generic over the model type, allowing the same algorithm to work with different optimization problems
- Type constraints ensure safety and performance (Sync, Send, Ord, etc.)

## Zero-Cost Abstractions
- Use of traits with static dispatch
- Generics for type safety without runtime overhead
- Optimized for performance while maintaining flexibility

## Parallelization Pattern
- Use of Rayon for parallelization of algorithms
- Thread-safe design with Sync and Send constraints
- Algorithms can operate on multiple data structures in parallel

## Builder Pattern
- Optimizer construction through constructor methods (e.g., `HillClimbingOptimizer::new`)
- Configuration of optimizer parameters at construction time