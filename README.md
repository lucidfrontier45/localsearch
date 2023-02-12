# localsearch
Rust library for local search optimization

# Implemented Algorithms

All of the algorithms are parallelized with Rayon.

1. Hill Climbing.
2. Tabu Search.
3. Simulated Annealing
4. Epsilon Greedy Search, a variant of Hill Climbing which accepts the trial state with a constant probabilith even if the score of the trial state is worse than the previous one.
5. Logistic Annealing, a variabt of Simulated Annealing which uses relative score diff to calculate transition probability.

# How to use

You need to implement your own model that implements `OptModel` trait. See examples for detail.