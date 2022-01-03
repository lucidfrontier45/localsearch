# localsearch
Rust library for local search optimization

# Implemented Algorithms

1. Hill Climbing with multi-restarts, parallelized by Rayon.
2. Tabu Search, parallelized by Rayon.

# How to use

You need to implement your own model that implements `OptModel` trait. See examples for detail.