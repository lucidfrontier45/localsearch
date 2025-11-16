# localsearch
Rust library for local search optimization

# Implemented Algorithms

All of the algorithms are parallelized with Rayon.

1. Hill Climbing.
2. Tabu Search. To use this optimizer you also need to implement your problem specific tabu list.
3. Simulated Annealing
4. Epsilon Greedy Search, a variant of Hill Climbing which accepts the trial solution with a constant probability even if the score of the trial solution is worse than the previous one.
5. Relative Annealing, a variant of Simulated Annealing which uses relative score diff to calculate transition probability.
6. Logistic Annealing, a variant of Relative Annealing which uses logistic function instead of simple exponential.

# How to use

```toml
[dependencies]
localsearch = "0.18.1"
```

You need to implement your own model that implements `OptModel` trait. Actual optimization is handled by each algorithm functions. Here is a simple example to optimize a quadratic function with Hill Climbing algorithm.

```rust
use std::time::Duration;

use anyhow::Result as AnyResult;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use localsearch::{
    optim::{HillClimbingOptimizer, LocalSearchOptimizer},
    OptModel, OptProgress,
};
use ordered_float::NotNan;
use rand::{self, distr::Uniform, prelude::Distribution};

type SolutionType = Vec<f64>;
type ScoreType = NotNan<f64>;

#[derive(Clone)]
struct QuadraticModel {
    k: usize,
    centers: Vec<f64>,
    dist: Uniform<f64>,
}

impl QuadraticModel {
    fn new(k: usize, centers: Vec<f64>, value_range: (f64, f64)) -> Self {
        let (low, high) = value_range;
        let dist = Uniform::new(low, high).unwrap();
        Self { k, centers, dist }
    }

    fn evaluate_solution(&self, solution: &SolutionType) -> NotNan<f64> {
        let score = (0..self.k)
            .map(|i| (solution[i] - self.centers[i]).powf(2.0))
            .sum();
        NotNan::new(score).unwrap()
    }
}

impl OptModel for QuadraticModel {
    type SolutionType = SolutionType;
    type TransitionType = ();
    type ScoreType = ScoreType;
    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> AnyResult<(Self::SolutionType, Self::ScoreType)> {
        let solution = self.dist.sample_iter(rng).take(self.k).collect::<Vec<_>>();
        let score = self.evaluate_solution(&solution);
        Ok((solution, score))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: Self::SolutionType,
        _current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, NotNan<f64>) {
        let k = rng.random_range(0..self.k);
        let v = self.dist.sample(rng);
        let mut new_solution = current_solution;
        new_solution[k] = v;
        let score = self.evaluate_solution(&new_solution);
        (new_solution, (), score)
    }
}

fn create_pbar(n_iter: u64) -> ProgressBar {
    let pb = ProgressBar::new(n_iter);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (eta={eta}) {msg} ",
            ).unwrap()
            .progress_chars("#>-")
    );
    pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
    pb
}

fn main() {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));

    println!("running Hill Climbing optimizer");
    let n_iter = 10000;
    let time_limit = Duration::from_secs_f32(1.0);
    let patience = 1000;
    let n_trials = 50;
    let opt = HillClimbingOptimizer::new(patience, n_trials);
    let pb = create_pbar(n_iter as u64);
    let mut callback = |op: OptProgress<SolutionType, ScoreType>| {
        pb.set_message(format!("best score {:e}", op.score.into_inner()));
        pb.set_position(op.iter as u64);
    };

    let res = opt.run_with_callback(&model, None, n_iter, time_limit, &mut callback);
    pb.finish();
    dbg!(res.unwrap());
}
```

In addition you can also add `preprocess_solution` and `postprocess_solution` to your model.
`preprocess_solution` is called before start of optimization iteration.
If initial solution is not supplied, `generate_initial_solution` is called and the generated solution is then passed to `preprocess_solution`.
`postprocess_solution` is called after the optimization iteration.


Further details can be found at API document, example and test codes.