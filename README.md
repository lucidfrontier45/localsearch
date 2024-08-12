# localsearch
Rust library for local search optimization

# Implemented Algorithms

All of the algorithms are parallelized with Rayon.

1. Hill Climbing.
2. Tabu Search.
3. Simulated Annealing
4. Epsilon Greedy Search, a variant of Hill Climbing which accepts the trial solution with a constant probabilith even if the score of the trial solution is worse than the previous one.
5. Relative Annealing, a variant of Simulated Annealing which uses relative score diff to calculate transition probability.
6. Logistic Annealing, a variant of Relative Annealing which uses logistic function instead of simple exponential.

# How to use

You need to implement your own model that implements `OptModel` trait. Actual optimization is handled by each algorithm functions. Here is a simple example to optimize a quadratic function with Hill Climbing algorithm.

```rust
use std::{time::Duration};

use anyhow::Result as AnyResult;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use localsearch::{
    optim::{HillClimbingOptimizer, LocalSearchOptimizer},
    OptModel, OptProgress,
};
use ordered_float::NotNan;
use rand::{self, distributions::Uniform, prelude::Distribution};

#[derive(Clone)]
struct QuadraticModel {
    k: usize,
    centers: Vec<f64>,
    dist: Uniform<f64>,
}

impl QuadraticModel {
    fn new(k: usize, centers: Vec<f64>, value_range: (f64, f64)) -> Self {
        let (low, high) = value_range;
        let dist = Uniform::new(low, high);
        Self { k, centers, dist }
    }
}

type SolutionType = Vec<f64>;
type ScoreType = NotNan<f64>;

impl OptModel for QuadraticModel {
    type SolutionType = SolutionType;
    type TransitionType = ();
    type ScoreType = ScoreType;
    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> AnyResult<Self::SolutionType> {
        let solution = self.dist.sample_iter(rng).take(self.k).collect::<Vec<_>>();
        Ok(solution)
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: &Self::SolutionType,
        rng: &mut R,
        _current_score: Option<NotNan<f64>>,
    ) -> (Self::SolutionType, Self::TransitionType, NotNan<f64>) {
        let k = rng.gen_range(0..self.k);
        let v = self.dist.sample(rng);
        let mut new_solution = current_solution.clone();
        new_solution[k] = v;
        let score = self.evaluate_solution(&new_solution);
        (new_solution, (), score)
    }

    fn evaluate_solution(&self, solution: &Self::SolutionType) -> NotNan<f64> {
        let score = (0..self.k)
            .into_iter()
            .map(|i| (solution[i] - self.centers[i]).powf(2.0))
            .sum();
        NotNan::new(score).unwrap()
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
    let time_limit = Duration::from_secs(60);
    let patiance = 1000;
    let n_trials = 50;
    let opt = HillClimbingOptimizer::new(patiance, n_trials);
    let pb = create_pbar(n_iter as u64);
    let callback = |op: OptProgress<SolutionType, ScoreType>| {
        pb.set_message(format!("best score {:e}", op.score.into_inner()));
        pb.set_position(op.iter as u64);
    };

    let res = opt.run(&model, None, n_iter, time_limit, Some(&callback), ());
    pb.finish();
    dbg!(res);
}

```

In addition you can also add `preprocess_initial_solution` and `postprocess_final_solution` to your model.
`preprocess_initial_solution` is called before start of optimization iteration.
If initial solution is not supplied, `generate_initial_solution` is called and the generate solution is then passed to `preprocess_initial_solution`.
`postprocess_final_solution` is called after the optimization iteration.


Further details can be found at API document, example and test codes.
