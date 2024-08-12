use anyhow::Result as AnyResult;
use ordered_float::NotNan;
use rand::{distributions::Uniform, prelude::Distribution};

use crate::OptModel;

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
type TransitionType = (usize, f64, f64);

impl OptModel for QuadraticModel {
    type SolutionType = SolutionType;
    type TransitionType = TransitionType;
    type ScoreType = NotNan<f64>;
    fn generate_random_solution<R: rand::Rng>(&self, rng: &mut R) -> AnyResult<Self::SolutionType> {
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
        (new_solution, (k, current_solution[k], v), score)
    }

    fn evaluate_solution(&self, solution: &Self::SolutionType) -> NotNan<f64> {
        let score = (0..self.k)
            .into_iter()
            .map(|i| (solution[i] - self.centers[i]).powf(2.0))
            .sum();
        NotNan::new(score).unwrap()
    }
}

mod test_epsilon_greedy;
mod test_hill_climbing;
mod test_logistic_annealing;
mod test_relative_annealing;
mod test_simulated_annealing;
mod test_tabu_search;
