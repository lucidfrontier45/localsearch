use std::error::Error;

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

type StateType = Vec<f64>;
type TransitionType = (usize, f64, f64);

impl OptModel<StateType, TransitionType> for QuadraticModel {
    fn generate_random_state<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<StateType, Box<dyn Error>> {
        let state = self.dist.sample_iter(rng).take(self.k).collect::<Vec<_>>();
        Ok(state)
    }

    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &StateType,
        rng: &mut R,
    ) -> (StateType, TransitionType) {
        let k = rng.gen_range(0..self.k);
        let v = self.dist.sample(rng);
        let mut new_state = current_state.clone();
        new_state[k] = v;
        (new_state, (k, current_state[k], v))
    }

    fn evaluate_state(&self, state: &StateType) -> f64 {
        (0..self.k)
            .into_iter()
            .map(|i| (state[i] - self.centers[i]).powf(2.0))
            .sum()
    }
}

#[cfg(test)]
mod test_hill_climbing;
