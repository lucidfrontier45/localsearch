use std::error::Error;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use localsearch::{optim::HillClimbingOptimizer, OptModel, OptProgress};
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

type StateType = Vec<f64>;
type ScoreType = NotNan<f64>;

impl OptModel for QuadraticModel {
    type StateType = StateType;
    type TransitionType = ();
    type ScoreType = ScoreType;
    fn generate_random_state<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<Self::StateType, Box<dyn Error>> {
        let state = self.dist.sample_iter(rng).take(self.k).collect::<Vec<_>>();
        Ok(state)
    }

    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &Self::StateType,
        rng: &mut R,
        _current_score: Option<NotNan<f64>>,
    ) -> (Self::StateType, Self::TransitionType, NotNan<f64>) {
        let k = rng.gen_range(0..self.k);
        let v = self.dist.sample(rng);
        let mut new_state = current_state.clone();
        new_state[k] = v;
        let score = self.evaluate_state(&new_state);
        (new_state, (), score)
    }

    fn evaluate_state(&self, state: &Self::StateType) -> NotNan<f64> {
        let score = (0..self.k)
            .into_iter()
            .map(|i| (state[i] - self.centers[i]).powf(2.0))
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
    let patiance = 1000;
    let n_trials = 50;
    let opt = HillClimbingOptimizer::new(patiance, n_trials);
    let pb = create_pbar(n_iter as u64);
    let callback = |op: OptProgress<StateType, ScoreType>| {
        pb.set_message(format!("best score {:e}", op.score.into_inner()));
        pb.set_position(op.iter as u64);
    };

    let res = opt.optimize(&model, None, n_iter, Some(&callback));
    pb.finish();
    dbg!(res);
}
