use std::error::Error;

use indicatif::{ProgressBar, ProgressStyle};
use localsearch::{
    optim::{callback::OptProgress, HillClimbingOptimizer, TabuList, TabuSearchOptimizer},
    utils::RingBuffer,
    OptModel,
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

type StateType = Vec<f64>;
type TransitionType = (usize, f64, f64);

impl OptModel for QuadraticModel {
    type StateType = StateType;
    type TransitionType = TransitionType;
    type ScoreType = NotNan<f64>;
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
        _current_score: Option<NotNan<f64>>,
    ) -> (StateType, TransitionType, NotNan<f64>) {
        let k = rng.gen_range(0..self.k);
        let v = self.dist.sample(rng);
        let mut new_state = current_state.clone();
        new_state[k] = v;
        let score = self.evaluate_state(&new_state);
        (new_state, (k, current_state[k], v), score)
    }

    fn evaluate_state(&self, state: &StateType) -> NotNan<f64> {
        let score = (0..self.k)
            .into_iter()
            .map(|i| (state[i] - self.centers[i]).powf(2.0))
            .sum();
        NotNan::new(score).unwrap()
    }
}

#[derive(Debug)]
struct DequeTabuList {
    buff: RingBuffer<TransitionType>,
}

impl DequeTabuList {
    fn new(size: usize) -> Self {
        let buff = RingBuffer::new(size);
        Self { buff }
    }
}

impl TabuList for DequeTabuList {
    type Item = (StateType, TransitionType);

    fn contains(&self, item: &Self::Item) -> bool {
        let (k1, _, x) = item.1;
        self.buff
            .iter()
            .any(|&(k2, y, _)| (k1 == k2) && (x - y).abs() < 0.0001)
    }

    fn append(&mut self, item: Self::Item) {
        self.buff.append(item.1);
    }
}

fn main() {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));

    println!("running Hill Climbing optimizer");
    let n_iter = 10000;
    let patiance = 1000;
    let n_trials = 50;
    let opt = HillClimbingOptimizer::new(patiance, n_trials);
    let pb = ProgressBar::new(n_iter as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (eta={eta}) {msg} ",
            )
            .progress_chars("#>-"),
    );
    let callback = |op: OptProgress<_>| {
        let pb = pb.clone();
        pb.set_message(format!("best score {:e}", op.score));
        pb.set_position(op.iter as u64);
    };

    let res = opt.optimize(&model, None, n_iter, Some(&callback));
    pb.finish_at_current_pos();
    dbg!(res);

    pb.finish_and_clear();
    pb.reset();
    println!("running Tabu Search optimizer");
    let opt = TabuSearchOptimizer::new(patiance, n_trials, 20);
    let tabu_list = DequeTabuList::new(2);

    let res = opt.optimize(&model, None, n_iter as usize, tabu_list, Some(&callback));
    pb.finish_at_current_pos();
    dbg!((res.0, res.1));
}
