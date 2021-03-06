use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    io::{self, BufRead},
    path::Path,
};

use indicatif::{ProgressBar, ProgressStyle};
use localsearch::{
    optim::{HillClimbingOptimizer, TabuList, TabuSearchOptimizer},
    utils::RingBuffer,
    OptModel,
};
use rand::seq::SliceRandom;

fn min_sorted(c1: usize, c2: usize) -> (usize, usize) {
    if c1 < c2 {
        (c1, c2)
    } else {
        (c2, c1)
    }
}

type Edge = (usize, usize);

#[derive(Clone, Debug)]
struct TSPModel {
    start: usize,
    distance_matrix: HashMap<Edge, f64>,
}

impl TSPModel {
    fn new(start: usize, distance_matrix: HashMap<Edge, f64>) -> Self {
        Self {
            start,
            distance_matrix,
        }
    }

    fn from_coords(coords: &[(usize, f64, f64)]) -> TSPModel {
        let start = coords.iter().map(|(i, _, _)| *i).min().unwrap();
        let mut mat = HashMap::new();
        for &(c1, x1, y1) in coords {
            for &(c2, x2, y2) in coords {
                if c1 == c2 {
                    continue;
                }

                let key = min_sorted(c1, c2);
                if mat.contains_key(&key) {
                    continue;
                }

                let dist = ((x1 - x2).powf(2.0) + (y1 - y2).powf(2.0)).sqrt();
                mat.insert(key, dist);
            }
        }
        TSPModel::new(start, mat)
    }

    fn get_distance(&self, key: &(usize, usize), sorted: bool) -> f64 {
        if sorted {
            self.distance_matrix[key]
        } else {
            let key = min_sorted(key.0, key.1);
            self.distance_matrix[&key]
        }
    }
}

type StateType = Vec<usize>;

fn select_two_indides<R: rand::Rng>(lb: usize, ub: usize, rng: &mut R) -> (usize, usize) {
    let n1 = rng.gen_range(lb..ub);
    let n2 = loop {
        let n_ = rng.gen_range(lb..ub);
        if n_ != n1 {
            break n_;
        }
    };
    min_sorted(n1, n2)
}

// remvoed edges and inserted edges
type TransitionType = ([Edge; 2], [Edge; 2]);

impl OptModel<StateType, TransitionType> for TSPModel {
    fn generate_random_state<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<StateType, Box<dyn Error>> {
        let mut cities = self
            .distance_matrix
            .keys()
            .copied()
            .into_iter()
            .flat_map(|(i, j)| [i, j])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        cities.shuffle(rng);

        // ensure start city is located in the 0th
        let i = cities.iter().position(|&c| c == self.start).unwrap();
        cities.swap(0, i);

        // append start city to the last
        cities.push(self.start);

        Ok(cities)
    }

    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &StateType,
        rng: &mut R,
        current_score: Option<f64>,
    ) -> (StateType, TransitionType, f64) {
        let (ind1, ind2) = select_two_indides(1, current_state.len() - 1, rng);

        let mut new_state = current_state.clone();
        for (i, ind) in (ind1..=ind2).enumerate() {
            new_state[ind] = current_state[ind2 - i];
        }

        let removed_edges = [
            min_sorted(current_state[ind1 - 1], current_state[ind1]),
            min_sorted(current_state[ind2 + 1], current_state[ind2]),
        ];

        let inserted_edges = [
            min_sorted(new_state[ind1 - 1], new_state[ind1]),
            min_sorted(new_state[ind2 + 1], new_state[ind2]),
        ];

        // calculate new score
        let new_score = match current_score {
            Some(s) => {
                s - self.get_distance(&removed_edges[0], true)
                    - self.get_distance(&removed_edges[1], true)
                    + self.get_distance(&inserted_edges[0], true)
                    + self.get_distance(&inserted_edges[1], true)
            }
            None => self.evaluate_state(&new_state),
        };

        // create transition
        let trans = (removed_edges, inserted_edges);

        (new_state, trans, new_score)
    }

    fn evaluate_state(&self, state: &StateType) -> f64 {
        (0..state.len() - 1)
            .map(|i| {
                let key = min_sorted(state[i], state[i + 1]);
                self.get_distance(&key, true)
            })
            .sum()
    }
}

#[derive(Debug)]
struct DequeTabuList {
    buff: RingBuffer<Edge>,
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
        let (_, (_, inserted_edges)) = item;
        inserted_edges
            .iter()
            .any(|edge| self.buff.iter().any(|e| *e == *edge))
    }

    fn append(&mut self, item: Self::Item) {
        let (_, (removed_edges, _)) = item;
        for edge in removed_edges {
            if self.buff.iter().all(|e| *e != edge) {
                self.buff.append(edge);
            }
        }
    }
}

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn create_pbar(n_iter: u64) -> ProgressBar {
    let pb = ProgressBar::new(n_iter as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (eta={eta}) {msg} ",
            )
            .progress_chars("#>-"),
    );
    pb
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let input_file = args.get(1).unwrap();
    let coords = read_lines(input_file)
        .unwrap()
        .into_iter()
        .map(|line| {
            let line = line.unwrap();
            let splt = line.split(' ').collect::<Vec<_>>();
            let id: usize = splt[0].parse().unwrap();
            let x: f64 = splt[1].parse().unwrap();
            let y: f64 = splt[2].parse().unwrap();
            (id, x, y)
        })
        .collect::<Vec<_>>();

    let tsp_model = TSPModel::from_coords(&coords);

    let n_iter: usize = 100000;

    let pb = create_pbar(n_iter as u64);
    let callback = |it, _state, score| {
        let pb = pb.clone();
        pb.set_message(format!("best score {:e}", score));
        pb.set_position(it as u64);
    };

    println!("run hill climbing");
    let optimizer = HillClimbingOptimizer::new(2000, 200);
    let (_, final_score) = optimizer.optimize(&tsp_model, None, n_iter, Some(&callback));
    pb.finish_at_current_pos();
    println!("final score = {}", final_score);

    pb.finish_and_clear();
    pb.reset();
    println!("run tabu search");
    let tabu_list = DequeTabuList::new(20);
    let optimizer = TabuSearchOptimizer::new(2000, 200, 10);
    let (final_state, final_score, _) =
        optimizer.optimize(&tsp_model, None, n_iter, tabu_list, Some(&callback));
    pb.finish_at_current_pos();
    println!(
        "final score = {}, num of cities {}",
        final_score,
        final_state.len()
    );

    let opt_route_file = args.get(2).unwrap();
    let opt_state = read_lines(opt_route_file)
        .unwrap()
        .into_iter()
        .map(|line| {
            let i: usize = line.unwrap().parse().unwrap();
            i
        })
        .collect::<Vec<_>>();

    let opt_score = tsp_model.evaluate_state(&opt_state);
    println!(
        "optimal score = {}, num of cities {}",
        opt_score,
        opt_state.len()
    );
}
