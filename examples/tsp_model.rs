use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead},
    path::Path,
    time::Duration,
};

use anyhow::Result as AnyResult;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use localsearch::{
    optim::{
        EpsilonGreedyOptimizer, HillClimbingOptimizer, LocalSearchOptimizer,
        RelativeAnnealingOptimizer, SimulatedAnnealingOptimizer, TabuList, TabuSearchOptimizer,
    },
    utils::RingBuffer,
    OptModel, OptProgress,
};
use ordered_float::NotNan;
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

type SolutionType = Vec<usize>;
// remvoed edges and inserted edges
type TransitionType = ([Edge; 2], [Edge; 2]);
type ScoreType = NotNan<f64>;

impl OptModel for TSPModel {
    type SolutionType = SolutionType;
    type TransitionType = TransitionType;
    type ScoreType = ScoreType;
    fn generate_random_solution<R: rand::Rng>(&self, rng: &mut R) -> AnyResult<SolutionType> {
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

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: &SolutionType,
        rng: &mut R,
        current_score: Option<NotNan<f64>>,
    ) -> (SolutionType, TransitionType, NotNan<f64>) {
        let (ind1, ind2) = select_two_indides(1, current_solution.len() - 1, rng);

        let mut new_solution = current_solution.clone();
        for (i, ind) in (ind1..=ind2).enumerate() {
            new_solution[ind] = current_solution[ind2 - i];
        }

        let removed_edges = [
            min_sorted(current_solution[ind1 - 1], current_solution[ind1]),
            min_sorted(current_solution[ind2 + 1], current_solution[ind2]),
        ];

        let inserted_edges = [
            min_sorted(new_solution[ind1 - 1], new_solution[ind1]),
            min_sorted(new_solution[ind2 + 1], new_solution[ind2]),
        ];

        // calculate new score
        let new_score = match current_score {
            Some(s) => {
                s - self.get_distance(&removed_edges[0], true)
                    - self.get_distance(&removed_edges[1], true)
                    + self.get_distance(&inserted_edges[0], true)
                    + self.get_distance(&inserted_edges[1], true)
            }
            None => self.evaluate_solution(&new_solution),
        };

        // create transition
        let trans = (removed_edges, inserted_edges);

        (new_solution, trans, new_score)
    }

    fn evaluate_solution(&self, solution: &SolutionType) -> NotNan<f64> {
        let score = (0..solution.len() - 1)
            .map(|i| {
                let key = min_sorted(solution[i], solution[i + 1]);
                self.get_distance(&key, true)
            })
            .sum();
        NotNan::new(score).unwrap()
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
    type Item = (SolutionType, TransitionType);

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

    let n_iter: usize = 20000;
    let time_limit = Duration::from_secs(60);
    let patience = n_iter / 2;

    let mut rng = rand::thread_rng();
    let initial_solution = tsp_model.generate_random_solution(&mut rng).ok();

    let pb = create_pbar(n_iter as u64);
    let callback = |op: OptProgress<SolutionType, ScoreType>| {
        let ratio = op.accepted_count as f64 / op.iter as f64;
        pb.set_message(format!(
            "best score {:.4e}, count = {}, acceptance ratio {:.2e}",
            op.score.into_inner(),
            op.accepted_count,
            ratio
        ));
        pb.set_position(op.iter as u64);
    };

    println!("run hill climbing");
    let optimizer = HillClimbingOptimizer::new(1000, 200);
    let (final_solution, final_score, _) = optimizer.run(
        &tsp_model,
        initial_solution.clone(),
        n_iter,
        time_limit,
        Some(&callback),
        (),
    );
    println!(
        "final score = {}, num of cities {}",
        final_score,
        final_solution.len()
    );
    pb.finish_and_clear();
    pb.reset();

    println!("run tabu search");
    let tabu_list = DequeTabuList::new(20);
    let optimizer = TabuSearchOptimizer::new(patience, 200, 10);
    let (final_solution, final_score, _) = optimizer.run(
        &tsp_model,
        initial_solution.clone(),
        n_iter,
        time_limit,
        Some(&callback),
        tabu_list,
    );
    println!(
        "final score = {}, num of cities {}",
        final_score,
        final_solution.len()
    );
    pb.finish_and_clear();
    pb.reset();

    println!("run annealing");
    let optimizer = SimulatedAnnealingOptimizer::new(patience, 200);
    let (final_solution, final_score, _) = optimizer.run(
        &tsp_model,
        initial_solution.clone(),
        n_iter,
        time_limit,
        Some(&callback),
        (200.0, 50.0),
    );
    println!(
        "final score = {}, num of cities {}",
        final_score,
        final_solution.len()
    );
    pb.finish_and_clear();
    pb.reset();

    println!("run epsilon greedy");
    let optimizer = EpsilonGreedyOptimizer::new(patience, 200, 10, 0.3);
    let (final_solution, final_score, _) = optimizer.run(
        &tsp_model,
        initial_solution.clone(),
        n_iter,
        time_limit,
        Some(&callback),
        (),
    );
    println!(
        "final score = {}, num of cities {}",
        final_score,
        final_solution.len()
    );
    pb.finish_and_clear();
    pb.reset();

    println!("run relative annealing");
    let optimizer = RelativeAnnealingOptimizer::new(patience, 200, 10, 1e1);
    let (final_solution, final_score, _) = optimizer.run(
        &tsp_model,
        initial_solution,
        n_iter,
        time_limit,
        Some(&callback),
        (),
    );
    println!(
        "final score = {}, num of cities {}",
        final_score,
        final_solution.len()
    );

    let opt_route_file = args.get(2).unwrap();
    let opt_solution = read_lines(opt_route_file)
        .unwrap()
        .into_iter()
        .map(|line| {
            let i: usize = line.unwrap().parse().unwrap();
            i
        })
        .collect::<Vec<_>>();

    let opt_score = tsp_model.evaluate_solution(&opt_solution);
    println!(
        "optimal score = {}, num of cities {}",
        opt_score,
        opt_solution.len()
    );
}
