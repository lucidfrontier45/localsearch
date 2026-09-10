use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead},
    num::NonZero,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use localsearch::{
    optim::{
        AdaptiveAnnealingOptimizer, AlnsOperatorModel, AlnsOptimizer, EpsilonGreedyOptimizer,
        GreatDelugeOptimizer, HillClimbingOptimizer, LocalSearchOptimizer,
        ParallelTemperingOptimizer, PopulationAnnealingOptimizer, RelativeAnnealingOptimizer,
        SimulatedAnnealingOptimizer, TabuList, TabuSearchOptimizer,
        TsallisRelativeAnnealingOptimizer,
    },
    utils::RingBuffer,
    LocalsearchError, OptModel, OptProgress,
};
use ordered_float::NotNan;
use rand::{seq::SliceRandom, RngExt as _};

fn min_sorted(c1: usize, c2: usize) -> (usize, usize) {
    if c1 < c2 {
        (c1, c2)
    } else {
        (c2, c1)
    }
}

type Edge = (usize, usize);
type SolutionType = Vec<usize>;
// remvoed edges and inserted edges
type TransitionType = ([Edge; 2], [Edge; 2]);
type ScoreType = NotNan<f64>;

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

    fn evaluate_solution(&self, solution: &SolutionType) -> ScoreType {
        let score = (0..solution.len() - 1)
            .map(|i| {
                let key = min_sorted(solution[i], solution[i + 1]);
                self.get_distance(&key, true)
            })
            .sum();
        NotNan::new(score).unwrap()
    }
}

fn select_two_indices<R: rand::Rng>(lb: usize, ub: usize, rng: &mut R) -> (usize, usize) {
    let n1 = rng.random_range(lb..ub);
    let n2 = loop {
        let n_ = rng.random_range(lb..ub);
        if n_ != n1 {
            break n_;
        }
    };
    min_sorted(n1, n2)
}

impl OptModel for TSPModel {
    type SolutionType = SolutionType;
    type TransitionType = TransitionType;
    type ScoreType = ScoreType;
    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        let mut cities = self
            .distance_matrix
            .keys()
            .copied()
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

        let score = self.evaluate_solution(&cities);

        Ok((cities, score))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        let (ind1, ind2) = select_two_indices(1, current_solution.len() - 1, rng);

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
        let new_score = current_score
            - self.get_distance(&removed_edges[0], true)
            - self.get_distance(&removed_edges[1], true)
            + self.get_distance(&inserted_edges[0], true)
            + self.get_distance(&inserted_edges[1], true);

        // create transition
        let trans = (removed_edges, inserted_edges);

        (new_solution, trans, NotNan::new(new_score).unwrap())
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

impl Default for DequeTabuList {
    fn default() -> Self {
        Self::new(10)
    }
}

impl TabuList for DequeTabuList {
    type Item = TransitionType;
    fn contains(&self, transition: &TransitionType) -> bool {
        let (_, inserted_edges) = transition;
        inserted_edges
            .iter()
            .any(|edge| self.buff.iter().any(|e| *e == *edge))
    }

    fn append(&mut self, transition: TransitionType) {
        let (removed_edges, _) = transition;
        for edge in removed_edges {
            if self.buff.iter().all(|e| *e != edge) {
                self.buff.append(edge);
            }
        }
    }

    fn set_size(&mut self, n: usize) {
        self.buff = RingBuffer::new(n);
    }
}

const OPERATOR_NAMES: [&str; 3] = ["segment_reverse", "random_ruin_repair", "worst_ruin_repair"];
const N_OPERATORS: usize = OPERATOR_NAMES.len();
const MIN_REMOVAL: usize = 2;
const MAX_REMOVAL: usize = 12;

#[derive(Debug)]
struct AlnsTspModel {
    tsp: TSPModel,
    weights: Mutex<Vec<f64>>,
    uses: Vec<AtomicUsize>,
    improvements: Vec<AtomicUsize>,
}

impl AlnsTspModel {
    fn new(tsp: TSPModel) -> Self {
        Self {
            tsp,
            weights: Mutex::new(vec![1.0; N_OPERATORS]),
            uses: (0..N_OPERATORS).map(|_| AtomicUsize::new(0)).collect(),
            improvements: (0..N_OPERATORS).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    fn select_operator<R: rand::Rng>(&self, rng: &mut R) -> usize {
        let weights = self.weights.lock().unwrap();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return rng.random_range(0..N_OPERATORS);
        }
        let mut x = rng.random_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            x -= *w;
            if x <= 0.0 {
                return i;
            }
        }
        N_OPERATORS - 1
    }

    fn removal_count<R: rand::Rng>(tour_len: usize, rng: &mut R) -> usize {
        let max_removal = (tour_len - 2)
            .saturating_sub(2)
            .clamp(MIN_REMOVAL, MAX_REMOVAL);
        rng.random_range(MIN_REMOVAL..=max_removal)
    }

    fn select_random_positions<R: rand::Rng>(&self, tour_len: usize, rng: &mut R) -> Vec<usize> {
        let m = Self::removal_count(tour_len, rng);
        rand::seq::index::sample(rng, tour_len - 2, m)
            .into_iter()
            .map(|i| i + 1)
            .collect()
    }

    fn select_worst_positions<R: rand::Rng>(&self, tour: &[usize], rng: &mut R) -> Vec<usize> {
        let m = Self::removal_count(tour.len(), rng);
        let mut edges = (0..tour.len() - 1)
            .map(|i| {
                (
                    i,
                    self.tsp
                        .get_distance(&min_sorted(tour[i], tour[i + 1]), true),
                )
            })
            .collect::<Vec<_>>();
        edges.sort_by(|(_, d1), (_, d2)| d2.total_cmp(d1));

        let mut positions = Vec::with_capacity(m);
        for &(i, _) in &edges {
            for p in [i, i + 1] {
                if p > 0 && p + 1 < tour.len() && !positions.contains(&p) {
                    positions.push(p);
                }
                if positions.len() == m {
                    return positions;
                }
            }
        }
        positions
    }

    fn remove_cities(tour: &[usize], positions: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let removed = positions.iter().map(|&p| tour[p]).collect::<Vec<_>>();
        let removed_set = positions.iter().copied().collect::<HashSet<_>>();
        let partial = tour
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| (!removed_set.contains(&i)).then_some(c))
            .collect::<Vec<_>>();
        (partial, removed)
    }

    fn greedy_repair<R: rand::Rng>(
        &self,
        tour: &mut Vec<usize>,
        mut removed: Vec<usize>,
        rng: &mut R,
    ) {
        removed.shuffle(rng);
        for city in removed {
            let (best_pos, _) = tour
                .windows(2)
                .enumerate()
                .map(|(i, w)| {
                    let delta = self.tsp.get_distance(&min_sorted(w[0], city), false)
                        + self.tsp.get_distance(&min_sorted(city, w[1]), false)
                        - self.tsp.get_distance(&min_sorted(w[0], w[1]), false);
                    (i + 1, delta)
                })
                .min_by(|(_, d1), (_, d2)| d1.total_cmp(d2))
                .unwrap();
            tour.insert(best_pos, city);
        }
    }

    fn ruin_and_recreate<R: rand::Rng>(
        &self,
        current_solution: &SolutionType,
        worst: bool,
        rng: &mut R,
    ) -> (SolutionType, ScoreType) {
        let positions = if worst {
            self.select_worst_positions(current_solution, rng)
        } else {
            self.select_random_positions(current_solution.len(), rng)
        };
        let (mut tour, removed) = Self::remove_cities(current_solution, &positions);
        self.greedy_repair(&mut tour, removed, rng);
        let score = self.tsp.evaluate_solution(&tour);
        (tour, score)
    }
}

impl OptModel for AlnsTspModel {
    type SolutionType = SolutionType;
    type TransitionType = usize;
    type ScoreType = ScoreType;

    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        self.tsp.generate_random_solution(rng)
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        let operator = self.select_operator(rng);
        self.uses[operator].fetch_add(1, Ordering::Relaxed);

        let (new_solution, new_score) = if operator == 0 || current_solution.len() < 6 {
            let (solution, _, score) =
                self.tsp
                    .generate_trial_solution(current_solution, current_score, rng);
            (solution, score)
        } else {
            self.ruin_and_recreate(&current_solution, operator == 2, rng)
        };

        if new_score < current_score {
            self.improvements[operator].fetch_add(1, Ordering::Relaxed);
        }

        (new_solution, operator, new_score)
    }
}

impl AlnsOperatorModel for AlnsTspModel {
    fn n_operators(&self) -> usize {
        N_OPERATORS
    }

    fn set_operator_weights(&self, weights: &[f64]) {
        *self.weights.lock().unwrap() = weights.to_vec();
    }

    fn drain_operator_stats(&self) -> (Vec<usize>, Vec<usize>) {
        let uses = self
            .uses
            .iter()
            .map(|counter| counter.swap(0, Ordering::Relaxed))
            .collect();
        let improvements = self
            .improvements
            .iter()
            .map(|counter| counter.swap(0, Ordering::Relaxed))
            .collect();
        (uses, improvements)
    }
}

fn run_alns(
    tsp_model: &TSPModel,
    initial_solution: Option<(SolutionType, ScoreType)>,
    n_iter: usize,
    time_limit: Duration,
) -> (SolutionType, ScoreType) {
    let model = AlnsTspModel::new(tsp_model.clone());
    let initial_solution = match initial_solution {
        Some(solution) => solution,
        None => tsp_model
            .generate_random_solution(&mut rand::rng())
            .unwrap(),
    };

    let segment_len = (n_iter / 20).max(1);
    let return_iter = (n_iter / 50).max(1);
    let temperature = 0.02 * initial_solution.1.into_inner();

    let mut optimizer = AlnsOptimizer::new(
        segment_len,
        16,
        return_iter,
        move |current: ScoreType, trial: ScoreType| {
            let delta = (trial - current).into_inner();
            (-delta / temperature).exp()
        },
        N_OPERATORS,
        0.3,
    );

    let pb = create_pbar(n_iter as u64);
    let start_time = Instant::now();
    let mut best = initial_solution.clone();
    let mut current = initial_solution;
    let mut total_uses = vec![0usize; N_OPERATORS];
    let mut done_iter = 0usize;

    while done_iter < n_iter {
        let remaining = time_limit.saturating_sub(start_time.elapsed());
        if remaining.is_zero() {
            break;
        }

        let mut callback = |op: OptProgress<SolutionType, ScoreType>| {
            pb.set_message(format!(
                "best score {:.4e}, acceptance ratio {:.2}",
                op.score.into_inner(),
                op.acceptance_ratio
            ));
            pb.set_position((done_iter + op.iter) as u64);
        };

        let result = optimizer.run_segment(
            &model,
            current.0.clone(),
            current.1,
            best.1,
            segment_len,
            remaining,
            &mut callback,
        );
        done_iter += segment_len;

        if result.best_score < best.1 {
            best = (result.best_solution, result.best_score);
        }
        current = (result.last_solution, result.last_score);

        for (total, used) in total_uses.iter_mut().zip(&result.output.operator_uses) {
            *total += used;
        }

        pb.set_position(done_iter as u64);
        pb.set_message(format!(
            "best score {:.4e}, weights ({})",
            best.1.into_inner(),
            optimizer
                .weights()
                .iter()
                .map(|w| format!("{w:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    pb.finish_and_clear();

    println!(
        "operator uses: {}",
        OPERATOR_NAMES
            .iter()
            .zip(&total_uses)
            .map(|(name, uses)| format!("{name}={uses}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    best
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
    pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(20));
    pb
}

const ALL_OPTIMIZER_NAMES: [&str; 11] = [
    "AdaptiveAnnealingOptimizer",
    "AlnsOptimizer",
    "EpsilonGreedyOptimizer",
    "GreatDelugeOptimizer",
    "HillClimbingOptimizer",
    "ParallelTemperingOptimizer",
    "PopulationAnnealingOptimizer",
    "RelativeAnnealingOptimizer",
    "SimulatedAnnealingOptimizer",
    "TabuSearchOptimizer",
    "TsallisRelativeAnnealingOptimizer",
];

fn print_usage() {
    println!(
        "usage: tsp_model [--optimizer <name>] <coord_file> [optimal_route_file]\n\n\
         optimizers:\n  {}",
        ALL_OPTIMIZER_NAMES.join("\n  ")
    );
}

fn main() {
    let mut positional = Vec::<String>::new();
    let mut optimizer_name: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = if let Some(name) = arg.strip_prefix("--optimizer=") {
            Some(name.to_string())
        } else if arg == "--optimizer" {
            Some(args.next().unwrap_or_else(|| {
                eprintln!("error: --optimizer requires a value");
                std::process::exit(2);
            }))
        } else if arg == "-h" || arg == "--help" {
            print_usage();
            return;
        } else {
            positional.push(arg);
            None
        };
        if let Some(name) = value {
            if optimizer_name.is_some() {
                eprintln!("error: --optimizer specified more than once");
                std::process::exit(2);
            }
            optimizer_name = Some(name);
        }
    }

    if let Some(name) = &optimizer_name {
        if !ALL_OPTIMIZER_NAMES.contains(&name.as_str()) {
            eprintln!("error: unknown optimizer '{name}'");
            eprintln!("available optimizers: {}", ALL_OPTIMIZER_NAMES.join(", "));
            std::process::exit(2);
        }
    }

    let input_file = positional.first().unwrap_or_else(|| {
        print_usage();
        std::process::exit(2);
    });
    let coords = read_lines(input_file)
        .unwrap()
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
    let return_iter = n_iter / 50;
    let time_limit = Duration::from_secs(60);
    let patience = n_iter / 2;

    let mut rng = rand::rng();
    let initial_solution = tsp_model.generate_random_solution(&mut rng).ok();

    let pb = create_pbar(n_iter as u64);
    let mut callback = |op: OptProgress<SolutionType, ScoreType>| {
        // eprintln!("iter {}, score {}", op.iter, op.score);
        pb.set_message(format!(
            "best score {:.4e}, acceptance ratio {:.2}",
            op.score.into_inner(),
            op.acceptance_ratio
        ));
        pb.set_position(op.iter as u64);
    };

    let optimizers: Vec<(&str, Box<dyn LocalSearchOptimizer<TSPModel>>)> = vec![
        (
            "GreatDelugeOptimizer",
            Box::new(GreatDelugeOptimizer::new(patience, 16, return_iter, 1.05)),
        ),
        (
            "HillClimbingOptimizer",
            Box::new(HillClimbingOptimizer::new(patience, 16)),
        ),
        (
            "SimulatedAnnealingOptimizer",
            Box::new(
                SimulatedAnnealingOptimizer::new(
                    patience,
                    16,
                    return_iter,
                    1.0,
                    0.9,
                    NonZero::new(100).expect("update_frequency must be >= 1"),
                )
                .tune_initial_temperature(&tsp_model, None, 200, 0.5)
                .tune_cooling_rate(n_iter),
            ),
        ),
        (
            "AdaptiveAnnealingOptimizer",
            Box::new(
                AdaptiveAnnealingOptimizer::new(
                    patience,
                    16,
                    return_iter,
                    1.0,
                    Default::default(),
                    NonZero::new(100).expect("update_frequency must be >= 1"),
                )
                .tune_initial_temperature(&tsp_model, None, 200),
            ),
        ),
        (
            "PopulationAnnealingOptimizer",
            Box::new(
                PopulationAnnealingOptimizer::new(
                    patience,
                    16,
                    return_iter,
                    1.0,
                    0.9,
                    NonZero::new(100).expect("update_frequency must be >= 1"),
                    16,
                )
                .tune_initial_temperature(&tsp_model, None, 200, 0.5)
                .tune_cooling_rate(n_iter),
            ),
        ),
        (
            "ParallelTemperingOptimizer",
            Box::new(ParallelTemperingOptimizer::with_geometric_betas(
                patience,
                16,
                return_iter,
                8,                                                        // replicas
                1e-3,                                                     // beta_min
                1e2,                                                      // beta_max
                NonZero::new(10).expect("update_frequency must be >= 1"), // update_frequency
            )),
        ),
        (
            "TabuSearchOptimizer",
            Box::new(TabuSearchOptimizer::<DequeTabuList>::new(
                patience,
                128,
                return_iter,
                10,
            )),
        ),
        (
            "EpsilonGreedyOptimizer",
            Box::new(EpsilonGreedyOptimizer::new(patience, 16, return_iter, 0.9)),
        ),
        (
            "RelativeAnnealingOptimizer",
            Box::new(RelativeAnnealingOptimizer::new(
                patience,
                16,
                return_iter,
                1.0e2,
            )),
        ),
        (
            "TsallisRelativeAnnealingOptimizer",
            Box::new(TsallisRelativeAnnealingOptimizer::new(
                patience,
                16,
                return_iter,
                1.0e2,
                NonZero::new(100).expect("update_frequency must be >= 1"),
                2.5,
                1.0,
            )),
        ),
    ];

    let selected = optimizer_name.as_deref();
    let run_alns_flag = selected.is_none_or(|name| name == "AlnsOptimizer");

    for (name, optimizer) in optimizers {
        if selected.is_some_and(|selected| selected != name) {
            continue;
        }
        println!("run {}", name);
        pb.reset();
        let (final_solution, final_score) = optimizer
            .run_with_callback(
                &tsp_model,
                initial_solution.clone(),
                n_iter,
                time_limit,
                &mut callback,
            )
            .unwrap();
        pb.finish_and_clear();
        println!(
            "final score = {}, num of cities {}",
            final_score,
            final_solution.len()
        );
    }

    if run_alns_flag {
        println!("run AlnsOptimizer");
        let (final_solution, final_score) =
            run_alns(&tsp_model, initial_solution.clone(), n_iter, time_limit);
        println!(
            "final score = {}, num of cities {}",
            final_score,
            final_solution.len()
        );
    }

    if positional.len() < 2 {
        return;
    }
    let opt_route_file = positional.get(1).unwrap();
    let opt_solution = read_lines(opt_route_file)
        .unwrap()
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
