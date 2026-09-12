//! Solve the Travelling Salesman Problem with Adaptive Large Neighborhood
//! Search (ALNS).
//!
//! Each iteration picks one destroy operator (e.g. random-removal,
//! worst-removal) and one repair operator (greedy-insertion,
//! random-insertion) via independent roulette-wheel draws. After the trial
//! is scored by the acceptance handler, the chosen operators are credited
//! according to the outcome and their weights are blended at every segment
//! boundary using the reaction-factor rule from Ropke & Pisinger (2006).
//!
//! Usage:
//!
//! ```text
//! cargo run --example tsp_alns_model --release -- <input_file> [opt_route_file]
//! ```
//!
//! `<input_file>` contains one city per line as `<id> <x> <y>`. The optional
//! `<opt_route_file>` holds an optimal tour as one city id per line; when
//! supplied, the example prints the gap against the optimum at the end.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead},
    num::NonZero,
    path::Path,
    time::Duration,
};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use localsearch::{
    LocalsearchError, OptModel, OptProgress,
    optim::{
        AdaptiveScheduler, AlnsTrialGenerator, DestroyOperator, GenericLocalSearchOptimizer,
        LocalSearchOptimizer, RepairOperator, TargetAccScheduleMode, TsallisAnnealing,
    },
};
use ordered_float::NotNan;
use rand::{RngExt as _, seq::SliceRandom};

// ---------------------------------------------------------------------------
// TSP model
// ---------------------------------------------------------------------------

type Edge = (usize, usize);
type SolutionType = Vec<usize>;
/// Partial tour during ALNS: `(tour_with_gaps, removed_cities)`. Positions
/// in `tour_with_gaps` are fixed; `None` slots mark cities that have been
/// destroyed and still need to be reinserted.
type PartialTour = (Vec<Option<usize>>, Vec<usize>);
type ScoreType = NotNan<f64>;

fn min_sorted(c1: usize, c2: usize) -> (usize, usize) {
    if c1 < c2 { (c1, c2) } else { (c2, c1) }
}

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
                    // Self-loops are queried by Shaw removal when ranking
                    // cities against their own seed; keep the entry at 0.
                    let key = min_sorted(c1, c2);
                    mat.insert(key, 0.0);
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

    fn get_distance(&self, key: &(usize, usize)) -> f64 {
        self.distance_matrix[key]
    }

    fn evaluate_solution(&self, solution: &SolutionType) -> ScoreType {
        let score = (0..solution.len() - 1)
            .map(|i| {
                let key = min_sorted(solution[i], solution[i + 1]);
                self.get_distance(&key)
            })
            .sum();
        NotNan::new(score).unwrap()
    }
}

impl OptModel for TSPModel {
    type SolutionType = SolutionType;
    type TransitionType = ();
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

        // Pin start at index 0 and close the tour by repeating it at the end.
        let i = cities.iter().position(|&c| c == self.start).unwrap();
        cities.swap(0, i);
        cities.push(self.start);

        let score = self.evaluate_solution(&cities);
        Ok((cities, score))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        _rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        // Unused: ALNS short-circuits trial generation via
        // `TrialGenerator::generate_trial`, so this default is a no-op.
        (current_solution, (), current_score)
    }
}

// ---------------------------------------------------------------------------
// Destroy operators
// ---------------------------------------------------------------------------

/// Remove a random fraction of interior cities. `min_frac`/`max_frac`
/// bracket the share of cities destroyed per call.
#[derive(Clone)]
struct RandomRemoval {
    min_frac: f64,
    max_frac: f64,
}

impl DestroyOperator<TSPModel, PartialTour> for RandomRemoval {
    fn destroy(&self, _model: &TSPModel, solution: SolutionType) -> PartialTour {
        let mut rng = rand::rng();
        let n = solution.len() - 2; // exclude the start city pinned at both ends
        let frac = rng.random_range(self.min_frac..=self.max_frac);
        let n_remove = (((n as f64) * frac).round() as usize).clamp(1, n);

        let mut partial: Vec<Option<usize>> = solution.into_iter().map(Some).collect();
        let mut interior: Vec<usize> = (1..partial.len() - 1).collect();
        interior.shuffle(&mut rng);
        let mut removed = Vec::with_capacity(n_remove);
        for &i in interior.iter().take(n_remove) {
            // SAFETY: interior only references indices 1..len-1, all of which
            // hold `Some` at this point.
            removed.push(partial[i].unwrap());
            partial[i] = None;
        }
        (partial, removed)
    }

    fn dyn_clone(&self) -> Box<dyn DestroyOperator<TSPModel, PartialTour>> {
        Box::new(self.clone())
    }
}

/// Remove the cities whose removal saves the most distance — the "worst"
/// positions in the current tour. The same fraction range as
/// [`RandomRemoval`] is used.
#[derive(Clone)]
struct WorstRemoval {
    min_frac: f64,
    max_frac: f64,
}

impl DestroyOperator<TSPModel, PartialTour> for WorstRemoval {
    fn destroy(&self, model: &TSPModel, solution: SolutionType) -> PartialTour {
        let mut rng = rand::rng();
        let n = solution.len() - 2;
        let frac = rng.random_range(self.min_frac..=self.max_frac);
        let n_remove = (((n as f64) * frac).round() as usize).clamp(1, n);

        // Saving of removing interior city at index i:
        // d(prev, i) + d(i, next) - d(prev, next).
        let mut savings: Vec<(usize, f64)> = Vec::with_capacity(n);
        for i in 1..solution.len() - 1 {
            let prev = solution[i - 1];
            let cur = solution[i];
            let next = solution[i + 1];
            let save = model.get_distance(&min_sorted(prev, cur))
                + model.get_distance(&min_sorted(cur, next))
                - model.get_distance(&min_sorted(prev, next));
            savings.push((i, save));
        }
        // Highest savings first.
        savings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut partial: Vec<Option<usize>> = solution.into_iter().map(Some).collect();
        let mut removed = Vec::with_capacity(n_remove);
        for &(i, _) in savings.iter().take(n_remove) {
            removed.push(partial[i].unwrap());
            partial[i] = None;
        }
        (partial, removed)
    }

    fn dyn_clone(&self) -> Box<dyn DestroyOperator<TSPModel, PartialTour>> {
        Box::new(self.clone())
    }
}
/// Remove a contiguous cluster of related cities around a random seed.
///
/// Shaw removal (Shaw & Cordeau, 1997) destroys cities that are close in the
/// distance matrix, which keeps the rest of the tour tightly connected and
/// lets repair recombine good local structure. Combined with random/worst
/// removal it gives the ALNS pool enough diversity.
#[derive(Clone)]
struct ShawRemoval {
    min_frac: f64,
    max_frac: f64,
}

impl DestroyOperator<TSPModel, PartialTour> for ShawRemoval {
    fn destroy(&self, model: &TSPModel, solution: SolutionType) -> PartialTour {
        let mut rng = rand::rng();
        let n = solution.len() - 2;
        let frac = rng.random_range(self.min_frac..=self.max_frac);
        let n_remove = (((n as f64) * frac).round() as usize).clamp(1, n);

        // Pick a random interior seed; rank every other interior city by
        // distance to it, then take the `n_remove` nearest (the seed itself
        // is included with distance 0).
        let seed_idx = rng.random_range(1..solution.len() - 1);
        let seed = solution[seed_idx];
        let mut related: Vec<(usize, f64)> = (1..solution.len() - 1)
            .map(|i| (i, model.get_distance(&min_sorted(seed, solution[i]))))
            .collect();
        related.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut partial: Vec<Option<usize>> = solution.into_iter().map(Some).collect();
        let mut removed = Vec::with_capacity(n_remove);
        for &(i, _) in related.iter().take(n_remove) {
            removed.push(partial[i].unwrap());
            partial[i] = None;
        }
        (partial, removed)
    }

    fn dyn_clone(&self) -> Box<dyn DestroyOperator<TSPModel, PartialTour>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
/// Walk the partial tour and return the index of the nearest present city
/// to the left (resp. right) of position `i`. Both endpoints are pinned to
/// the start city, so an answer is guaranteed for any interior index.
fn neighbours(partial: &[Option<usize>], i: usize) -> (usize, usize) {
    let prev = (0..i)
        .rev()
        .find_map(|j| partial[j])
        .expect("partial tour must contain a city left of any interior gap");
    let next = (i + 1..partial.len())
        .find_map(|j| partial[j])
        .expect("partial tour must contain a city right of any interior gap");
    (prev, next)
}
/// 2-opt local search: repeatedly reverse every improving subsegment.
///
/// Returns `true` if any reversal was applied during the pass.
fn two_opt_pass(model: &TSPModel, sol: &mut SolutionType) -> bool {
    let n = sol.len();
    if n < 4 {
        return false;
    }
    // sol = [start, c1, ..., ck, start]. The interior is 1..n-1; the last
    // entry equals the start city and stays pinned.
    let mut improved = false;
    for i in 1..n - 2 {
        for j in (i + 1)..n - 1 {
            // Old edges: (sol[i-1], sol[i]), (sol[j], sol[j+1]).
            // New edges after reversing sol[i..=j]: (sol[i-1], sol[j]), (sol[i], sol[j+1]).
            let a = sol[i - 1];
            let b = sol[i];
            let c = sol[j];
            let d = sol[j + 1];
            let old = model.get_distance(&min_sorted(a, b)) + model.get_distance(&min_sorted(c, d));
            let new = model.get_distance(&min_sorted(a, c)) + model.get_distance(&min_sorted(b, d));
            if new + 1e-9 < old {
                sol[i..=j].reverse();
                improved = true;
            }
        }
    }
    improved
}

/// Iterate 2-opt until a pass produces no improvement.
fn two_opt(model: &TSPModel, mut sol: SolutionType) -> SolutionType {
    while two_opt_pass(model, &mut sol) {}
    sol
}

/// Insert every removed city at the gap with the smallest insertion cost.
#[derive(Clone)]
struct GreedyInsertion;

impl RepairOperator<TSPModel, PartialTour> for GreedyInsertion {
    fn repair(
        &self,
        model: &TSPModel,
        (mut partial, mut removed): PartialTour,
    ) -> (SolutionType, ScoreType) {
        let mut rng = rand::rng();
        // Process removed cities in random order to break symmetry.
        removed.shuffle(&mut rng);

        for city in removed {
            let mut best_pos = 0;
            let mut best_cost = f64::INFINITY;
            for (i, slot) in partial.iter().enumerate() {
                if slot.is_some() {
                    continue;
                }
                let (prev, next) = neighbours(&partial, i);
                let cost = model.get_distance(&min_sorted(prev, city))
                    + model.get_distance(&min_sorted(city, next))
                    - model.get_distance(&min_sorted(prev, next));
                if cost < best_cost {
                    best_cost = cost;
                    best_pos = i;
                }
            }
            partial[best_pos] = Some(city);
            let _ = &mut rng; // silence unused-mut when n_remove == 0
        }

        let solution: SolutionType = partial.into_iter().map(|s| s.unwrap()).collect();
        // Local-search polish: iterate 2-opt until no improving reversal is
        // left. This is the single biggest quality driver for TSP ALNS —
        // greedy insertion alone plateaus well above optimum.
        let solution = two_opt(model, solution);
        let score = model.evaluate_solution(&solution);
        (solution, score)
    }

    fn dyn_clone(&self) -> Box<dyn RepairOperator<TSPModel, PartialTour>> {
        Box::new(self.clone())
    }
}

/// Insert every removed city at a uniformly random gap.
#[derive(Clone)]
struct RandomInsertion;

impl RepairOperator<TSPModel, PartialTour> for RandomInsertion {
    fn repair(
        &self,
        _model: &TSPModel,
        (mut partial, mut removed): PartialTour,
    ) -> (SolutionType, ScoreType) {
        let mut rng = rand::rng();
        removed.shuffle(&mut rng);

        for city in removed {
            let gaps: Vec<usize> = partial
                .iter()
                .enumerate()
                .filter_map(|(i, s)| if s.is_none() { Some(i) } else { None })
                .collect();
            // gaps is non-empty: every removed city has at least one open slot.
            let pos = gaps[rng.random_range(0..gaps.len())];
            partial[pos] = Some(city);
        }

        let solution: SolutionType = partial.into_iter().map(|s| s.unwrap()).collect();
        // Score is recomputed from scratch — cheap relative to destroy.
        let score = _model.evaluate_solution(&solution);
        (solution, score)
    }

    fn dyn_clone(&self) -> Box<dyn RepairOperator<TSPModel, PartialTour>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Operator pool assembly
// ---------------------------------------------------------------------------

fn destroy_operators() -> Vec<Box<dyn DestroyOperator<TSPModel, PartialTour>>> {
    // Tighter fractions than the textbook 10-30%: greedy+2-opt already
    // reconstructs well from a small destruction, and smaller gaps mean
    // more iterations of meaningful search within the same iteration budget.
    vec![
        Box::new(RandomRemoval {
            min_frac: 0.05,
            max_frac: 0.15,
        }),
        Box::new(WorstRemoval {
            min_frac: 0.05,
            max_frac: 0.15,
        }),
        Box::new(ShawRemoval {
            min_frac: 0.05,
            max_frac: 0.15,
        }),
    ]
}

fn repair_operators() -> Vec<Box<dyn RepairOperator<TSPModel, PartialTour>>> {
    vec![Box::new(GreedyInsertion), Box::new(RandomInsertion)]
}

fn build_alns_generator() -> AlnsTrialGenerator<TSPModel, PartialTour> {
    AlnsTrialGenerator::new(destroy_operators(), repair_operators())
        .with_segment_size(100)
        .with_reaction_factor(0.3)
}

// ---------------------------------------------------------------------------
// CLI plumbing
// ---------------------------------------------------------------------------

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
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(20));
    pb
}

fn print_usage(program: &str) {
    eprintln!("Usage: {program} <input_file> [opt_route_file]");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  <input_file>       TSP coordinates file (<id> <x> <y> per line)");
    eprintln!("  [opt_route_file]   optional file with optimal route (one city id per line)");
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "tsp_alns_model".to_string());

    let mut input_file: Option<String> = None;
    let mut opt_route_file: Option<String> = None;

    let mut idx = 1;
    while idx < args.len() {
        let arg = args[idx].as_str();
        if arg == "--help" || arg == "-h" {
            print_usage(&program);
            return;
        } else if arg.starts_with('-') {
            eprintln!("error: unknown option '{arg}'");
            print_usage(&program);
            std::process::exit(2);
        } else if input_file.is_none() {
            input_file = Some(args[idx].clone());
        } else if opt_route_file.is_none() {
            opt_route_file = Some(args[idx].clone());
        } else {
            eprintln!("error: unexpected positional argument '{}'", args[idx]);
            print_usage(&program);
            std::process::exit(2);
        }
        idx += 1;
    }

    let Some(input_file) = input_file else {
        print_usage(&program);
        std::process::exit(2);
    };

    let coords = read_lines(&input_file)
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

    let n_iter: usize = 100_000;
    let return_iter = n_iter / 50;
    let time_limit = Duration::from_secs(120);
    let patience = n_iter / 2;

    let mut rng = rand::rng();
    let initial_solution = tsp_model.generate_random_solution(&mut rng).ok();

    let pb = create_pbar(n_iter as u64);
    let mut callback = |op: OptProgress<SolutionType, ScoreType>| {
        pb.set_message(format!(
            "best score {:.4e}, acceptance ratio {:.2}",
            op.score.into_inner(),
            op.acceptance_ratio
        ));
        pb.set_position(op.iter as u64);
    };

    // ALNS trial generator + Tsallis relative-annealing acceptance handler,
    // glued through the generic optimizer entry point. The Tsallis offset is
    // seeded from the initial random tour's score so that the first iteration's
    // denominator is well-defined.
    let initial_score = initial_solution
        .as_ref()
        .expect("random initial solution")
        .1
        .into_inner();
    let handler = TsallisAnnealing::new(
        initial_score,
        1.0e2,
        2.0,
        2.0,
        AdaptiveScheduler::new(0.3, 0.3, TargetAccScheduleMode::Constant, 0.10),
        NonZero::new(100).expect("update_frequency must be >= 1"),
    );
    let generator = build_alns_generator();
    let optimizer: GenericLocalSearchOptimizer<ScoreType, TsallisAnnealing, _> =
        GenericLocalSearchOptimizer::new(patience, 1, return_iter, handler)
            .with_trial_generator(generator);

    println!("run ALNS");
    pb.reset();
    let (sol, score) = optimizer
        .run_with_callback(
            &tsp_model,
            initial_solution,
            n_iter,
            time_limit,
            &mut callback,
        )
        .unwrap();
    pb.finish_and_clear();
    println!("ALNS: final score = {}, num of cities {}", score, sol.len());

    let Some(opt_route_file) = opt_route_file else {
        return;
    };
    let opt_solution = read_lines(opt_route_file)
        .unwrap()
        .map(|line| line.unwrap().parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let opt_score = tsp_model.evaluate_solution(&opt_solution);
    println!(
        "optimal score = {}, num of cities {}",
        opt_score,
        opt_solution.len()
    );
    let gap = (score.into_inner() - opt_score.into_inner()) / opt_score.into_inner() * 100.0;
    println!("gap = {gap:.3}%");
}
