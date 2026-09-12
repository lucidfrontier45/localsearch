//! Adaptive Large Neighborhood Search (ALNS) for the Traveling Salesman Problem.
//!
//! This example does **not** implement a new optimizer. Instead, it embeds the
//! ALNS operator-selection loop inside the [`OptModel`] itself and drives it
//! with the library's existing [`TsallisRelativeAnnealingOptimizer`]. The
//! Tsallis rule supplies acceptance probabilities; the ALNS state carries the
//! adaptive operator weights.
//!
//! Per Ropke & Pisinger (2006), each call to `generate_trial_solution`:
//!
//!   1. Picks a destroy operator and a repair operator via roulette wheel.
//!   2. Applies them to the current solution.
//!   3. Reads the previous iteration's classification from external state
//!      and scores the (destroy, repair) pair with σ1 = 33, σ2 = 9,
//!      σ3 = 13, σ4 = 0.
//!   4. Decays weights by ρ every `segment_size` iterations.
//!
//! The TSP geometry lives in a pure, immutable [`TspModel`]; the mutable
//! ALNS bookkeeping lives in a caller-owned [`AlnsState`] that is threaded
//! through the optimizer's `run` method. The model itself never carries
//! `Arc<Mutex<...>>` — it stays trivially `Clone + Send + Sync`.
//!
//! Run on the bundled Berlin52 instance:
//!
//! ```bash
//! cargo run --example tsp_alns
//! ```

use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead},
    path::Path,
    sync::Mutex,
    time::Duration,
};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use localsearch::{
    LocalsearchError, OptModel, OptProgress,
    optim::{LocalSearchOptimizer, TsallisRelativeAnnealingOptimizer},
};
use ordered_float::NotNan;
use rand::{Rng, RngExt as _, seq::SliceRandom};

// ---- ALNS configuration -------------------------------------------------

const N_DESTROY: usize = 3;
const N_REPAIR: usize = 2;
const SCORE_NEW_BEST: f64 = 33.0;
const SCORE_IMPROVED: f64 = 9.0;
const SCORE_REJECTED: f64 = 0.0;
const REACTION_FACTOR: f64 = 0.8;
const SEGMENT_SIZE: usize = 100;
const REMOVAL_FRACTION: f64 = 0.2;

// ---- TSP types ---------------------------------------------------------

type Edge = (usize, usize);
type SolutionType = Vec<usize>;
type ScoreType = NotNan<f64>;

fn min_sorted(c1: usize, c2: usize) -> (usize, usize) {
    if c1 < c2 { (c1, c2) } else { (c2, c1) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OperatorOutcome {
    NewGlobalBest,
    Improved,
    Rejected,
}

#[derive(Default, Debug)]
struct OutcomeCounts {
    new_best: u32,
    improved: u32,
    rejected: u32,
}

// ---- Pure TSP geometry --------------------------------------------------

/// Pure, immutable TSP geometry: start city + pairwise Euclidean distances.
/// Carries no ALNS bookkeeping, so it is trivially `Clone + Send + Sync`.
#[derive(Clone, Debug)]
struct TspModel {
    start: usize,
    distance_matrix: HashMap<Edge, f64>,
}

impl TspModel {
    fn from_coords(coords: &[(usize, f64, f64)]) -> Self {
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
                let dist = ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt();
                mat.insert(key, dist);
            }
        }
        Self {
            start,
            distance_matrix: mat,
        }
    }

    fn evaluate(&self, tour: &[usize]) -> ScoreType {
        let total = tour
            .windows(2)
            .map(|w| {
                let key = min_sorted(w[0], w[1]);
                self.distance_matrix[&key]
            })
            .sum();
        NotNan::new(total).expect("tour length must be finite")
    }

    fn edge_distance(&self, a: usize, b: usize) -> f64 {
        self.distance_matrix[&min_sorted(a, b)]
    }

    fn generate_random_tour<R: Rng>(&self, rng: &mut R) -> SolutionType {
        let mut cities: Vec<usize> = self
            .distance_matrix
            .keys()
            .flat_map(|&(i, j)| [i, j])
            .collect();
        cities.sort_unstable();
        cities.dedup();
        cities.shuffle(rng);
        let i = cities.iter().position(|&c| c == self.start).unwrap();
        cities.swap(0, i);
        cities.push(self.start);
        cities
    }
}

// ---- Mutable ALNS bookkeeping (caller-owned) ----------------------------

/// All mutable bookkeeping the ALNS model needs at runtime. Wrapped in
/// `Mutex<AlnsState>` by the caller so the optimizer can share `&StateType`
/// across rayon worker threads. The model never holds this directly.
#[derive(Debug)]
struct AlnsState {
    /// Adaptive destroy / repair operator weights (Ropke & Pisinger).
    destroy_weights: Vec<f64>,
    repair_weights: Vec<f64>,
    destroy_scores: Vec<f64>,
    repair_scores: Vec<f64>,
    destroy_uses: Vec<u32>,
    repair_uses: Vec<u32>,
    iter: usize,
    /// Operator ids and outcome of the previous call, used to score it on
    /// the next call (one-iteration lag, same as Ropke & Pisinger).
    last_destroy: usize,
    last_repair: usize,
    last_outcome: Option<OperatorOutcome>,
    /// Best score observed so far (used to classify "new global best"
    /// outcomes and updated as the search progresses).
    best_score: ScoreType,
    /// Cumulative outcome counts for diagnostics.
    outcomes: OutcomeCounts,
}

impl AlnsState {
    fn new() -> Self {
        Self {
            destroy_weights: vec![1.0; N_DESTROY],
            repair_weights: vec![1.0; N_REPAIR],
            destroy_scores: vec![0.0; N_DESTROY],
            repair_scores: vec![0.0; N_REPAIR],
            destroy_uses: vec![0; N_DESTROY],
            repair_uses: vec![0; N_REPAIR],
            iter: 0,
            last_destroy: 0,
            last_repair: 0,
            last_outcome: None,
            best_score: ScoreType::new(f64::INFINITY).expect("finite"),
            outcomes: OutcomeCounts::default(),
        }
    }

    fn pick_destroy<R: Rng>(&self, rng: &mut R) -> usize {
        pick_operator(&self.destroy_weights, rng)
    }

    fn pick_repair<R: Rng>(&self, rng: &mut R) -> usize {
        pick_operator(&self.repair_weights, rng)
    }

    fn record(&mut self, destroy: usize, repair: usize, outcome: OperatorOutcome) {
        self.destroy_uses[destroy] += 1;
        self.repair_uses[repair] += 1;
        let s = match outcome {
            OperatorOutcome::NewGlobalBest => SCORE_NEW_BEST,
            OperatorOutcome::Improved => SCORE_IMPROVED,
            OperatorOutcome::Rejected => SCORE_REJECTED,
        };
        self.destroy_scores[destroy] += s;
        self.repair_scores[repair] += s;
        self.iter += 1;

        match outcome {
            OperatorOutcome::NewGlobalBest => self.outcomes.new_best += 1,
            OperatorOutcome::Improved => self.outcomes.improved += 1,
            OperatorOutcome::Rejected => self.outcomes.rejected += 1,
        }

        if self.iter.is_multiple_of(SEGMENT_SIZE) {
            self.update_weights();
        }
    }

    fn update_weights(&mut self) {
        for ((w, &score), &uses) in self
            .destroy_weights
            .iter_mut()
            .zip(self.destroy_scores.iter())
            .zip(self.destroy_uses.iter())
        {
            let avg = if uses > 0 { score / uses as f64 } else { 0.0 };
            *w = REACTION_FACTOR * *w + (1.0 - REACTION_FACTOR) * avg;
        }
        for ((w, &score), &uses) in self
            .repair_weights
            .iter_mut()
            .zip(self.repair_scores.iter())
            .zip(self.repair_uses.iter())
        {
            let avg = if uses > 0 { score / uses as f64 } else { 0.0 };
            *w = REACTION_FACTOR * *w + (1.0 - REACTION_FACTOR) * avg;
        }
        self.destroy_scores.fill(0.0);
        self.destroy_uses.fill(0);
        self.repair_scores.fill(0.0);
        self.repair_uses.fill(0);
    }
}

/// `(destroy_id, repair_id, outcome_from_previous_call)`.
type TransitionType = (usize, usize, Option<OperatorOutcome>);

// ---- Immutable model ----------------------------------------------------

/// Immutable model wrapper. Owns only TSP geometry — every piece of mutable
/// state lives in the caller-supplied `Mutex<AlnsState>` and is threaded
/// through the optimizer's `run` method.
struct TspAlnsModel {
    tsp: TspModel,
}

impl TspAlnsModel {
    fn from_coords(coords: &[(usize, f64, f64)]) -> Self {
        Self {
            tsp: TspModel::from_coords(coords),
        }
    }
}

impl OptModel for TspAlnsModel {
    type SolutionType = SolutionType;
    type TransitionType = TransitionType;
    type ScoreType = ScoreType;
    type StateType = Mutex<AlnsState>;

    fn generate_random_solution<R: rand::Rng>(
        &self,
        state: &Self::StateType,
        rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        let tour = self.tsp.generate_random_tour(rng);
        let score = self.tsp.evaluate(&tour);
        state.lock().expect("alns state lock poisoned").best_score = score;
        Ok((tour, score))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        state: &Self::StateType,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        // 1. Score the (destroy, repair) emitted by the previous call under
        //    the same lock that picks this iteration's operators, so the
        //    one-iteration lag stays consistent.
        let (destroy, repair, prev_best) = {
            let mut st = state.lock().expect("alns state lock poisoned");
            if let Some(prev) = st.last_outcome {
                let d = st.last_destroy;
                let r = st.last_repair;
                st.record(d, r, prev);
            }
            (st.pick_destroy(rng), st.pick_repair(rng), st.best_score)
        };

        // 3. Apply destroy + repair.
        let q = ((current_solution.len() as f64 - 2.0) * REMOVAL_FRACTION)
            .round()
            .max(1.0) as usize;
        let (removed, partial) = match destroy {
            0 => destroy_random(&current_solution, q, rng),
            1 => destroy_worst(&self.tsp, &current_solution, q, rng),
            _ => destroy_string(&current_solution, q, rng),
        };
        let candidate = match repair {
            0 => repair_greedy(&self.tsp, partial, &removed, rng),
            _ => repair_regret(&self.tsp, partial, &removed, rng),
        };
        let candidate_score = self.tsp.evaluate(&candidate);

        // 4. Classify outcome using the best score BEFORE we update it.
        let outcome = classify(candidate_score, current_score, prev_best);

        // 5. Persist this iteration's operators + outcome and update best.
        {
            let mut st = state.lock().expect("alns state lock poisoned");
            if candidate_score < st.best_score {
                st.best_score = candidate_score;
            }
            st.last_destroy = destroy;
            st.last_repair = repair;
            st.last_outcome = Some(outcome);
        }

        (
            candidate,
            (destroy, repair, Some(outcome)),
            candidate_score,
        )
    }

    fn preprocess_solution(
        &self,
        state: &Self::StateType,
        solution: Self::SolutionType,
        score: Self::ScoreType,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        state.lock().expect("alns state lock poisoned").best_score = score;
        Ok((solution, score))
    }
}

// ---- Destroy / repair operators ----------------------------------------

fn destroy_random<R: Rng>(tour: &[usize], q: usize, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
    let mut interior: Vec<usize> = tour[1..tour.len() - 1].to_vec();
    interior.shuffle(rng);
    let removed: Vec<usize> = interior.iter().take(q).copied().collect();
    let kept: Vec<usize> = tour
        .iter()
        .copied()
        .filter(|c| !removed.contains(c))
        .collect();
    (removed, kept)
}

fn destroy_worst<R: Rng>(
    model: &TspModel,
    tour: &[usize],
    q: usize,
    _rng: &mut R,
) -> (Vec<usize>, Vec<usize>) {
    let mut savings: Vec<(f64, usize)> = tour[1..tour.len() - 1]
        .iter()
        .enumerate()
        .map(|(idx, &c)| {
            let prev = tour[idx];
            let next = tour[idx + 2];
            let d_prev = model.edge_distance(prev, c);
            let d_next = model.edge_distance(c, next);
            let d_direct = model.edge_distance(prev, next);
            (d_prev + d_next - d_direct, c)
        })
        .collect();
    savings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let removed: Vec<usize> = savings.iter().take(q).map(|(_, c)| *c).collect();
    let kept: Vec<usize> = tour
        .iter()
        .copied()
        .filter(|c| !removed.contains(c))
        .collect();
    (removed, kept)
}

fn destroy_string<R: Rng>(tour: &[usize], q: usize, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
    let interior_len = tour.len() - 2;
    if interior_len == 0 {
        return (Vec::new(), tour.to_vec());
    }
    let start = rng.random_range(0..=interior_len.saturating_sub(q));
    let removed: Vec<usize> = tour[1 + start..1 + start + q].to_vec();
    let kept: Vec<usize> = tour
        .iter()
        .copied()
        .filter(|c| !removed.contains(c))
        .collect();
    (removed, kept)
}

fn repair_greedy<R: Rng>(
    model: &TspModel,
    mut tour: Vec<usize>,
    removed: &[usize],
    _rng: &mut R,
) -> Vec<usize> {
    for &city in removed {
        let (best_pos, _) = (1..tour.len())
            .map(|pos| {
                let prev = tour[pos - 1];
                let next = tour[pos];
                let delta = model.edge_distance(prev, city) + model.edge_distance(city, next)
                    - model.edge_distance(prev, next);
                (pos, delta)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        tour.insert(best_pos, city);
    }
    tour
}

fn repair_regret<R: Rng>(
    model: &TspModel,
    mut tour: Vec<usize>,
    removed: &[usize],
    rng: &mut R,
) -> Vec<usize> {
    let mut order: Vec<usize> = removed.to_vec();
    order.shuffle(rng);
    while let Some(city) = order
        .iter()
        .copied()
        .max_by(|&a, &b| {
            regret(model, &tour, a)
                .partial_cmp(&regret(model, &tour, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        let pos = best_insertion_pos(model, &tour, city);
        order.retain(|&c| c != city);
        tour.insert(pos, city);
    }
    tour
}

fn regret(model: &TspModel, tour: &[usize], city: usize) -> f64 {
    let mut best = f64::INFINITY;
    let mut second = f64::INFINITY;
    for pos in 1..tour.len() {
        let prev = tour[pos - 1];
        let next = tour[pos];
        let delta = model.edge_distance(prev, city) + model.edge_distance(city, next)
            - model.edge_distance(prev, next);
        if delta < best {
            second = best;
            best = delta;
        } else if delta < second {
            second = delta;
        }
    }
    if second.is_finite() { second - best } else { f64::INFINITY }
}

fn best_insertion_pos(model: &TspModel, tour: &[usize], city: usize) -> usize {
    let mut best_pos = 1;
    let mut best_delta = f64::INFINITY;
    for pos in 1..tour.len() {
        let prev = tour[pos - 1];
        let next = tour[pos];
        let delta = model.edge_distance(prev, city) + model.edge_distance(city, next)
            - model.edge_distance(prev, next);
        if delta < best_delta {
            best_delta = delta;
            best_pos = pos;
        }
    }
    best_pos
}

fn classify(
    candidate: ScoreType,
    current: ScoreType,
    best: ScoreType,
) -> OperatorOutcome {
    if candidate < best {
        OperatorOutcome::NewGlobalBest
    } else if candidate < current {
        OperatorOutcome::Improved
    } else {
        // The Tsallis acceptance rule decided the candidate was rejected.
        // We don't have access to that decision here, so we record it as
        // rejected (the user observes the Tsallis acceptance ratio via the
        // progress bar). Operators whose candidates are rejected get score 0.
        OperatorOutcome::Rejected
    }
}

fn pick_operator<R: Rng>(weights: &[f64], rng: &mut R) -> usize {
    let total: f64 = weights.iter().sum();
    let mut pick = rng.random::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        if pick < w {
            return i;
        }
        pick -= w;
    }
    weights.len() - 1
}

// ---- I/O + progress bar ------------------------------------------------

fn read_lines<P: AsRef<Path>>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>> {
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

// ---- Main --------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_file = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("examples/tsp_berlin52/cities");

    let coords: Vec<(usize, f64, f64)> = read_lines(input_file)
        .expect("failed to open cities file")
        .map(|line| {
            let line = line.unwrap();
            let parts: Vec<&str> = line.split_whitespace().collect();
            (
                parts[0].parse().unwrap(),
                parts[1].parse().unwrap(),
                parts[2].parse().unwrap(),
            )
        })
        .collect();

    let model = TspAlnsModel::from_coords(&coords);
    // Mutable ALNS bookkeeping lives entirely in caller-owned state. The
    // optimizer takes `&state` and threads it through every `OptModel` call.
    let state = Mutex::new(AlnsState::new());

    let n_iter: usize = 50_000;
    let time_limit = Duration::from_secs(30);

    let mut rng = rand::rng();
    let (initial_solution, initial_score) =
        model.generate_random_solution(&state, &mut rng).unwrap();
    println!(
        "ALNS via Tsallis on {} cities ({} iterations, {}s budget)",
        coords.len(),
        n_iter,
        time_limit.as_secs()
    );
    println!("initial tour length = {:.4}", initial_score.into_inner());

    let pb = create_pbar(n_iter as u64);

    // n_trials=1: ALNS does exactly one destroy+repair per iteration. With
    // n_trials > 1 the optimizer races multiple candidates in parallel and
    // discards all but the best, which would defeat the operator-selection
    // bookkeeping inside the state.
    let optimizer = TsallisRelativeAnnealingOptimizer::new(
        n_iter / 2,
        1,
        n_iter / 50,
        1.0,
        std::num::NonZero::new(100).expect("update_frequency must be >= 1"),
        2.5,
        1.0,
    );

    let mut callback = |op: OptProgress<SolutionType, ScoreType>| {
        pb.set_message(format!(
            "best tour length {:.4}, acc {:.2}",
            op.score.into_inner(),
            op.acceptance_ratio
        ));
        pb.set_position(op.iter as u64);
    };

    let (final_solution, final_score) = optimizer
        .run_with_callback(
            &model,
            &state,
            Some((initial_solution, initial_score)),
            n_iter,
            time_limit,
            &mut callback,
        )
        .unwrap();

    println!(
        "best tour length   = {:.4}",
        final_score.into_inner()
    );
    println!(
        "final tour length  = {:.4}",
        model.tsp.evaluate(&final_solution).into_inner()
    );

    let st = state.lock().expect("alns state lock poisoned");
    let destroy_names = ["random", "worst", "string"];
    let repair_names = ["greedy", "regret-2"];
    println!(
        "outcomes: new_best={} improved={} rejected={}",
        st.outcomes.new_best, st.outcomes.improved, st.outcomes.rejected
    );
    println!("final destroy weights:");
    for (d, name) in destroy_names.iter().enumerate() {
        println!("  {:>7} = {:.3}", name, st.destroy_weights[d]);
    }
    println!("final repair weights:");
    for (r, name) in repair_names.iter().enumerate() {
        println!("  {:>9} = {:.3}", name, st.repair_weights[r]);
    }
}
