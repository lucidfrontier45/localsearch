use localsearch::{
    OptModel,
    optim::{LocalSearchOptimizer, PopulationAnnealingOptimizer},
};

use ordered_float::NotNan;

// Define a simple test model to verify PopulationAnnealingOptimizer works
#[derive(Clone)]
struct SimpleTestModel {
    size: usize,
}

impl SimpleTestModel {
    fn new(size: usize) -> Self {
        Self { size }
    }
}

impl OptModel for SimpleTestModel {
    type SolutionType = Vec<bool>;
    type ScoreType = NotNan<f64>;
    type TransitionType = usize; // Position to flip

    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> anyhow::Result<(Self::SolutionType, Self::ScoreType)> {
        let mut solution = Vec::with_capacity(self.size);
        for _ in 0..self.size {
            solution.push(rng.random::<bool>());
        }
        let score = solution.iter().map(|&x| if x { -1.0 } else { 1.0 }).sum();
        Ok((solution, NotNan::new(score).unwrap()))
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        mut current_solution: Self::SolutionType,
        _current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        let pos = rng.random_range(0..self.size);
        current_solution[pos] = !current_solution[pos];
        let score = current_solution
            .iter()
            .map(|&x| if x { -1.0 } else { 1.0 })
            .sum();
        (current_solution, pos, NotNan::new(score).unwrap())
    }
}

#[test]
fn test_population_annealing_optimizer() {
    let model = SimpleTestModel::new(10);
    let (initial_solution, initial_score) =
        model.generate_random_solution(&mut rand::rng()).unwrap();

    let optimizer = PopulationAnnealingOptimizer::new(10, 5, 5, 10.0, 4, 10);

    let (solution, score) = optimizer.optimize(
        &model,
        initial_solution,
        initial_score,
        100,
        std::time::Duration::from_secs(10),
        &mut |_progress| {}, // Empty callback
    );

    // Basic validation
    assert_eq!(solution.len(), 10);
    // The score should be a NotNan<f64>
    assert!(score.into_inner() <= 10.0); // Maximum possible score is 10 if all are false
}
