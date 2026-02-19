use crate::{
    Duration, OptModel,
    optim::{LocalSearchOptimizer, ParallelTemperingOptimizer},
    tests::QuadraticModel,
};

#[test]
fn test_parallel_tempering_basic() {
    let model = QuadraticModel::new(3, vec![0.1, -0.2, 0.3], (-1.0, 1.0));
    let mut rng = rand::rng();
    let (init_sol, init_score) = model.generate_random_solution(&mut rng).unwrap();

    let pt = ParallelTemperingOptimizer::with_geometric_betas(
        50,   // patience
        10,   // n_trials
        10,   // return_iter
        6,    // n_replicas
        1e-2, // beta_min
        1e2,  // beta_max
        5,    // update_frequency
    );

    let (_best_solution, best_score) = pt
        .run(
            &model,
            Some((init_sol, init_score)),
            200,
            Duration::from_secs(1),
        )
        .unwrap();

    assert!(best_score.into_inner().is_finite());
}
