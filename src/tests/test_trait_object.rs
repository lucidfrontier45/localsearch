use std::time::Duration;

use approx::assert_abs_diff_eq;

use crate::optim::{EpsilonGreedyOptimizer, HillClimbingOptimizer, LocalSearchOptimizer};

use super::QuadraticModel;

#[test]
fn test() {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
    let opts: [Box<dyn LocalSearchOptimizer<QuadraticModel>>; 2] = [
        Box::new(HillClimbingOptimizer::new(1000, 10)),
        Box::new(EpsilonGreedyOptimizer::new(1000, 10, 200, 0.1)),
    ];

    for opt in opts.iter() {
        let (final_solution, final_score) = opt
            .run(&model, None, 10000, Duration::from_secs(10))
            .unwrap();
        assert_abs_diff_eq!(2.0, final_solution[0], epsilon = 0.05);
        assert_abs_diff_eq!(0.0, final_solution[1], epsilon = 0.05);
        assert_abs_diff_eq!(-3.5, final_solution[2], epsilon = 0.05);
        assert_abs_diff_eq!(0.0, final_score.into_inner(), epsilon = 0.05);
    }
}
