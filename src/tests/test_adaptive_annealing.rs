use std::{num::NonZero, time::Duration};

use approx::assert_abs_diff_eq;

use super::QuadraticModel;
use crate::optim::{
    AdaptiveAnnealing, AdaptiveAnnealingOptimizer, AdaptiveScheduler, LocalSearchOptimizer,
    TargetAccScheduleMode,
};

#[test]
fn test() {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
    let opt = AdaptiveAnnealingOptimizer::new(
        10000,
        10,
        500,
        1e-2,
        AdaptiveScheduler::default(),
        NonZero::new(100).expect("update_frequency must be >= 1"),
    );
    let scheduler = AdaptiveScheduler::new(0.3, 0.3, TargetAccScheduleMode::Constant, 0.05);
    let handler = AdaptiveAnnealing::new(1e-2, scheduler, NonZero::new(100).unwrap());
    let (final_solution, final_score) = opt
        .run(&model, None, 10000, Duration::from_secs(10), handler)
        .unwrap();
    assert_abs_diff_eq!(2.0, final_solution[0], epsilon = 0.05);
    assert_abs_diff_eq!(0.0, final_solution[1], epsilon = 0.05);
    assert_abs_diff_eq!(-3.5, final_solution[2], epsilon = 0.05);
    assert_abs_diff_eq!(0.0, final_score.into_inner(), epsilon = 0.05);
}
