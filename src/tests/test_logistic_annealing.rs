use std::time::Duration;

use approx::assert_abs_diff_eq;
use ordered_float::NotNan;

use super::QuadraticModel;
use crate::optim::{LocalSearchOptimizer, LogisticAnnealingOptimizer};

#[test]
fn test() {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
    let opt = LogisticAnnealingOptimizer::new(5000, 10, 200, 1e1);
    let (final_solution, final_score) = opt
        .run(&model, None, 10000, Duration::from_secs(10))
        .unwrap();
    assert_abs_diff_eq!(2.0, final_solution[0], epsilon = 0.05);
    assert_abs_diff_eq!(0.0, final_solution[1], epsilon = 0.05);
    assert_abs_diff_eq!(-3.5, final_solution[2], epsilon = 0.05);
    assert_abs_diff_eq!(0.0, final_score.into_inner(), epsilon = 0.05);
}

/// Regression: starting at a solution whose score is exactly `0.0` forces the
/// inner loop through `current_score == 0`, the degenerate-input case that
/// previously produced `inf`/`NaN` from the logistic relative-eps denominator.
#[test]
fn test_initial_score_zero() {
    let centers = vec![2.0, 0.0, -3.5];
    let model = QuadraticModel::new(3, centers.clone(), (-10.0, 10.0));
    let opt = LogisticAnnealingOptimizer::new(500, 10, 100, 1e1);
    let initial_score = NotNan::new(0.0).unwrap();
    let (final_solution, final_score) = opt
        .run(
            &model,
            Some((centers, initial_score)),
            1000,
            Duration::from_secs(10),
        )
        .unwrap();
    assert!(final_score.into_inner().is_finite());
    assert_abs_diff_eq!(0.0, final_solution[0] - 2.0, epsilon = 0.05);
    assert_abs_diff_eq!(0.0, final_solution[1] - 0.0, epsilon = 0.05);
    assert_abs_diff_eq!(0.0, final_solution[2] - -3.5, epsilon = 0.05);
}
