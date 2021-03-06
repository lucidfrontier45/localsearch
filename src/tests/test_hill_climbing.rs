use approx::assert_abs_diff_eq;

use crate::optim::HillClimbingOptimizer;

use super::QuadraticModel;

#[test]
fn test() {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
    let opt = HillClimbingOptimizer::new(1000, 10);
    let null_closure = None::<&fn(_, _, _)>;
    let (final_state, final_score) = opt.optimize(&model, None, 10000, null_closure);
    assert_abs_diff_eq!(2.0, final_state[0], epsilon = 0.05);
    assert_abs_diff_eq!(0.0, final_state[1], epsilon = 0.05);
    assert_abs_diff_eq!(-3.5, final_state[2], epsilon = 0.05);
    assert_abs_diff_eq!(0.0, final_score, epsilon = 0.05);
}
