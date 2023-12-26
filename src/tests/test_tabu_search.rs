use std::time::Duration;

use approx::assert_abs_diff_eq;

use crate::{
    optim::{LocalSearchOptimizer, TabuList, TabuSearchOptimizer},
    utils::RingBuffer,
};

use super::{QuadraticModel, SolutionType, TransitionType};

#[derive(Debug)]
struct MyTabuList {
    buff: RingBuffer<TransitionType>,
}

impl MyTabuList {
    fn new(size: usize) -> Self {
        let buff = RingBuffer::new(size);
        Self { buff }
    }
}

impl TabuList for MyTabuList {
    type Item = (SolutionType, TransitionType);

    fn contains(&self, item: &Self::Item) -> bool {
        let (k1, _, x) = item.1;
        self.buff
            .iter()
            .any(|&(k2, y, _)| (k1 == k2) && (x - y).abs() < 0.005)
    }

    fn append(&mut self, item: Self::Item) {
        self.buff.append(item.1);
    }
}

#[test]
fn test() {
    let model = QuadraticModel::new(3, vec![2.0, 0.0, -3.5], (-10.0, 10.0));
    let opt = TabuSearchOptimizer::new(1000, 25, 5);
    let tabu_list = MyTabuList::new(10);
    let null_closure = None::<&fn(_)>;
    let (final_solution, final_score, _) = opt.optimize(
        &model,
        None,
        10000,
        Duration::from_secs(10),
        null_closure,
        tabu_list,
    );
    assert_abs_diff_eq!(2.0, final_solution[0], epsilon = 0.1);
    assert_abs_diff_eq!(0.0, final_solution[1], epsilon = 0.1);
    assert_abs_diff_eq!(-3.5, final_solution[2], epsilon = 0.1);
    assert_abs_diff_eq!(0.0, final_score.into_inner(), epsilon = 0.05);
}
