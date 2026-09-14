use auto_impl::auto_impl;
use rayon::prelude::*;

use crate::LocalsearchError;

/// OptModel is a trait that defines requirements to be used with optimization algorithm
#[auto_impl(&, Box, Rc, Arc)]
pub trait OptModel: Sync + Send {
    /// Type of the Score
    type ScoreType: Ord + Copy + Sync + Send;
    /// Type of the Solution
    type SolutionType: Clone + Sync + Send;
    /// Type of the Transition
    type TransitionType: Clone + Sync + Send;

    /// Randomly generate a solution
    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError>;

    /// Generate a new trial solution from current solution.
    ///
    /// This method is mandatory for compatibility. A batch-native model may
    /// provide a documented panic here when it is used exclusively through
    /// [`crate::optim::DefaultTrialGenerator`] or another batch consumer;
    /// direct callers remain responsible for avoiding that unsupported path.
    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rng: &mut R,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType);

    /// Generate a batch of trial solutions from the current solution.
    ///
    /// The default implementation runs [`Self::generate_trial_solution`] in
    /// parallel, one time for each RNG in `rngs`. Rayon preserves the input
    /// order in the returned vector. An empty RNG slice returns an empty
    /// vector.
    ///
    /// Batch-oriented models may override this method to use a native batch
    /// implementation. Such models still need to implement the mandatory
    /// single-trial method; built-in optimizers using
    /// [`crate::optim::DefaultTrialGenerator`] call only this batch method.
    fn generate_trial_solutions<R: rand::Rng + Send>(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
        rngs: &mut [R],
    ) -> Vec<(Self::SolutionType, Self::TransitionType, Self::ScoreType)> {
        rngs.par_iter_mut()
            .map(|rng| self.generate_trial_solution(current_solution.clone(), current_score, rng))
            .collect()
    }

    /// Preprocess the solution
    fn preprocess_solution(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
    ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
        Ok((current_solution, current_score))
    }

    /// Postprocess the solution
    fn postprocess_solution(
        &self,
        current_solution: Self::SolutionType,
        current_score: Self::ScoreType,
    ) -> (Self::SolutionType, Self::ScoreType) {
        (current_solution, current_score)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rand::{RngExt as _, SeedableRng as _, rngs::StdRng};

    use super::*;

    struct CountingModel {
        single_calls: AtomicUsize,
    }

    impl OptModel for CountingModel {
        type ScoreType = u64;
        type SolutionType = u64;
        type TransitionType = ();

        fn generate_random_solution<R: rand::Rng>(
            &self,
            _rng: &mut R,
        ) -> Result<(Self::SolutionType, Self::ScoreType), LocalsearchError> {
            Ok((0, 0))
        }

        fn generate_trial_solution<R: rand::Rng>(
            &self,
            _current_solution: Self::SolutionType,
            _current_score: Self::ScoreType,
            rng: &mut R,
        ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
            self.single_calls.fetch_add(1, Ordering::Relaxed);
            let value = rng.random();
            (value, (), value)
        }
    }

    #[test]
    fn default_batch_delegates_in_rng_order() {
        let seeds = [11, 22, 33, 44];
        let expected_model = CountingModel {
            single_calls: AtomicUsize::new(0),
        };
        let mut expected_rngs: Vec<_> = seeds.iter().copied().map(StdRng::seed_from_u64).collect();
        let expected: Vec<_> = expected_rngs
            .iter_mut()
            .map(|rng| expected_model.generate_trial_solution(0, 0, rng))
            .collect();

        let model = CountingModel {
            single_calls: AtomicUsize::new(0),
        };
        let mut rngs: Vec<_> = seeds.iter().copied().map(StdRng::seed_from_u64).collect();
        let actual = model.generate_trial_solutions(0, 0, &mut rngs);

        assert_eq!(actual, expected);
        assert_eq!(model.single_calls.load(Ordering::Relaxed), seeds.len());
    }

    #[test]
    fn default_batch_accepts_empty_rngs() {
        let model = CountingModel {
            single_calls: AtomicUsize::new(0),
        };
        let mut rngs: Vec<StdRng> = Vec::new();

        let actual = model.generate_trial_solutions(0, 0, &mut rngs);

        assert!(actual.is_empty());
        assert_eq!(model.single_calls.load(Ordering::Relaxed), 0);
    }
}
