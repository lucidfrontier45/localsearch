//! Callback of Optimization Iteration

use std::{cell::RefCell, rc::Rc};

/// OptProgress expresses Optimization Progress that is passed to a [`OptCallbackFn`]
#[derive(Debug, Clone)]
pub struct OptProgress<S, SC> {
    /// current iteration step
    pub iter: usize,
    /// number of accepted transitions
    pub accepted_count: usize,
    /// current best solution
    pub solution: Rc<RefCell<S>>,
    /// current best score
    pub score: SC,
}

impl<S, SC: Ord> OptProgress<S, SC> {
    /// constuctor of OptProgress
    pub fn new(iter: usize, accepted_count: usize, solution: Rc<RefCell<S>>, score: SC) -> Self {
        Self {
            iter,
            accepted_count,
            solution,
            score,
        }
    }
}

/// OptCallbackFn is a trait of a callback function for optimization
/// Typical usage is to show progress bar and save current result to the file
///
/// Example
///
/// ```rust
/// let pb = ProgressBar::new(n_iter);
/// pb.set_style(
///        ProgressStyle::default_bar()
///             .template(
///                 "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (eta={eta}) {msg} ",
///             ).unwrap()
///             .progress_chars("#>-")
///     );
///     pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
///     pb
/// };
/// let mut callback = |op: OptProgress<SolutionType, ScoreType>| {
///     let ratio = op.accepted_count as f64 / op.iter as f64;
///     pb.set_message(format!(
///         "best score {:.4e}, count = {}, acceptance ratio {:.2e}",
///         op.score.into_inner(),
///         op.accepted_count,
///         ratio
///     ));
///     pb.set_position(op.iter as u64);
/// };
/// ```
pub trait OptCallbackFn<S, SC: PartialOrd>: FnMut(OptProgress<S, SC>) {}

impl<F, S, SC: PartialOrd> OptCallbackFn<S, SC> for F where F: FnMut(OptProgress<S, SC>) {}
