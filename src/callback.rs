//! Callback of Optimization Iteration

use std::{cell::RefCell, rc::Rc};

/// OptProgress expresses Optimization Progress that is passed to a [`OptCallbackFn`]
#[derive(Debug, Clone)]
pub struct OptProgress<S, SC> {
    /// current iteration step
    pub iter: usize,
    /// number of accepted transitions
    pub accepted_count: usize,
    /// current best state
    pub state: Rc<RefCell<S>>,
    /// current best score
    pub score: SC,
}

impl<S, SC: Ord> OptProgress<S, SC> {
    /// constuctor of OptProgress
    pub fn new(iter: usize, accepted_count: usize, state: Rc<RefCell<S>>, score: SC) -> Self {
        Self {
            iter,
            accepted_count,
            state,
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
/// let callback = |op: OptProgress<StateType, ScoreType>| {
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
pub trait OptCallbackFn<S, SC: PartialOrd>: Fn(OptProgress<S, SC>) {}

impl<T: Fn(OptProgress<S, SC>), S, SC: PartialOrd> OptCallbackFn<S, SC> for T {}
