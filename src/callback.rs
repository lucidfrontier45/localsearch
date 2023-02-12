//! Callback of Optimization Iteration

use std::{cell::RefCell, rc::Rc};

/// OptProgress expresses Optimization Progress that is passed to a OptCallbackFn
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
pub trait OptCallbackFn<S, SC: Ord>: Fn(OptProgress<S, SC>) {}

impl<T: Fn(OptProgress<S, SC>), S, SC: Ord> OptCallbackFn<S, SC> for T {}
