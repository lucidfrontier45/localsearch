use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub struct OptProgress<S, SC> {
    pub iter: usize,
    pub accepted_count: usize,
    pub state: Rc<RefCell<S>>,
    pub score: SC,
}

impl<S, SC: Ord> OptProgress<S, SC> {
    pub fn new(iter: usize, accepted_count: usize, state: Rc<RefCell<S>>, score: SC) -> Self {
        Self {
            iter,
            accepted_count,
            state,
            score,
        }
    }
}

pub trait OptCallbackFn<S, SC: Ord>: Fn(OptProgress<S, SC>) {}

impl<T: Fn(OptProgress<S, SC>), S, SC: Ord> OptCallbackFn<S, SC> for T {}
