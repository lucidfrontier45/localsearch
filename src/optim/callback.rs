use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub struct OptProgress<S> {
    pub iter: usize,
    pub accepted_count: usize,
    pub state: Rc<RefCell<S>>,
    pub score: f64,
}

impl<S> OptProgress<S> {
    pub fn new(iter: usize, accepted_count: usize, state: Rc<RefCell<S>>, score: f64) -> Self {
        Self {
            iter,
            accepted_count,
            state,
            score,
        }
    }
}

pub trait OptCallbackFn<S>: Fn(OptProgress<S>) {}

impl<T: Fn(OptProgress<S>), S> OptCallbackFn<S> for T {}
