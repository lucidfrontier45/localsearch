//! Utilities

use std::collections::VecDeque;

/// RingBuffer to be used to implement a Tabu List
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    capacity: usize,
    buff: VecDeque<T>,
}

impl<T> RingBuffer<T> {
    /// Constructor of RingBuffer
    pub fn new(capacity: usize) -> Self {
        let buff = VecDeque::with_capacity(capacity);
        Self { capacity, buff }
    }

    /// Append a new item to the buffer
    pub fn append(&mut self, item: T) {
        if self.buff.len() == self.capacity {
            self.buff.pop_front();
        }
        self.buff.push_back(item);
    }

    /// Convert to an iterator
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buff.iter()
    }
}
