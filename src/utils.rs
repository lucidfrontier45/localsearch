use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    capacity: usize,
    buff: VecDeque<T>,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let buff = VecDeque::with_capacity(capacity);
        Self { capacity, buff }
    }

    pub fn append(&mut self, item: T) {
        if self.buff.len() == self.capacity {
            self.buff.pop_front();
        }
        self.buff.push_back(item);
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<T> {
        self.buff.iter()
    }
}
