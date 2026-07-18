use std::collections::VecDeque;

/// Sliding window based acceptance counter
#[derive(Debug, Clone)]
pub struct AcceptanceCounter {
    window_size: usize,
    buffer: std::collections::VecDeque<u8>,
    n_accepted: usize,
}

impl Default for AcceptanceCounter {
    fn default() -> Self {
        Self::new(100)
    }
}

impl AcceptanceCounter {
    /// Constructor of AcceptanceCounter
    /// - `window_size` : size of the sliding window
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            n_accepted: 0,
        }
    }

    /// Update the counter with a new result
    pub fn enqueue(&mut self, accepted: bool) {
        if self.window_size == 0 {
            return;
        }
        if self.buffer.len() == self.window_size {
            let old = self.buffer.pop_front().unwrap();
            self.n_accepted -= old as usize;
        }
        let value = accepted as u8;
        self.buffer.push_back(value);
        self.n_accepted += value as usize;
    }

    /// Get the acceptance ratio
    pub fn acceptance_ratio(&self) -> f64 {
        let total = self.buffer.len();
        if total == 0 {
            0.0
        } else {
            self.n_accepted as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AcceptanceCounter;

    #[test]
    fn window_zero_enqueue_is_noop_and_does_not_panic() {
        let mut c = AcceptanceCounter::new(0);

        c.enqueue(true);
        c.enqueue(false);

        assert_eq!(c.acceptance_ratio(), 0.0);
    }

    #[test]
    fn sliding_window_behavior_unchanged() {
        let mut c = AcceptanceCounter::new(2);

        c.enqueue(true);
        c.enqueue(true);
        assert_eq!(c.acceptance_ratio(), 1.0);

        c.enqueue(false);
        assert_eq!(c.acceptance_ratio(), 0.5);

        c.enqueue(false);
        assert_eq!(c.acceptance_ratio(), 0.0);
    }
}
