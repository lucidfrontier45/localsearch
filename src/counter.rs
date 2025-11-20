/// Sliding window based acceptance counter
#[derive(Debug, Clone, Copy)]
pub struct AcceptanceCounter {
    window_size: usize,
    current_sample_count: usize,
    accepted_count: usize,
    last_result_is_accepted: bool,
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
            current_sample_count: 0,
            accepted_count: 0,
            last_result_is_accepted: false,
        }
    }

    /// Update the counter with a new result
    pub fn enqueue(&mut self, accepted: bool) {
        // if current_sample_count < window_size, just add the new result
        if self.current_sample_count < self.window_size {
            self.current_sample_count += 1;
            if accepted {
                self.accepted_count += 1;
            }
            self.last_result_is_accepted = accepted;
        } else {
            // if the window is full, remove the oldest result and add the new result
            if self.last_result_is_accepted {
                self.accepted_count -= 1;
            }
            if accepted {
                self.accepted_count += 1;
            }
            self.last_result_is_accepted = accepted;
        }
    }

    /// Get the acceptance ratio
    pub fn acceptance_ratio(&self) -> f64 {
        if self.current_sample_count == 0 {
            0.0
        } else {
            self.accepted_count as f64 / self.current_sample_count as f64
        }
    }
}
